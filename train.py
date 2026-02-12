#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import torch
import torch.distributed as dist
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        world_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, world_rank, world_size, local_rank
    return False, 0, 1, 0


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_sync_model_path(args, is_distributed: bool, world_rank: int):
    if not is_distributed or args.model_path:
        return
    if world_rank == 0:
        model_path = os.path.join("./output/", str(uuid.uuid4())[0:10])
    else:
        model_path = ""
    obj_list = [model_path]
    dist.broadcast_object_list(obj_list, src=0)
    args.model_path = obj_list[0]


def _gather_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    device = tensor.device
    local_n = torch.tensor([tensor.shape[0]], device=device, dtype=torch.int64)
    sizes = [torch.zeros_like(local_n) for _ in range(world_size)]
    dist.all_gather(sizes, local_n)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes)
    if tensor.shape[0] < max_n:
        pad = torch.zeros(
            (max_n - tensor.shape[0], *tensor.shape[1:]),
            device=device,
            dtype=tensor.dtype,
        )
        padded = torch.cat([tensor, pad], dim=0)
    else:
        padded = tensor
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    sliced = [gathered[i][: sizes[i]] for i in range(world_size)]
    return torch.cat(sliced, dim=0)


def save_checkpoint_ddp(gaussians, iteration: int, model_path: str, world_rank: int, world_size: int):
    if world_size == 1:
        torch.save((gaussians.capture(), iteration), f"{model_path}/chkpnt{iteration}.pth")
        return

    xyz = _gather_tensor(gaussians._xyz, world_size)
    features_dc = _gather_tensor(gaussians._features_dc, world_size)
    features_rest = _gather_tensor(gaussians._features_rest, world_size)
    scaling = _gather_tensor(gaussians._scaling, world_size)
    rotation = _gather_tensor(gaussians._rotation, world_size)
    opacity = _gather_tensor(gaussians._opacity, world_size)
    max_radii2d = _gather_tensor(gaussians.max_radii2D, world_size)
    xyz_grad_accum = _gather_tensor(gaussians.xyz_gradient_accum, world_size)
    denom = _gather_tensor(gaussians.denom, world_size)

    if world_rank == 0:
        model_params = (
            gaussians.active_sh_degree,
            xyz,
            features_dc,
            features_rest,
            scaling,
            rotation,
            opacity,
            max_radii2d,
            xyz_grad_accum,
            denom,
            None,
            gaussians.spatial_lr_scale,
        )
        torch.save((model_params, iteration), f"{model_path}/chkpnt{iteration}.pth")


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    world_rank: int = 0,
    world_size: int = 1,
    distributed: bool = False,
):

    first_iter = 0
    if distributed:
        # Keep camera shuffling consistent across ranks.
        seed_all(0)
    tb_writer = prepare_output_and_logger(dataset) if world_rank == 0 else None
    gaussians = GaussianModel(
        dataset.sh_degree, opt.optimizer_type, world_rank=world_rank, world_size=world_size
    )
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    # Reseed per rank for training-time randomness.
    seed_all(1 + world_rank)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam"
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = (
        tqdm(range(first_iter, opt.iterations), desc="Training progress")
        if world_rank == 0
        else range(first_iter, opt.iterations)
    )
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if world_rank == 0:
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(
                            custom_cam,
                            gaussians,
                            pipe,
                            background,
                            scaling_modifier=scaling_modifer,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=False,
                            packed=use_sparse_adam,
                            sparse_grad=use_sparse_adam,
                            distributed=False,
                        )["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=False,
            packed=use_sparse_adam,
            sparse_grad=use_sparse_adam,
            distributed=distributed,
        )
        image = render_pkg["render"]
        render_info = render_pkg["info"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        if gaussians.strategy is not None:
            gaussians.strategy.step_pre_backward(
                gaussians.splats,
                gaussians.optimizers,
                gaussians.strategy_state,
                iteration,
                render_info,
            )

        loss.backward()

        if gaussians.strategy is not None:
            gaussians.strategy.step_post_backward(
                gaussians.splats,
                gaussians.optimizers,
                gaussians.strategy_state,
                iteration,
                render_info,
                packed=use_sparse_adam,
            )
            gaussians._sync_from_splats()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if world_rank == 0:
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                Ll1depth,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background, 1.0, False, None, dataset.train_test_exp, use_sparse_adam, use_sparse_adam, distributed),
                dataset.train_test_exp,
                world_rank=world_rank,
                world_size=world_size,
                distributed=distributed,
            )
            if iteration in saving_iterations:
                if world_rank == 0:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                for optimizer in gaussians.optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                if world_rank == 0:
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                save_checkpoint_ddp(
                    gaussians,
                    iteration,
                    scene.model_path,
                    world_rank=world_rank,
                    world_size=world_size,
                )

    if gaussians.strategy is not None and hasattr(gaussians.strategy, "split_call_count"):
        print(f"\nSplit called {gaussians.strategy.split_call_count} times.")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, Ll1depth, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, world_rank: int = 0, world_size: int = 1, distributed: bool = False):
    if tb_writer and world_rank == 0:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/depth_loss', Ll1depth, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        if distributed:
            dist.barrier()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                if world_rank == 0:
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer and world_rank == 0:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer and world_rank == 0:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        if distributed:
            dist.barrier()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    is_distributed, world_rank, world_size, local_rank = init_distributed()
    if is_distributed:
        # safe_state pins cuda:0; reset to local rank device
        torch.cuda.set_device(local_rank)
        if world_rank != 0:
            args.disable_viewer = True
        maybe_sync_model_path(args, is_distributed, world_rank)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        world_rank=world_rank,
        world_size=world_size,
        distributed=is_distributed,
    )

    # All done
    print("\nTraining complete.")
