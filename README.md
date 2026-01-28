# Steps to run

1. **Download the dataset**
```
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
unzip tandt_db.zip
```

2. **Launch and connect to the container**
```
docker run --shm-size=128GB --device=/dev/kfd --device=/dev/dri --group-add video -it -v "$(pwd)/tandt_db:/tandt_db" --name rocm_pytorch rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0
```

> Following steps are inside the container.

3. **Install rocm/gsplat and other packages**

```
pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-7.0.0/simple/
pip install --no-build-isolation git+https://github.com/amd-wangfan/simple-knn.git@hip_support
pip install opencv-python plyfile
```

4. **Clone this repo**
```
git clone https://github.com/yudigege86/gaussian-splatting.git
cd gaussian-splatting
```

5. **Run training**
```
python train.py -s /tandt_db/tandt/truck -m output/truck --checkpoint_iterations 30000
```

6. **Render images**
```
python render.py -s /tandt_db/tandt/truck -m output/truck
```
