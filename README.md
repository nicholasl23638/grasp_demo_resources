# grasp_demo_resources

Setup instructions:
1. build docker container with ros2 (I recommend emiliano's ros-docker container setup. Build command: ```make humble.build.cuda cuda-image="devel" cuda-version="12.1"```)
2. clone this repo
3. clone emiliano's anygrasp repo (will upload link)
4. clone lingbot-depth ```https://github.com/Robbyant/lingbot-depth.git```
5. install and setup a venv environment (python3 -m venv my_env)

## python installation
```
source my_env/bin/activate
cd graspnet-baseline/
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
cd pointnet2
python setup.py install
cd ..
cd knn
python setup.py install
cd ..
cd graspnetAPI
python -m pip pip install .
cd ../..

cd lingbot-depth/
python -m pip install -e .
python -m pip install xformers==0.0.28.post3
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
cd ..
```

Running demo:
```
python merge_test.py
```
merge_test.py will pull intrinsics & input photo data from a target directory (specified at bottom of file) and treat depth data with lingbot, then plug treated depth into anygrasp demo for visualization. 

- Raw Depth: raw_depth.png
- Treated depth: depth.png
- Color: color.png
