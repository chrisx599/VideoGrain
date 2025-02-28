# VideoGrain: Modulating Space-Time Attention for Multi-Grained Video Editing (ICLR 2025)
## [<a href="https://knightyxp.github.io/VideoGrain_project_page/" target="_blank">Project Page</a>]

[![arXiv](https://img.shields.io/badge/arXiv-2502.17258-B31B1B.svg)](https://arxiv.org/abs/2502.17258) 
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/papers/2502.17258)
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://knightyxp.github.io/VideoGrain_project_page/)

## ▶️ Setup Environment
Our method is tested using cuda12.1, fp16 of accelerator and xformers on a single L40.

```bash
# Step 1: Create and activate Conda environment
conda create -n videograin python==3.10 
conda activate videograin

# Step 2: Install PyTorch, CUDA and Xformers
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install --pre -U xformers==0.0.27
# Step 3: Install additional dependencies with pip
pip install -r requirements.txt
```

`xformers` is recommended to save memory and running time. 

</details>

You may download all the base model checkpoints using the following bash command
```bash
## download sd 1.5, controlnet depth/pose v10/v11
bash download_all.sh
```

Prepare ControlNet annotator weights (e.g., DW-Pose, depth_zoe, depth_midas, OpenPose)

```bash
## Switch to the ckpts directory
cd ckpts

Method 1: Download individual models
Download the DW-Pose models (dw-ll_ucoco_384.onnx and yolo_l.onnx), as we found them to be more robust than OpenPose. (Note: Other models, such as depth_zoe, depth_midas, and OpenPose, can be automatically downloaded from HuggingFace.)
Available from:
  - [Baidu](https://pan.baidu.com/s/1nuBjw-KKSxD_BkpmwXUJiw?pwd=28d7)
  - [Google](https://drive.google.com/file/d/12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2/view?usp=sharing)
Download the detection model (yolox_l.onnx) from:
  - [Baidu](https://pan.baidu.com/s/1fpfIVpv5ypo4c1bUlzkMYQ?pwd=mjdn)
  - [Google](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view?usp=sharing)
Then place both files into ./annotator/ckpts

Method 2: Download all annotator checkpoints in one package
# Note: This package includes all the required annotator models, such as DW-Pose, depth_zoe, depth_midas, and OpenPose. Requires approximately 4GB of storage space.
If you cannot access HuggingFace, you can download all the annotator checkpoints from:
  - [BaiduYun](https://pan.baidu.com/s/1sgBFLFkdTCDTn4oqHjGb9A?pwd=pdm5)
  - [Google](https://drive.google.com/file/d/1qOsmWshnFMMr8x1HteaTViTSQLh_4rle/view?usp=drive_link)
Then extract them into ./annotator/ckpts

```

## 🔛 Prepare all the data

```
gdown https://drive.google.com/file/d/1dzdvLnXWeMFR3CE2Ew0Bs06vyFSvnGXA/view?usp=drive_link
tar -zxvf videograin_data.tar.gz
```

## 🔥 VideoGrain Editing

You could reproduce multi-grained editing results in our teaser by running:

```bash
bash test.sh 
#or accelerate launch test.py --config config/instance_level/running_two_man/running_3cls_polar_spider_vis_weight.yaml
```

<details><summary>The result is saved at `./result` . (Click for directory structure) </summary>

```
result
├── run_two_man
│   ├── infer_samples
│   ├── sample
│           ├── step_0         # result image folder
│           ├── step_0.mp4       # result video
│           ├── source_video.mp4    # the input video

```

</details>