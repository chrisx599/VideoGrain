# VideoGrain: Modulating Space-Time Attention for Multi-Grained Video Editing (ICLR 2025)
## [<a href="https://knightyxp.github.io/VideoGrain_project_page/" target="_blank">Project Page</a>]

[![arXiv](https://img.shields.io/badge/arXiv-2502.17258-B31B1B.svg)](https://arxiv.org/abs/2502.17258) 
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/papers/2502.17258)
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://knightyxp.github.io/VideoGrain_project_page/)


<table class="center" border="1" cellspacing="0" cellpadding="5">
  <tr>
    <td colspan="2" style="text-align:center;"><img src="assets/teaser/class_level.gif"  style="width:250px; height:auto;"></td>
    <td colspan="2" style="text-align:center;"><img src="assets/teaser/instance_part.gif"  style="width:250px; height:auto;"></td>
    <td colspan="2" style="text-align:center;"><img src="assets/teaser/2monkeys.gif" style="width:250px; height:auto;"></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align:right; width:250px;">class level</td>
    <td colspan="1" style="text-align:center; width:125px;">instance level</td>
    <td colspan="1" style="text-align:center; width:125px;">part level</td>
    <td colspan="2" style="text-align:center; width:250px;">animal instances</td>
  </tr>
  
  <tr>
    <td colspan="2" style="text-align:center;"><img src="assets/teaser/2cats.gif" style="width:250px; height:auto;"></td>
    <td colspan="2" style="text-align:center;"><img src="assets/teaser/soap-box.gif" style="width:250px; height:auto;"></td>
    <td colspan="2" style="text-align:center;"><img src="assets/teaser/man-text-message.gif" style="width:250px; height:auto;"></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align:center; width:250px;">animal instances</td>
    <td colspan="2" style="text-align:center; width:250px;">human instances</td>
    <td colspan="2" style="text-align:center; width:250px;">part-level modification</td>
  </tr>
</table>


<!-- <table class="center">
<td><img src="assets/teaser/multi-grain-demo.gif"></td>
<td><img src="assets/teaser/2monkeys.gif"></td>
<tr>
<td width=16% style="text-align:center;">Multi-Grained Video Editing</td>
<td width=16% style="text-align:center;">Class Level: human class → spiderman</td>
<td width=16% style="text-align:center;">Instance Level: left → Spiderman, right → Polar Bear</td>
<td width=16% style="text-align:center;">Part Level: Polar Bear + Sunglasses</td>
<td width=20% style="text-align:center;">left → teddy bear, right → golden retriever</td>
</tr>
<td><img src="assets/teaser/2cats.gif"></td>
<td><img src="assets/teaser/soap-box.gif"></td>
<td><img src="assets/teaser/man-text-message.gif"></td>
<tr>
<td width=25% style="text-align:center;">left cat→ Samoyed, right cat→ Tiger</td>
<td width=25% style="text-align:center;">behind→ Iron Man, front→ Stormtrooper</td>
<td width=25% style="text-align:center;">half-sleeve gray shirt→ a black sui</td>
</tr>

</table > -->

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

```
mkdir annotator/ckpts
```
Method 1: Download dwpose models 

(Note: if your are avaiable to huggingface, other models like depth_zoe etc can be automatically downloaded)

Download dwpose model dw-ll_ucoco_384.onnx ([baidu](https://pan.baidu.com/s/1nuBjw-KKSxD_BkpmwXUJiw?pwd=28d7), [google](https://drive.google.com/file/d/12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2/view?usp=sharing)) and Det model yolox_l.onnx ([baidu](https://pan.baidu.com/s/1fpfIVpv5ypo4c1bUlzkMYQ?pwd=mjdn), [google](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view?usp=sharing)), 
Then put them into ./annotator/ckpts. 

Method 2: Download all annotator checkpoints from google or baiduyun (when can not access to huggingface) 

If you cannot access HuggingFace, you can download all the annotator checkpoints (such as DW-Pose, depth_zoe, depth_midas, and OpenPose, cost around 4G.) from [baidu](https://pan.baidu.com/s/1sgBFLFkdTCDTn4oqHjGb9A?pwd=pdm5) or [google](https://drive.google.com/file/d/1qOsmWshnFMMr8x1HteaTViTSQLh_4rle/view?usp=drive_link)
Then extract them into ./annotator/ckpts


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