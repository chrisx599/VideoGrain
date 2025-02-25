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

You may download all data and checkpoints using the following bash command
```bash
bash download_all.sh
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