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
    <!-- <td colspan="1" style="text-align:right; width:125px;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td> -->
    <td colspan="2" style="text-align:right; width:250px;"> class level</td>
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

## 📀 Demo Video
https://github.com/user-attachments/assets/dc54bc11-48cc-4814-9879-bf2699ee9d1d


## 📣 News
* **[2025/2/25]**  Our VideoGrain is posted and recommended  by Gradio on [LinkedIn](https://www.linkedin.com/posts/gradio_just-dropped-videograin-a-new-zero-shot-activity-7300094635094261760-hoiE) and [Twitter](https://x.com/Gradio/status/1894328911154028566), and recommended by [AK](https://x.com/_akhaliq/status/1894254599223017622).
* **[2025/2/25]**  Our VideoGrain is submited by AK to [HuggingFace-daily papers](https://huggingface.co/papers?date=2025-02-25), and rank [#1](https://huggingface.co/papers/2502.17258) paper of that day.
* **[2025/2/24]**  We release our paper on [arxiv](https://arxiv.org/abs/2502.17258), we also release [code](https://github.com/knightyxp/VideoGrain) and [full-data](https://drive.google.com/file/d/1dzdvLnXWeMFR3CE2Ew0Bs06vyFSvnGXA/view?usp=drive_link) on google drive.
* **[2025/1/23]**  Our paper is accepted to [ICLR2025](https://openreview.net/forum?id=SSslAtcPB6)! Welcome to **watch** 👀 this repository for the latest updates.


## 🍻 Setup Environment
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

<details><summary>Click for ControlNet annotator weights (if you can not access to huggingface)</summary>

You can download all the annotator checkpoints (such as DW-Pose, depth_zoe, depth_midas, and OpenPose, cost around 4G.) from [baidu](https://pan.baidu.com/s/1sgBFLFkdTCDTn4oqHjGb9A?pwd=pdm5) or [google](https://drive.google.com/file/d/1qOsmWshnFMMr8x1HteaTViTSQLh_4rle/view?usp=drive_link)
Then extract them into ./annotator/ckpts

</details>

## 🔛 Prepare all the data

```
gdown https://drive.google.com/file/d/1dzdvLnXWeMFR3CE2Ew0Bs06vyFSvnGXA/view?usp=drive_link
tar -zxvf videograin_data.tar.gz
```

## 🔥 VideoGrain Editing

### Inference
VideoGrain is a training-free framework. To run the inference script, use the following command:

```bash
bash test.sh 
or accelerate launch test.py --config config/part_level/adding_new_object/run_two_man/running_spider_polar_sunglass.yaml
```

<details><summary>The result is saved at `./result` . (Click for directory structure) </summary>

```
result
├── run_two_man
│   ├── control                # control conditon 
│   ├── infer_samples
│           ├── input             # the input video frames
│           ├── masked_video.mp4    # check whether edit regions are accuratedly covered
│   ├── sample
│           ├── step_0                  # result image folder
│           ├── step_0.mp4              # result video
│           ├── source_video.mp4        # the input video
│           ├── visualization_denoise   # cross attention weight
│           ├── sd_study                # cluster inversion feature
```

</details>
