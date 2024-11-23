## 🛡 Setup Environment
Our method is tested using cuda12.1, fp16 of accelerator and xformers on a single L40.

```bash
# Step 1: Create and activate Conda environment
conda create -n st-modulator python==3.10 
conda activate st-modulator

# Step 2: Install PyTorch and CUDA
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Step 3: Install additional dependencies with pip
pip install -r requirements.txt
```

`xformers` is recommended to save memory and running time. 

</details>

You may download all data and checkpoints using the following bash command
```
bash download_all.sh
```

## 🔥 ST-Modulator Editing

You could reproduce multi-grained editing results in our teaser by running:

```
bash test.sh 
#or accelerate launch test.py --config config/run_two_man.yaml
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