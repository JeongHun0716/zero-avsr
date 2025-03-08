# Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations

This repository contains the PyTorch implementation of the following paper:
> **Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations**<be>
><br>
> Authors: Jeong Hun Yeo*, Minsu Kim* (*equal contributor), Chae Won Kim, Stavros Petridis, Yong Man Ro<br>


## Introduction
Zero-AVSR is a zero-shot audio-visual speech recognition, which enables speech recognition in target languages without requiring any audio-visual speech data in those languages. 


## Environment Setup
```bash
conda create -n zero-avsr python=3.9 -y
conda activate zero-avsr
git clone https://github.com/JeongHun0716/zero-avsr
cd zero-avsr
```
```bash
# PyTorch and related packages
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.23.5 scipy opencv-python
pip install editdistance python_speech_features einops soundfile sentencepiece tqdm tensorboard unidecode librosa pandas
pip install omegaconf==2.0.6 hydra-core==1.0.7 (If your pip version > 24.1, please run "python3 -m pip install --upgrade pip==24.0")
pip install transformers peft bitsandbytes
cd fairseq
pip install --editable ./
```

## Preparation


## Dataset
We propose Mulitlingual Audio-Visual Romanized Corpus (MARC), the Roman transcription labels for 2,916 hours of audiovisual speech data across 82 languages.

```
marc/
├── manifest/              
│   ├── stage1/            # Files for av-romanizer training
│   │   ├── all/           # All training/test files for stage1
│   │   └── zero_shot/     # Zero-shot experiment files for stage1
│   └── stage2/            # Files for zero-avsr training and evaluation
└── update_dataset_paths.py   # Script to update placeholders to absolute paths in TSV files
└── avspeech_train_segments.txt   # Metadata file for AVSpeech training segments
```

Before running any training or evaluation, you must update the dataset file paths in the TSV files. These TSV files contain placeholders (e.g., ```{LRS3_ROOT}```) that need to be replaced with the absolute paths to your local copies of the datasets. The provided script (```update_dataset_paths.py```) automates this process, ensuring that all references in the TSV files point to the correct locations on your system.

The required datasets are:

* **VoxCeleb2**  
  Download from the [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) website.

* **MuAViC**  
  Download from the [MuAViC](https://github.com/facebookresearch/muavic) dataset page.

* **LRS3**  
  Download from the [LRS3-TED](https://mmai.io/datasets/lip_reading/) dataset page.

* **AVSpeech**  
  Download from the [AVSpeech](https://looking-to-listen.github.io/avspeech/) dataset page.

Once you have downloaded these datasets, update the TSV files with the absolute paths to the dataset directories using the provided script. This ensures that all dataset references point to the correct locations on your system.

```bash
cd marc
python update_dataset_paths.py --input_dir ./ --vox2 'path for the VoxCeleb2 dataset' --muavic 'path for the MuAViC dataset' --lrs3 'path for the LRS3 dataset' --avs 'path for the AVSpeech dataset'
```

For example:
```bash
python update_dataset_paths.py --input_dir ./ --vox2 /Dataset/vox2 --muavic /Dataset/muavic --lrs3 /Dataset/lrs3 --avs /Dataset/avs
```


## Load a pretrained model
### AV-Romanizer
```bash
$ cd stage1
$ python
>>> import fairseq
>>> import model
>>> ckpt_path = "/path/to/the/av-romanizer-checkpoint.pt"
>>> models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
>>> model = models[0]
```


### Zero-AVSR
```bash
$ cd stage2
$ python
>>> import fairseq
>>> import model
>>> ckpt_path = "/path/to/the/zero-avsr-checkpoint.pt"
>>> models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
>>> model = models[0]
```


## Training
### Train a new AV-Romanizer

```bash
bash scripts stage1/train.sh
```

### Evaluation of the AV-Romanizer (Cascaded Zero-AVSR)
To evaluate the performance of Cascaded Zero-AVSR, follow these steps:

1. **Obtain a GPT API Key:**
   Here is an example that runs using GPT, subject to be required payment as per GPT's regulations.<br>
   You need a GPT API key from OpenAI. After generating your API key, open the file:
   ```bash
   stage1/de_romanize_w_gpt_api.py
   ```
   and locate line 15:
   OPENAI_KEY = '<your gpt api key>'
   Replace ```<your gpt api key>``` with your actual API key. This key is required for enabling GPT-based processing in the evaluation pipeline.

3. **Run the Evaluation Script:**  
    Once your API key is set, execute the evaluation script by running:
    ```bash
    bash scripts stage1/eval.sh
    ```
4. **Run the Evaluation Script (under noisy environment):**
    ```bash
    bash scripts stage1/eval_snr.sh
    ```


### Train a new Zero-AVSR

```bash
bash scripts stage2/train.sh
```

### Evaluation of the Zero-AVSR
To evaluate the performance of Zero-AVSR, execute the evaluation script by running:

```bash
bash scripts stage2/eval.sh
```

### Evaluation of the Zero-AVSR, under noisy environment
To evaluate the performance of Zero-AVSR, execute the evaluation script by running:

```bash
bash scripts stage2/eval_snr.sh
```


## Pretrained Models
1. Download the ```AV-HuBERT Large model``` from this [link](https://github.com/facebookresearch/av_hubert) 
2. Download the ```LLaMA-3.2 3B model``` from this [link](https://huggingface.co/meta-llama/Llama-3.2-3B)
3. Download the ```AV-Romanizer model``` from the link below, which can interact with various types of LLMs (e.g., gpt4o-mini, gpt4, llama, etc.) in a cascaded manner.
4. Download the ```Zero-AVSR model``` from the link below, which is built on the LLaMA 3.2 3B model.

> ```AV-Romanizer```

| Model         | Zero-Shot Language  | Training data (# of Languages)  |
|--------------|:----------:|:------------------:|
| [ckpt.pt](https://www.dropbox.com/scl/fi/mph3q2rtgwcb0sn9na47g/checkpoint_best.pt?rlkey=r6t9l2l27cmtgj10yt1uio5mi&st=erysg487&dl=0) |       Arabic(ara)       |        81           | 
| [ckpt.pt](https://www.dropbox.com/scl/fi/5pbsx6n882prjltisxd84/checkpoint_best.pt?rlkey=8ujitzdmhbulb2dman1xi6z9l&st=7adl87e4&dl=0) |        German(deu)            |     81           | 
| [ckpt.pt](https://www.dropbox.com/scl/fi/9euaiuv512bbjb5pj0dgi/checkpoint_best.pt?rlkey=m2z42myzvmmqkupg5757tbp92&st=p0blvuog&dl=0) |        Greek(ell)       | 81           | 
| [ckpt.pt](https://www.dropbox.com/scl/fi/4itcjvwxi96zd48z7k984/checkpoint_best.pt?rlkey=sm0b2yvu9eyp3mvkrpvv9ehnt&st=s16uuqy9&dl=0) |        Spanish(spa)       | 81           | 
| [ckpt.pt](https://www.dropbox.com/scl/fi/i7k9jlk2z36kpbr1yv20s/checkpoint_best.pt?rlkey=txupq7eop1ikgmntak131ldwj&st=akzfr61a&dl=0) |        French(fra)       | 81           | 
| [ckpt.pt](https://www.dropbox.com/scl/fi/y2qwsuftnwz3zbkor1cn8/checkpoint_best.pt?rlkey=0y0ss30zjyfrxrfzrjo5520c5&st=3quyoz2p&dl=0) |        Italian(ita)       | 81           | 
| [ckpt.pt](https://www.dropbox.com/scl/fi/9mdet0q1vbif6gi7z9wgx/checkpoint_best.pt?rlkey=gtwmvivghzql9q2tc7fv0jm3x&st=a8uvdprl&dl=0) |        Portuguese(por)       | 81           | 
| [ckpt.pt](https://www.dropbox.com/scl/fi/cq8v9k6qhejl04pa1qdzr/checkpoint_best.pt?rlkey=arbmj1ui1mmpokykc4stev1mk&st=wjevh56r&dl=0) |        Russian(rus)       | 81           | 
| [ckpt.pt](https://www.dropbox.com/scl/fi/p6i141tfrp0kbqiu1cepy/checkpoint_best.pt?rlkey=hyck9668w9bgx0io2tkc6rdux&st=j37n3fpg&dl=0) |        All      | 82           | 


> ```Zero-AVSR```

| Model         | Zero-Shot Language  | Training data (# of Languages)  |
|--------------|:----------:|:------------------:|
| [ckpt.pt](https://www.dropbox.com/scl/fi/bw0jwqo3widniiv4jm6ug/checkpoint_best.pt?rlkey=1agqfao8rfch2epa9suxfe8g9&st=xpeyc187&dl=0) |       Arabic(ara)       |        81           | 
| [ckpt.pt](https://www.dropbox.com/scl/fi/5yu1cinmwtkq30yjom55w/checkpoint_best.pt?rlkey=die6brzvzb6kwi8qglxwrtw5s&st=2zasun4c&dl=0) |        German(deu)            |     81           | 
| [ckpt.pt](https://www.dropbox.com/scl/fi/0h7chny92u024dhz7ps6n/checkpoint_best.pt?rlkey=hqtumplvoey956xplx1emdkb4&st=reurwhgc&dl=0) |        Greek(ell)       | 81           | 
| [ckpt.pt](https://www.dropbox.com/scl/fi/pn57wyicai0ilsia0bmjl/checkpoint_best.pt?rlkey=fe82psi10aeeypyusgb4r5byz&st=cjw4xbyn&dl=0) |        Spanish(spa)       | 81           | 
| [ckpt.pt](https://www.dropbox.com/scl/fi/02uxoa555flwld4kenqqw/checkpoint_best.pt?rlkey=a9fl6bkkfwr07tktxvzh5thbl&st=99p6wx4t&dl=0) |        French(fra)       | 81           | 
| [ckpt.pt](https://www.dropbox.com/scl/fi/49qndou53mwyz801zcoxq/checkpoint_best.pt?rlkey=qj1k26md93zmgrxp6qcvljdnl&st=lbowscks&dl=0) |        Italian(ita)       | 81           | 
| [ckpt.pt](https://www.dropbox.com/scl/fi/z82fg2jgdniwncjn32mkm/checkpoint_best.pt?rlkey=6yobx01l2tx85gfhg02o3dzlk&st=39ytp0wn&dl=0) |        Portuguese(por)       | 81           | 
| [ckpt.pt](https://www.dropbox.com/scl/fi/3g0kog5f98wo9jey7fbyn/checkpoint_best.pt?rlkey=sp1nt4j6k8tu7oykh3tddnbhx&st=76l2iiim&dl=0) |        Russian(rus)       | 81           | 
| [ckpt.pt](https://www.dropbox.com/scl/fi/4t6b3zrw6d2d3iwfrp37s/checkpoint_best.pt?rlkey=g01nl8yafbigyu5h0kweat106&st=7cj4ggu2&dl=0) |        All      | 82           | 
