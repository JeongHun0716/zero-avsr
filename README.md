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
| [ckpt.pt]() |       Arabic(ara)       |        81           | 
| [ckpt.pt]() |        German(deu)            |     81           | 
| [ckpt.pt]() |        Greek(ell)       | 81           | 
| [ckpt.pt]() |        Spanish(spa)       | 81           | 
| [ckpt.pt]() |        French(fra)       | 81           | 
| [ckpt.pt]() |        Italian(ita)       | 81           | 
| [ckpt.pt]() |        Portuguese(por)       | 81           | 
| [ckpt.pt]() |        Russian(rus)       | 81           | 
| [ckpt.pt]() |        All      | 82           | 


> ```Zero-AVSR```

| Model         | Zero-Shot Language  | Training data (# of Languages)  |
|--------------|:----------:|:------------------:|
| [ckpt.pt]() |       Arabic(ara)       |        81           | 
| [ckpt.pt]() |        German(deu)            |     81           | 
| [ckpt.pt]() |        Greek(ell)       | 81           | 
| [ckpt.pt]() |        Spanish(spa)       | 81           | 
| [ckpt.pt]() |        French(fra)       | 81           | 
| [ckpt.pt]() |        Italian(ita)       | 81           | 
| [ckpt.pt]() |        Portuguese(por)       | 81           | 
| [ckpt.pt]() |        Russian(rus)       | 81           | 
| [ckpt.pt]() |        All      | 82           | 
