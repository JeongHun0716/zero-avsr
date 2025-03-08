## Dataset Preparation
We propose Mulitlingual Audio-Visual Romanized Corpus (MARC), the Roman transcription labels for 2,916 hours of audiovisual speech data across 82 languages.
All manifests files for training and evaluation are available for download from [this link](https://www.dropbox.com/scl/fi/05hbxmxo0ltu9thpxszn1/manifests.tar.gz?rlkey=befdyzsjy9g7bmg0k41ad90o9&st=j9reloy4&dl=0).

Download the manifests.tar.gz file into the marc folder and extract it, and then please run: ```tar -xzvf manifests.tar.gz```

This will result in the following directory structure:

```
marc/
├── manifest/              
│   ├── stage1/            # Files for av-romanizer training
│   │   ├── all/           # All training/test files for stage1
│   │   └── zero_shot/     # Zero-shot experiment files for stage1
│   └── stage2/            # Files for zero-avsr training and evaluation
└── update_dataset_paths.py   # Script to update placeholders to absolute paths in tsv files
└── avspeech_train_segments.txt   # Metadata file for AVSpeech training segments
```

Before running any training or evaluation, you must update the dataset file paths in the ```tsv files```. These ```tsv files``` contain placeholders (e.g., ```{LRS3_ROOT}```) that need to be replaced with the absolute paths to your local copies of the datasets. The provided script (```update_dataset_paths.py```) automates this process, ensuring that all references in the ```tsv files``` point to the correct locations on your system.

The required datasets are:

* **VoxCeleb2**  
  Download from the [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) website.

* **MuAViC**  
  Download from the [MuAViC](https://github.com/facebookresearch/muavic) dataset page.

* **LRS3**  
  Download from the [LRS3-TED](https://mmai.io/datasets/lip_reading/) dataset page.

* **AVSpeech**  
  Download from the [AVSpeech](https://looking-to-listen.github.io/avspeech/) dataset page.


Once you have downloaded these datasets, you should pre-process every video clip to crop the mouth regions. You can follow the pre-processing instructions provided in [Auto-AVSR](https://github.com/mpc001/auto_avsr/tree/main/preparation).

Note that for the LRS3 and VoxCeleb2 datasets, the facial landmarks are already provided in the Auto-AVSR repository. For MuAViC, facial landmarks can be found at [MuAViC GitHub](https://github.com/facebookresearch/muavic). Therefore, only for the AVSpeech dataset, you will need to extract the landmarks yourself.



After the pre-processing, update the ```tsv files``` with the absolute paths to the dataset directories using the provided script. This ensures that all dataset references point to the correct locations on your system.


```bash
python update_dataset_paths.py --input_dir ./ --vox2 'path for the VoxCeleb2 dataset' --muavic 'path for the MuAViC dataset' --lrs3 'path for the LRS3 dataset' --avs 'path for the AVSpeech dataset'
```

For example:
```bash
python update_dataset_paths.py --input_dir ./ --vox2 /Dataset/vox2 --muavic /Dataset/muavic --lrs3 /Dataset/lrs3 --avs /Dataset/avs
```

The above command updates the placeholder paths in the ```tsv files``` to your absolute dataset paths.

Each ```tsv files``` contains one line per data sample, with the following fields separated by a tab ```(\t)```:

* **language**
* **video_path**
* **audio_path**
* **num_video_frames**
* **num_audio_frames**    

Below are the expected directory structures for each dataset:

### LRS3
```
lrs3/
├── lrs3_video_seg24s/              
│   ├── pretrain/
│   ├── test/
│   ├── trainval/            
└── lrs3_text_seg24s/
    ├── pretrain/
    ├── test/
    └── trainval/    
```


### VoxCeleb2
```
vox2/
├── audio/              
│   ├── dev_seg24s/
│   ├── test_seg24s/            
└── video/
    ├── dev_seg24s/
    └── test_seg24s/    
```

### MuAViC
```
muavic/
└── {lang}/   # {lang} ∈ {ar, de, el, es, fr, it, pt, ru}
    ├── audio/
    │   ├── train_seg24s/
    │   ├── valid_seg24s/
    │   └── test_seg24s/
    └── video/
        ├── train_seg24s/
        ├── valid_seg24s/
        └── test_seg24s/
```

### AVSpeech
```
avs/
├── audio/              
│   ├── test/
│   ├── train/            
└── video/
    ├── test/
    └── train/    
```

