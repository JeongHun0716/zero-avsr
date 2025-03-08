## Dataset
We propose Mulitlingual Audio-Visual Romanized Corpus (MARC), the Roman transcription labels for 2,916 hours of audiovisual speech data across 82 languages.
All manifests files for training and evaluation are available for download from [this link](https://www.dropbox.com/scl/fi/9nksl4rgykxd9m8pi5kds/manifests.tar.gz?rlkey=f6ri1l2qezays4e5ja0ntc7yc&st=1k2lyjpq&dl=0).

Download the manifests.tar.gz file into the marc folder and extract it, and then please run: ```tar -xzvf manifests.tar.gz```

This will result in the following directory structure:

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

Then, you should update the ```.tsv files``` with the absolute paths to the dataset directories using the provided script. This ensures that all dataset references point to the correct locations on your system.

```bash
python update_dataset_paths.py --input_dir ./ --vox2 'path for the VoxCeleb2 dataset' --muavic 'path for the MuAViC dataset' --lrs3 'path for the LRS3 dataset' --avs 'path for the AVSpeech dataset'
```

For example:
```bash
python update_dataset_paths.py --input_dir ./ --vox2 /Dataset/vox2 --muavic /Dataset/muavic --lrs3 /Dataset/lrs3 --avs /Dataset/avs
```


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


### MuaViC
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

