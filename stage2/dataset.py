import itertools
import logging
import os
import sys
import time
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from python_speech_features import logfbank
from scipy.io import wavfile
from transformers import AutoTokenizer
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
import json

import soundfile as sf 


from . import utils as custom_utils
from avhubert.hubert_dataset import load_label, load_label_offset, verify_label_lengths, AVHubertDataset


logger = logging.getLogger(__name__)




def load_audio_visual(manifest_path, max_keep, min_keep, frame_rate, label_paths, label_rates, tol=0.1):
    def is_audio_label_aligned(audio_dur, label_durs):
        return all([abs(audio_dur - label_dur)<tol for label_dur in label_durs])

    n_long, n_short, n_unaligned = 0, 0, 0
    names, inds, sizes = [], [], []
    dur_from_label_list = []
    is_seq_label = any([x==-1 for x in label_rates])
    for label_path, label_rate in zip(label_paths, label_rates):
        label_lengths = [len(line.rstrip().split())/label_rate for line in open(label_path).readlines()]
        dur_from_label_list.append(label_lengths)
    dur_from_label_list = list(zip(*dur_from_label_list))

    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            sz = int(items[-2]) # 
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            elif (not is_seq_label) and (not is_audio_label_aligned(sz/frame_rate, dur_from_label_list[ind])):
                n_unaligned += 1
            else:
                lang = items[0]
                video_path = items[1]
                audio_path = items[2]

                names.append((lang, video_path, audio_path))
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long and {n_unaligned} unaligned,"
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes


class zero_avsr_dataset(FairseqDataset):
    def __init__(
            self,
            manifest_path: str,
            sample_rate: float,
            llm_path: str,
            label_paths: List[str],
            label_rates: Union[List[float], float],  # -1 for sequence labels
            label_processors: Optional[List[Any]] = None,
            max_keep_sample_size: Optional[int] = None,
            min_keep_sample_size: Optional[int] = None,
            max_sample_size: Optional[int] = None,
            shuffle: bool = True,
            pad_audio: bool = False,
            normalize: bool = False,
            store_labels: bool = True,
            random_crop: bool = False,
            single_target: bool = False,
            stack_order_audio: int=1,
            skip_verify: bool=False,
            image_mean: float=0,
            image_std: float=1,
            image_crop_size: int=88,
            image_aug: bool=False,
            modalities: Optional[List[str]]=None,
            is_s2s=False,
            noise_fn=None,
            noise_prob=0,
            noise_snr=0,
            noise_num=1,
            time_mask: bool = False,
            random_erase: bool = False,
            zero_shot: bool = True,
            zero_shot_lang: Optional[str] = None,
    ):
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, int)
            else label_rates
        )

        self.modalities = set(modalities)
        self.audio_root, self.names, inds, tot, self.sizes = load_audio_visual(manifest_path,
                                                                               max_keep_sample_size,
                                                                               min_keep_sample_size,
                                                                               frame_rate=sample_rate,
                                                                               label_paths=label_paths,
                                                                               label_rates=self.label_rates,
                                                                               )
        
        self.zero_shot = zero_shot
        self.zero_shot_lang = zero_shot_lang
        self.sample_rate = sample_rate
        self.stack_order_audio = stack_order_audio
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        with open(f'{root_dir}/marc/manifest/languages.json', 'r') as f:
            self.lang_dict = json.load(f)
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id
        self.llm_tokenizer.sep_token_id = self.llm_tokenizer.unk_token_id
        
        self.single_target = single_target
        self.store_labels = store_labels
        self.is_s2s = is_s2s
        
        roman_label_path = [label_paths[0].replace('.wrd','.ltr')] 
        self.roman_label_list = [load_label(p, inds, tot) for p in roman_label_path]
        
        self.noise_wav, self.noise_prob, self.noise_snr, self.noise_num = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else [], noise_prob, noise_snr, noise_num

        assert self.single_target == (self.label_rates[0] == -1), f"single target should be equivalent to sequence label (label_rate==-1)"
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        
        assert(len(self.roman_label_list) == len(self.label_list))
        
        assert (
            label_processors is None
            or len(label_processors) == self.num_labels
        )
        if not skip_verify:
            for label_path, label_rate in zip(label_paths, self.label_rates):
                verify_label_lengths(self.sizes, self.sample_rate, label_path, label_rate, inds, tot)
        else:
            logger.info(f"Skip label alignment verifying")

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        if image_aug:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                custom_utils.HorizontalFlip(0.5),
                custom_utils.Normalize(image_mean, image_std) ]
                + ([custom_utils.RandomErase(0.5)] if random_erase else [])
                + ([custom_utils.TimeMask()] if time_mask else []) )
        else:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                custom_utils.Normalize(image_mean, image_std) ])
        logger.info(f"image transform: {self.transform}")

        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}, "
            f"seqs2seq data={self.is_s2s},")
        logger.info(
            f"Noise wav: {noise_fn}->{len(self.noise_wav)} wav, Prob: {self.noise_prob}, SNR: {self.noise_snr}, Number of mixture: {self.noise_num}"
        )

    def __len__(self):
        return len(self.sizes)
    
    
    def load_feature(self, mix_name):
        """
        Load image and audio feature
        Returns:
        video_feats: numpy.ndarray of shape [T, H, W, 1], audio_feats: numpy.ndarray of shape [T, F]
        """
        def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats
        _, video_fn, audio_fn = mix_name
        if 'video' in self.modalities:
            video_feats = self.load_video(video_fn) # [T, H, W, 1]
        else:
            video_feats = None
        if 'audio' in self.modalities:
            wav_data, sample_rate = sf.read(audio_fn, dtype="float32")
            assert sample_rate == 16_000 and len(wav_data.shape) == 1
            if np.random.rand() < self.noise_prob:
                wav_data = self.add_noise(wav_data)
            audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32) # [T, F]
            audio_feats = stacker(audio_feats, self.stack_order_audio) # [T/stack_order_audio, F*stack_order_audio]
        else:
            audio_feats = None
        if audio_feats is not None and video_feats is not None:
            diff = len(audio_feats) - len(video_feats)
            if diff < 0:
                audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
            elif diff > 0:
                audio_feats = audio_feats[:-diff]
        return video_feats, audio_feats

    def load_video(self, audio_name):
        feats = custom_utils.load_video(os.path.join(self.audio_root, audio_name))
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def select_noise(self):
        rand_indexes = np.random.randint(0, len(self.noise_wav), size=self.noise_num)
        noise_wav = []
        for x in rand_indexes:
            noise_wav.append(sf.read(self.noise_wav[x])[0].astype(np.float32))
        if self.noise_num == 1:
            return noise_wav[0]
        else:
            min_len = min([len(x) for x in noise_wav])
            noise_wav = [x[:min_len] for x in noise_wav]
            noise_wav = np.floor(np.stack(noise_wav).mean(axis=0))
            return noise_wav

    def add_noise(self, clean_wav):
        clean_wav = clean_wav.astype(np.float32)
        noise_wav = self.select_noise()
        if type(self.noise_snr) == int or type(self.noise_snr) == float:
            snr = self.noise_snr
        elif type(self.noise_snr) == tuple:
            snr = np.random.randint(self.noise_snr[0], self.noise_snr[1]+1)
        clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
        if len(clean_wav) > len(noise_wav):
            ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
            noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
        if len(clean_wav) < len(noise_wav):
            start = 0
            noise_wav = noise_wav[start: start + len(clean_wav)]
            
        if noise_wav.sum() == 0:
            return clean_wav
        
        noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
        adjusted_noise_rms = clean_rms / (10**(snr/20))
        adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
        mixed = clean_wav + adjusted_noise_wav

        return mixed
    
    def __getitem__(self, index):
        # self.names -> (lang, vid_pth, aud_pth)
        lang = self.names[index][0]
        video_feats, audio_feats = self.load_feature(self.names[index])
        audio_feats, video_feats = torch.from_numpy(audio_feats.astype(np.float32)) if audio_feats is not None else None, torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None
        if self.normalize and 'audio' in self.modalities:
            with torch.no_grad():
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
                
        labels = [self.llm_tokenizer(self.label_list[0][index], return_tensors="pt").input_ids[0]]
        labels = [torch.cat((labels[0], torch.tensor([self.llm_tokenizer.eos_token_id]).long()))]

        lang = self.lang_dict[lang]
        txt_feats = self.llm_tokenizer(f"Given romanized transcriptions extracted from audio-visual materials, back-transliterate them into the original script of {lang}. Input:", return_tensors="pt").input_ids[0]
        
        roman_source = self.roman_label_list[0][index].replace(" ", "").replace("|", " ").strip()
        
        return {"id": index, 'lang': [lang], "video_source": video_feats, 'audio_source': audio_feats, "label_list": labels, 'text_source':[txt_feats], 'roman_source': [roman_source]}
    
    def collater(self, samples):
        samples = [s for s in samples if s["id"] is not None]
        if len(samples) == 0:
            return {}

        audio_source, video_source = [s["audio_source"] for s in samples], [s["video_source"] for s in samples]
        if audio_source[0] is None:
            audio_source = None
        if video_source[0] is None:
            video_source = None
        if audio_source is not None:
            audio_sizes = [len(s) for s in audio_source]
        else:
            audio_sizes = [len(s) for s in video_source]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        if audio_source is not None:
            collated_audios, padding_mask, audio_starts = self.collater_audio(audio_source, audio_size)
        else:
            collated_audios, audio_starts = None, None
        if video_source is not None:
            collated_videos, padding_mask, audio_starts = self.collater_audio(video_source, audio_size, audio_starts)
        else:
            collated_videos = None
            
         
        targets_by_label = [
            [s["label_list"][i] for s in samples]
            for i in range(self.num_labels)
        ]
        
        text_instructions = [
            [s["text_source"][i] for s in samples]
            for i in range(self.num_labels)
        ]
        
        roman_sources = [
            [s["roman_source"][i] for s in samples]
            for i in range(self.num_labels)
        ]

        langs = [
            [s["lang"][i] for s in samples]
            for i in range(self.num_labels)
        ]
        
        if self.zero_shot_lang == 'no':
            zero_shot_samples = [False for x in langs[0]]
        else:
            zero_shot_samples = [self.lang_dict[self.zero_shot_lang] == x for x in langs[0]]
        
        source = {"audio": collated_audios,
                  "video": collated_videos,
                  "instruction": text_instructions[0],
                  "roman_sources":roman_sources[0],
                  "langs": langs[0],
                  "zero_shot_samples": torch.tensor(zero_shot_samples)}
        
        net_input = {"source": source, "padding_mask": padding_mask}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
            #"utt_id": [s['fid'] for s in samples]
        }


        batch['target'] = targets_by_label[0]
            
        return batch
    
    def crop_to_max_size(self, wav, target_size, start=None):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0
        # longer utterances
        if start is None:
            start, end = 0, target_size
            if self.random_crop:
                start = np.random.randint(0, diff + 1)
                end = size - diff + start
        else:
            end = start + target_size
        return wav[start:end], start
    
    def collater_audio(self, audios, audio_size, audio_starts=None):
        audio_feat_shape = list(audios[0].shape[1:])
        collated_audios = audios[0].new_zeros([len(audios), audio_size]+audio_feat_shape)
        padding_mask = (
            torch.BoolTensor(len(audios), audio_size).fill_(False) # 
        )
        start_known = audio_starts is not None
        audio_starts = [0 for _ in audios] if not start_known else audio_starts
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat(
                    [audio, audio.new_full([-diff]+audio_feat_shape, 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size, audio_starts[i] if start_known else None
                )
        if len(audios[0].shape) == 2:
            collated_audios = collated_audios.transpose(1, 2) # [B, T, F] -> [B, F, T]
        else:
            collated_audios = collated_audios.permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]
        return collated_audios, padding_mask, audio_starts

    
    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates)
        for targets, label_rate in itr:
            if label_rate == -1:  
                targets, lengths, ntokens = self.collater_seq_label_llm(targets)
            else:
                raise NotImplementedError("not yet")
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list
    
    def collater_seq_label_llm(self, targets):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        pad, eos = self.llm_tokenizer("<|finetune_right_pad_id|>").input_ids[1], self.llm_tokenizer.eos_token_id
        targets_ = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False)
       
        new_targets = []
        for tar in targets:
            new_targets.append(tar[1:])

        prev_output_tokens = data_utils.collate_tokens(new_targets, pad_idx=pad, eos_idx=eos, left_pad=False, move_eos_to_beginning=False)
        
        
        return (targets_, prev_output_tokens), lengths, ntokens
    
    def collate_tokens(self,
        values,
        pad_idx,
        eos_idxs,
        left_pad=False,
        move_eos_to_beginning=False,
        pad_to_length=None,
        pad_to_multiple=1,
        pad_to_bsz=None,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

        batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
        res = values[0].new(batch_size, size).fill_(pad_idx)

        def copy_tensor(src, dst, eos_idx):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                if eos_idx is None:
                    # if no eos_idx is specified, then use the last token in src
                    dst[0] = src[-1]
                else:
                    dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)], eos_idxs[i])
        return res



    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]