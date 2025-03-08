import json
import torch
import requests
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import editdistance
from llamaapi import LlamaAPI
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os


# Multiprocess
torch.multiprocessing.set_sharing_strategy('file_system')

from text_normalization.text_normalization import text_normalize

llama = LlamaAPI('your llama api key')
RESTORE_PROMPT = "This is the romanized <LANG> transcription '<TEXT>' \nConvert Roman Text into <LANG>. Don't answer additional text."


def calculate_wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    distance = editdistance.eval(ref_words, hyp_words)
    wer = distance / max(len(ref_words), 1)
    return wer, len(ref_words)

def calculate_cer(reference, hypothesis):
    reference = reference.replace(" ", "")
    hypothesis = hypothesis.replace(" ", "")
    distance = editdistance.eval(reference, hypothesis)
    cer = distance / max(len(reference), 1)
    return cer, len(reference)



def process_sample(llm_type, gt_text, text, target_lang, count=1):
    
    dict = {'Ground truth': gt_text}    
    
    while True:
        try:            
            restore_prompt = RESTORE_PROMPT.replace("<TEXT>", text).replace("<LANG>", target_lang)
            
            request_json = {
                "model": llm_type, #"llama3.1-70b", "llama3-8b", "llama3.2-3b", "mistral-7b-instruct"
                "messages": [
                    {"role": "user", "content": restore_prompt}
                
                ],
                "max_tokens": 256,
                "temperature": 0,
                "top_p": 1.0,
                "stream": False
            }
            response = llama.run(request_json)
            prediction = response.json()['choices'][0]['message']['content'].split("\n\n")[0].strip()
            if prediction:
                dict['prediction'] = prediction
                return dict
        except:
            pass


error_count = 0
        
def process_chunk(llm_type, gt_data_chunk, data_chunk, target_lang):
    results = []
    for idx, line in enumerate(data_chunk):
        result = process_sample(llm_type, gt_data_chunk[idx], line, target_lang, count=1)
        results.append(result)
    return results

def recognition(args, num_chunks=16):
    with open(args.predicted_roman_file, 'r', encoding='utf-8') as file:
        data = file.readlines()

    with open(args.gt_label, 'r', encoding='utf-8') as file:
        gt_data = file.readlines()
    
    chunk_size = len(data) // num_chunks
    data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    gt_data_chunks = [gt_data[i:i + chunk_size] for i in range(0, len(gt_data), chunk_size)]

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_chunk, [args.llm_type] * len(data_chunks), gt_data_chunks, data_chunks, [args.target_lang] * len(data_chunks))
    
    processed_data = []
    for result in results:
        processed_data.extend(result)
    

    total_wer = 0
    total_cer = 0
    total_ref_words = 0
    total_ref_chars = 0
    for idx, data in enumerate(tqdm(processed_data)):
        wer, ref_word_count = calculate_wer(text_normalize(data['Ground truth'].strip(), args.lang), text_normalize(data['prediction'].strip(), args.lang))
        cer, ref_char_count = calculate_cer(text_normalize(data['Ground truth'].strip(), args.lang), text_normalize(data['prediction'].strip(), args.lang))
        total_wer += wer * ref_word_count
        total_cer += cer * ref_char_count
        total_ref_words += ref_word_count
        total_ref_chars += ref_char_count
    
    weighted_average_wer = total_wer / total_ref_words
    weighted_average_cer = total_cer / total_ref_chars
    
    print(f'Zero shot language WER, CER: {weighted_average_wer *100}, {weighted_average_cer * 100}')
    
    
    processed_data.append(f'Zero shot language WER, CER: {weighted_average_wer *100}, {weighted_average_cer * 100}')
    

    os.makedirs(os.path.dirname(args.output_pth), exist_ok=True)
    with open(args.output_pth, 'w') as fw:
        json.dump(processed_data, fw, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_label", type=str)
    parser.add_argument("--target_lang", type=str)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--predicted_roman_file", type=str)
    parser.add_argument("--output_pth",type=str)
    parser.add_argument("--llm_type", type=str)

    args = parser.parse_args()
    
    recognition(args)