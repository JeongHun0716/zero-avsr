import pandas as pd
import os
import argparse

lang_dict = {
    'ar': 'ara',
    'de': 'deu',
    'el': 'ell',
    'es': 'spa',
    'fr': 'fra',
    'it': 'ita',
    'pt': 'por',
    'ru': 'rus'
}

def sort_and_save(args):
    file_path = args.input_pth
    
    if not os.path.exists(file_path):
        print(f"No file exists")
        return
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()

    split_data = []
    for item in data:
        text, number = item.rsplit('\t', 1)
        number = int(number.split('-')[-1].replace(')', ''))
        split_data.append((text, number))

    df = pd.DataFrame(split_data, columns=["Text", "Number"])
    df_sorted = df.sort_values(by="Number")


    output_file_path = args.output_pth
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for text in df_sorted["Text"]:
            if text.strip(): 
                f.write(text + '\n')
            else: 
                f.write("\t" + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pth", type=str)
    parser.add_argument("--output_pth", type=str)
    args = parser.parse_args()
    
    sort_and_save(args)
    
    
    

