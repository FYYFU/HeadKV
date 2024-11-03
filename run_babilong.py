import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from datasets import Dataset
from tqdm.auto import tqdm
import time
import json

import ipdb
import pandas as pd
import numpy as np
from pathlib import Path

from reason_needle.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
from datasets import load_from_disk

import random
import argparse
from tqdm import tqdm


datasets = [
    'qa1',
    'qa2',
    'qa3',
    'qa4',
    'qa5'
]


model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 7950,
    "llama-3": 7950,
    "mistral": 31500,
    'qwen2': 31500,
    'phi': 31500
}



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def main(args):
    

    print("Loading data...")
    
    test_data = []

    input_max_len = 0
    
    model_path = args.model_path.lower()

    
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]

    output_max_len = 15
    if model_max_len < 10000:
        splits = ['0k', '1k', '2k', '4k', '8k']
    else:
        splits = ['0k', '1k', '2k', '4k', '8k', '16k', '32k']

    
    model_name = model_path.split("/")[-1]
    os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset), exist_ok=True)
    fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, f"{args.method}.json"), "w")
    
    for split_index, split_name in enumerate(splits):

        data_path = './reason_needle/babilong-100examples'
        reason_dataset = load_from_disk(f'{data_path}/{split_name}')[dataset]
        pbar = tqdm(total = len(reason_dataset), desc=f'[{args.dataset} - {split_name}]')
        # ipdb.set_trace()
        for sample in reason_dataset:
            target = sample['target']
            context = sample['input']
            question = sample['question']

            prompt = get_formatted_input(
                context=context, 
                question=question,
                examples=DEFAULT_PROMPTS[args.dataset]['examples'],
                instruction=DEFAULT_PROMPTS[args.dataset]['instruction'], 
                post_prompt=DEFAULT_PROMPTS[args.dataset]['post_prompt'],
                template=DEFAULT_TEMPLATE
            )
            prompt = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            tokenized_prompts = tokenizer([prompt], padding="longest", return_tensors="pt", add_special_tokens=False).to('cuda')

            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask

            if len(batch_input_ids[0]) > model_max_len:
                half = int(model_max_len/2)
                prompt = tokenizer.decode(batch_input_ids[0][:half])+tokenizer.decode(batch_input_ids[0][-half:])
                tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=False).to('cuda')
                batch_input_ids = tokenized_prompts.input_ids
                attention_mask = tokenized_prompts.attention_mask

            model.model.config.window_size = 8
            model.model.config.base_capacity = args.max_capacity_prompts
            model.model.config.head_choice = args.head_choice
            model.model.config.beta = args.beta
            model.model.config.temp = args.temp
            model.model.config.kernel_size = 7
            model.model.config.skip = 0
            model.model.config.normalize = True
            model.model.config.pooling = "maxpool"
            model.model.config.floor = 0.2

            context_length = batch_input_ids.shape[-1]

            output = model.generate(
                **tokenized_prompts,
                output_attentions = args.output_attentions,
                max_new_tokens=output_max_len,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id]
            )

            batch_outputs =tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)
            torch.cuda.empty_cache()

            example = {}
            example["prompt"] = prompt
            example["input"] = question
            example["answers"] = target
            example["pred"] = batch_outputs[0]
            example['setting'] = f'split {split_index}: {split_name}'
            example["dataset"] = args.dataset

            fout.write(json.dumps(example) + "\n")
            pbar.update(1)

        pbar.close()
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
        
        
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")
    parser.add_argument("--head_choice", type=str, default='random', choices=['random', 'copy', 'musique', 'reason', 'mix', 'mix_top1', 'musique_top1', 'mix_top3', 'musique_top3', 'merge', 'final', 'reason', 'final_reason', 'final_merge'])
    parser.add_argument('--beta', type=float, default=1.5)
    parser.add_argument('--temp', type=float, default=1.0)

    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.model_path == 'mistralai/Mistral-7B-Instruct-v0.2':
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=args.use_fast_tokenizer,
            padding_side="left",
            revision='dca6e4b60aca009ed25ffa70c9bb65e46960a573'
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=args.use_fast_tokenizer,
            padding_side="left"
        )

    if args.method.lower() != 'fullkv':
        from headkv.monkeypatch import replace_llama, replace_mistral 
        replace_llama(args.method)
        replace_mistral(args.method)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation
    )
    

        

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    

        
    model.eval()
    
    save_dir = args.save_dir
    
    max_capacity_prompts = args.max_capacity_prompts
    


    for idx, dataset in enumerate(datasets):
        
        print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(datasets)}")
        print(f'base capacity: {args.max_capacity_prompts}\thead_choice:{args.head_choice}\tbeta:{args.beta}\ttemp:{args.temp}')

        args.dataset = dataset
        
        main(args)










