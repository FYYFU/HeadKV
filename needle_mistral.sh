#!/bin/bash

mkdir ./longbench_logs/
mkdir ./needle_results/

devices=(0 1 2 3 4 5 6 7)
head_choices=('copy') # copy, reason
betas=(1.005 1.01 1.1 1.2 1.5 2 5 10)
max_capacity_prompts=128
counter=0

for((i=0;i<1;i++));do 
    for((j=0;j<8;j++));do
        device=${devices[counter]}
        head_choice=${head_choices[i]}
        beta=${betas[j]}
        temp=1
        alpha=1

        bash needle_base.sh \
            $device \
            ReasonKV \
            ${max_capacity_prompts} \
            flash_attention_2 \
            mistralai/Mistral-7B-Instruct-v0.2 \
            $head_choice \
            $beta \
            $temp \
            1000 \
            33000 \
            100 \
            Mistral \
            Mistral-7B-Instruct-v0.2 > /tmp/zefan/projects/HeadKV/longbench_logs/llama3_ReasonKV_${head_choice}_base${max_capacity_prompts}_beta${beta}_temp${temp}.txt 2>&1 &
        ((counter+=1))
    done
done