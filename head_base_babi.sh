export CUDA_VISIBLE_DEVICES=$1

method=$2 # Support AdativeKV, ReasonKV
max_capacity_prompts=$3 
attn_implementation=$4 # Support "flash_attention_2"
model_path=$5
head_choice=$6
beta=$7
temp=$8

mkdir ./reason_needle_results/
save_dir="./reason_needle_results/results_long_bench_${head_choice}_base${max_capacity_prompts}_beta${beta}_temp${temp}" # path to result save_dir


python3 run_babilong.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --head_choice ${head_choice} \
    --beta ${beta} \
    --temp ${temp} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --use_cache True
