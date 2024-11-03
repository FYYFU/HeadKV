import os
import json
import argparse
import numpy as np
import ipdb
from collections import defaultdict
from reason_needle.metrics import compare_answers, TASK_LABELS


import seaborn as sns
import matplotlib
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--model', type=str, default='meta-llama-3-8b-instruct')
    parser.add_argument('--capacity', type=int, default=128)
    return parser.parse_args(args)


def vis_save(accuracy, save_path, xlabels, ylabels, args):
    matplotlib.rc('font', size=14)
    # Base colormap
    # cmap = sns.diverging_palette(200,20,sep=20,as_cmap=True)
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    
    fig, ax = plt.subplots(figsize=(accuracy.shape[1] + accuracy.shape[1] // 2, accuracy.shape[0]))  
    sns.heatmap(accuracy, vmin=0, vmax=100, cmap=cmap, annot=False, fmt=".0f", 
                xticklabels=xlabels, yticklabels=ylabels, ax=ax,
                cbar_kws={'label': f'{args.dataset} {args.method} Score'},
                linewidths=0.5,  # Adjust the thickness of the grid lines here
                linecolor='grey',  # Set the color of the grid lines
                linestyle='--')
    ax.set_xlabel('Context')
    ax.set_ylabel('Depth')
    plt.xticks(rotation=45)

    plt.savefig(save_path, dpi=1080, bbox_inches='tight')  

model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 7950,
    "llama-3": 7950,
    "mistral": 31500,
    'qwen2': 31500,
    'phi': 31500
}



if __name__ == '__main__':
    args = parse_args()
    args.results_dir = f"{args.results_dir}/{args.model}_{args.capacity}"
    dataset_list = [
        'qa1',
        'qa2',
        'qa3',
        'qa4',
        'qa5',
        ]
    

    results_list = [
        ["dataset"],
        ["ReasonKV"],
    ]

    for key in model2maxlen:
        if key in args.model:
            model_max_len = model2maxlen[key]

    output_max_len = 15
    if model_max_len < 10000:
        splits = ['0k', '1k', '2k', '4k', '8k']
    else:
        splits = ['0k', '1k', '2k', '4k', '8k', '16k', '32k']


    total_scores = []

    for dataset in dataset_list:
        
        results_list[0].append(dataset)
    
        for idx, method in enumerate(["ReasonKV"]):
            # try:
            args.method = method
            args.dataset = dataset
            args.eval_file = os.path.join(args.results_dir,dataset,f"{method}.json")
            scores = dict()
            predictions, answers, lengths = [], [], []
            total_examples = defaultdict(list)
            # dataset = filename.split('.')[0]
            with open(args.eval_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        settings = data['setting']
                        total_examples[settings].append(data)
                    except:
                        print("error")
            
            scores = defaultdict(float)

            sorted_key = sorted(total_examples.keys())

            for key in sorted_key:
                cur_scores = []
                cur_examples = total_examples[key]
                for ex in cur_examples:
                    target = ex['answers']
                    pred = ex['pred']
                    question = ex['input']
                    cur_scores.append(compare_answers(
                        target=target, 
                        output=pred, 
                        question=question, 
                        task_labels=TASK_LABELS[dataset]
                    ))
                scores[key] = 100 * sum(cur_scores) / len(cur_scores)
        
            output_dir = os.path.dirname(args.eval_file)
            
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(scores, f, ensure_ascii=False, indent=4)
            
            accuracy = np.array(list(scores.values()))
            avg_accuracy = accuracy.sum() / accuracy.size
            total_scores.append(accuracy)

            for key in sorted_key:
                print(f"dataset {args.dataset} method {args.method} split {key} scores {scores[key]}")

            print(f"dataset {args.dataset} method {args.method} avg scores {avg_accuracy}")

    # ipdb.set_trace()
    total_scores = np.array(total_scores)
    print(f'avg dataset:{np.average(total_scores, axis=1)} sum dataset: {np.average(total_scores, axis=1).sum()}')
    print(f'avg split: {np.average(total_scores, axis=0)} sum split: {np.average(total_scores, axis=0).sum()}')
