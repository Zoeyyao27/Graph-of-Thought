# Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Large Language Models

## Installation

```
bash install.sh
```

## Datasets

### ScienceQA

#### (1) Dataset

Download the dataset from the following repository and put `name_map.json`,`pid_splits.json` and `problems.json` under `data/Scienceqa/`:

```
https://github.com/lupantech/ScienceQA/tree/main/data
```

#### (2) Vision Features and Instruct Captions

We use the same extracted vision features and  instruct captions from  [mm-cot](https://github.com/amazon-science/mm-cot).

You can download  [vision_features](https://huggingface.co/cooelf/vision_features/tree/main) and put the files under `vision_features`

Instruct captions can be found in `data/Scienceqa/instruct_captions.json`

### AQUA-RAT

Download the dataset from the following repository and put all json files under `data/AQuA/`

```
https://github.com/google-deepmind/AQuA
```



## Ready! You GoT it!

### ScienceQA

```
# Thought graph construction
python construct_GoT_scienceqa.py
```

```
#train stage1: rationale generation
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py \
    --data_root data     --dataset ScienceQA \
    --caption_file data/ScienceQA/instruct_captions.json \
    --model declare-lab/flan-alpaca-base \
    --user_msg rationale --img_type vit \
    --got_root GoT_dataset \
    --bs 8 --eval_bs 16 --epoch 100 --lr 5e-5 \
    --output_len 512 --use_caption --use_generate --prompt_format QCM-E \
    --output_dir experiments/ScienceQA_GoT_base/ \
    --bf16 

#evaluate stage1
CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_root data     --dataset ScienceQA \
    --caption_file data/ScienceQA/instruct_captions.json \
    --model declare-lab/flan-alpaca-base \
    --user_msg rationale --img_type vit \
    --got_root GoT_dataset \
    --bs 8 --eval_bs 16 --epoch 100 --lr 5e-5 \
    --output_len 512 --use_caption --use_generate --prompt_format QCM-E \
    --output_dir experiments/ScienceQA_GoT_base/ \
    --bf16 \
    --evaluate_dir {PATH_TO_CHECKPOINT}
```

```
#construct GoT thought graph for stage 2
python construct_GoT_scienceqa.py --generate_pred {PATH_TO_CHECKPOINT} \
    --output_dir GoT_output/scienceqa_pred/base
```

```
#train stage2: answer generation
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py \
    --data_root data     --dataset ScienceQA \
    --caption_file data/ScienceQA/instruct_captions.json \
    --model declare-lab/flan-alpaca-base \
    --user_msg answer --img_type vit \
    --bs 8 --eval_bs 16 --epoch 50 --lr 4e-5 --output_len 64 \
    --use_generate --prompt_format QCMG-A \
    --got_root GoT_output/scienceqa_pred/base \
    --output_dir experiments/ScienceQA_GoT_base/  \
    --eval_le {PATH_TO_CHECKPOINT}/predictions_ans_eval.json \
    --test_le {PATH_TO_CHECKPOINT}/predictions_ans_test.json \
    --bf16 

#evaluate stage2
CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_root data     --dataset ScienceQA \
    --caption_file data/ScienceQA/instruct_captions.json \
    --model declare-lab/flan-alpaca-base \
    --user_msg answer --img_type vit \
    --bs 8 --eval_bs 16 --epoch 50 --lr 4e-5 --output_len 64 \
    --use_generate --prompt_format QCMG-A \
    --got_root GoT_output/scienceqa_pred/base \
    --output_dir experiments/ScienceQA_GoT_base/  \
    --eval_le {PATH_TO_CHECKPOINT}/predictions_ans_eval.json \
    --test_le {PATH_TO_CHECKPOINT}/predictions_ans_test.json \
    --bf16 \
    --evaluate_dir {PATH_TO_CHECKPOINT_STAGE2}
```

### AQUA-RAT

```
# construct AQUA thought graph
python construct_GoT_aqua.py
```



```
#train stage1: rationale generation
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py \
    --data_root data     --dataset AQuA \
    --model declare-lab/flan-alpaca-base \
    --user_msg rationale --bs 8 --eval_bs 16 \
    --epoch 100 --lr 5e-5 --output_len 512 \
    --use_generate --prompt_format QC-E \
    --output_dir experiments/AQuA_GoT_base \
    --bf16 

#evaluate stage1
CUDA_VISIBLE_DEVICES=0 python  main.py  \
    --data_root data     --dataset AQuA \
    --model declare-lab/flan-alpaca-base \
    --user_msg rationale --bs 8 --eval_bs 16 \
    --epoch 100 --lr 5e-5 --output_len 512 \
    --use_generate --prompt_format QC-E \
    --output_dir experiments/AQuA_GoT_base \
    --bf16 --evaluate_dir {PATH_TO_CHECKPOINT}

```

```
##construct stage 2 AQUA thought graph
python construct_GoT_aqua.py --generate_pred {PATH_TO_CHECKPOINT} \
    --output_dir 'GoT_output/AQuA_pred/base'


## stage 2 train answer generation
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master-port 1979 main.py \
    --data_root data \
    --dataset AQuA \
    --model declare-lab/flan-alpaca-base \
    --user_msg answer \
    --bs 20 --eval_bs 20 --epoch 20 --lr 4e-5 --output_len 64 \
    --prompt_format QCG-A --use_generate\
    --got_root /data/yaoy/GoT/GoT/GoT_output/AQuA_pred/base_w_tf \
    --output_dir experiments/AQuA_GoT_base \
    --eval_le {PATH_TO_CHECKPOINT}/predictions_ans_eval.json \
    --test_le {PATH_TO_CHECKPOINT}/predictions_ans_test.json \
    --bf16 

#evaluate stage2
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master-port 1979 main.py \
    --data_root data \
    --dataset AQuA \
    --model declare-lab/flan-alpaca-base \
    --user_msg answer \
    --bs 20 --eval_bs 20 --epoch 20 --lr 4e-5 --output_len 64 \
    --prompt_format QCG-A --use_generate\
    --got_root /data/yaoy/GoT/GoT/GoT_output/AQuA_pred/base_w_tf \
    --output_dir experiments/AQuA_GoT_base \
    --eval_le {PATH_TO_CHECKPOINT}/predictions_ans_eval.json \
    --test_le {PATH_TO_CHECKPOINT}/predictions_ans_test.json \
    --bf16 \
    --evaluate_dir {PATH_TO_CHECKPOINT_STAGE2}
```

