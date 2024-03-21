## construct aqua thought graph

python construct_GoT_aqua.py

## stage 1 train rationale generation

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master-port 1979  main.py  \
    --data_root data     --dataset AQuA     --model declare-lab/flan-alpaca-base   \
    --user_msg rationale     --bs 16 --eval_bs 24 --epoch 100 --lr 5e-5 --output_len 512 \
    --use_generate --prompt_format QC-E     --output_dir experiments/AQuA_GoT_base
    # --bf16 #can be added to accelerate training

##inference
CUDA_VISIBLE_DEVICES=6 python  main.py  \
    --data_root data     --dataset AQuA     --model declare-lab/flan-alpaca-base   \
    --user_msg rationale     --bs 24 --eval_bs 24 --epoch 100 --lr 5e-5 --output_len 512 \
    --use_generate --prompt_format QC-E     --output_dir experiments/AQuA_GoT_base \
    --evaluate_dir {PATH_TO_CHECKPOINT}


##construct stage 2 aqua thought graph
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
    # --bf16 #can be added to accelerate training

##answer genration 
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
    --evaluate_dir {PATH_TO_CHECKPOINT_STAGE2}
    # --bf16 #can be added to accelerate training

