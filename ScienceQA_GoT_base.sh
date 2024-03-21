python construct_GoT_scienceqa.py

##train rationale generation
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master-port 2000 main.py \
    --data_root data     --dataset ScienceQA     --caption_file data/ScienceQA/instruct_captions.json \
    --model declare-lab/flan-alpaca-base \
    --user_msg rationale --img_type vit \
    --got_root GoT_dataset     --bs 6 --eval_bs 6 --epoch 100 --lr 5e-5 \
    --output_len 512     --use_caption --use_generate --prompt_format QCM-E     \
    --output_dir experiments/ScienceQA_GoT_base/

###rationale base generate 

CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_root data     --dataset ScienceQA     --caption_file data/ScienceQA/instruct_captions.json \
    --model declare-lab/flan-alpaca-base \
    --user_msg rationale --img_type vit \
    --got_root GoT_dataset     --bs 6 --eval_bs 6 --epoch 100 --lr 5e-5 \
    --output_len 512     --use_caption --use_generate --prompt_format QCM-E \
    --output_dir experiments/ScienceQA_GoT_base/ \
    --evaluate_dir {PATH_TO_CHECKPOINT}

###construct GoT thought graph for stage 2
python construct_GoT_scienceqa.py --generate_pred {PATH_TO_CHECKPOINT} \
    --output_dir 'GoT_output/scienceqa_pred/base'

##train answer generation
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master-port 2000 main.py \
    --data_root data \
    --dataset ScienceQA --caption_file data/ScienceQA/instruct_captions.json \
    --model declare-lab/flan-alpaca-base \
    --user_msg answer --img_type vit \
    --bs 8 --eval_bs 16 --epoch 30 --lr 5e-5 --output_len 64 \
    --use_generate --prompt_format QCMG-A \
    --got_root GoT_output/scienceqa_pred/base \
    --output_dir experiments/ScienceQA_GoT_base/  \
    --eval_le {PATH_TO_CHECKPOINT}/predictions_ans_eval.json \
    --test_le {PATH_TO_CHECKPOINT}/predictions_ans_test.json

### answer base generate

CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_root data \
    --dataset ScienceQA --caption_file data/ScienceQA/instruct_captions.json \
    --model declare-lab/flan-alpaca-base \
    --user_msg answer --img_type vit \
    --bs 8 --eval_bs 16 --epoch 30 --lr 5e-5 --output_len 64 \
    --use_generate --prompt_format QCMG-A \
    --got_root GoT_output/scienceqa_pred/base \
    --output_dir experiments/ScienceQA_GoT_base/  \
    --eval_le {PATH_TO_CHECKPOINT}/predictions_ans_eval.json \
    --test_le {PATH_TO_CHECKPOINT}/predictions_ans_test.json \
    --evaluate_dir {PATH_TO_CHECKPOINT_STAGE2   }