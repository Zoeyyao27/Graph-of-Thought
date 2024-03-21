import os
import numpy as np
import torch
import os
import re
import json
import argparse
import random
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from model import T5ForGoTGeneration
from utils_data import img_shape,load_data_std_aqua, load_data_img, ScienceQADatasetImg_GoT,AQuADatasetGoT
from utils_prompt import *
from utils_evaluate import get_scores,get_scores_aqua
from rich.table import Column, Table
from rich import box
from rich.console import Console
console = Console(record=True)
import nltk
import evaluate
from accelerate.utils import DistributedType


deepspeed_config = {
  "bf16": {
    "enabled": "auto"
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
   }, 
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": True,
    "contiguous_gradients": True,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": True
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 2000,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": False
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='ScienceQA')
    parser.add_argument('--got_root', type=str, default='GoT_dataset/')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='declare-lab/flan-alpaca-large')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=64)
    parser.add_argument('--eval_bs', type=int, default=4)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])
    
    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default=None, choices=['detr', 'clip', 'resnet','vit'], help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--caption_file', type=str, default= None)
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-A', help='prompt format template',
                        choices=['QCM-A', 'QCM-E', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE','QC-E','QCG-A'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--eval_strategy', type=str, default="steps", help='evaluation strategy', choices=['steps', 'epoch'])
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--bf16', action='store_true', help='use bf16 dtype')
    args = parser.parse_args()


    return args
        
def T5Trainer(
    dataframe, args,
):
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<s>']})
    vocab = tokenizer.get_vocab()
    s_token_id = vocab["<s>"]

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")
    if args.dataset == "ScienceQA":
        problems = dataframe['problems']
        qids = dataframe['qids']
        train_qids = qids['train']
        test_qids = qids['test']
        val_qids = qids['val']
    elif args.dataset == "AQuA":
        train_problems = dataframe['train']
        dev_problems = dataframe['dev']
        test_problems = dataframe['test']
    
    if args.evaluate_dir is not None:
        save_dir = args.evaluate_dir
    else:
        model_name = args.model.replace("/","-")
        gpu_count = torch.cuda.device_count()
        save_dir = f"{args.output_dir}/{args.user_msg}_{model_name}_{args.img_type}_{args.prompt_format}_lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}_useG{args.use_generate}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    print(save_dir)

    if args.img_type is not None:
        ##make sure aqua dataset does not have img_type
        if args.dataset == "AQuA":
            raise ValueError("AQuA dataset should not have img_type")
        patch_size = img_shape[args.img_type]
        model = T5ForGoTGeneration.from_pretrained(args.model,s_token_id=s_token_id, patch_size=patch_size) 
        model.resize_token_embeddings(len(tokenizer))

        name_maps = dataframe['name_maps'] 
        image_features = dataframe['image_features']
        
        train_set = ScienceQADatasetImg_GoT(
            problems,
            train_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
        )
        eval_set = ScienceQADatasetImg_GoT(
            problems,
            val_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.eval_le,
        )
        test_set = ScienceQADatasetImg_GoT(
            problems,
            test_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.test_le,
        )       

    else:
        ##make sure scienceqa dataset has img_type
        if args.dataset == "ScienceQA":
            raise ValueError("ScienceQA dataset should have img_type")
        model = T5ForGoTGeneration.from_pretrained(args.model,s_token_id=s_token_id) 
        model.resize_token_embeddings(len(tokenizer))
        train_set = AQuADatasetGoT(
            train_problems,
            "train",
            tokenizer,
            args.input_len,
            args.output_len,
            args,
        )
        eval_set = AQuADatasetGoT(
            dev_problems,
            "dev",
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.eval_le,
        )
        test_set = AQuADatasetGoT(
            test_problems,
            "test",
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.test_le,
        )

    datacollator = DataCollatorForSeq2Seq(tokenizer)
    print("model parameters: ", model.num_parameters())

    def extract_ans(ans):
        #extract the answer for scienceqa dataset
        if "The answer is" in ans:
            pattern = re.compile(r'The answer is \(([A-Z])\)')
            res = pattern.findall(ans)
            
            if len(res) == 1:
                answer = res[0]  # 'A', 'B', ...
            else:
                answer = "FAILED" 
            return answer  
        #extract the answer for aqua dataset
        else:
            match = re.search(r'[a-zA-Z]', ans)
            if match:
                first_letter = match.group().upper()

                if first_letter in 'ABCDE':
                    answer = first_letter
                else:
                    answer = "FAILED" 
            else:
                answer = "FAILED" 
            return answer

        
    # accuracy for answer inference
    def compute_metrics_acc(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            #preds = preds.argmax(axis=2)
        pred_result= np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(pred_result, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        correct = 0
        assert len(preds) == len(targets)
        for idx, pred in enumerate(preds):
            reference = targets[idx]
            reference = extract_ans(reference)
            extract_pred = extract_ans(pred)
            best_option = extract_pred
            if reference == best_option and reference!="FAILED":
                correct +=1 
        return {'accuracy': 1.0*correct/len(targets)}
    
    # rougel for rationale generation
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        processed_preds = []
        for pred in preds:
            pred = pred.strip()
            try:
                # use nltk to split the text into sentences
                processed_pred = "\n".join(nltk.sent_tokenize(pred))
            except IndexError:
                # if the text is too long, it may cause an IndexError
                print(f"IndexError occurred with text: {pred}")
                processed_pred = pred
            processed_preds.append(processed_pred)

        processed_labels = []
        for label in labels:
            label = label.strip()
            try:
                # use nltk to split the text into sentences
                processed_label = "\n".join(nltk.sent_tokenize(label))
            except IndexError:
                print(f"IndexError occurred with text: {label}")
                processed_label = label
            processed_labels.append(processed_label)

        return processed_preds, processed_labels

    def compute_metrics_rougel(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids 
        pred_result= np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(pred_result, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        decoded_preds, decoded_labels = postprocess_text(preds, targets)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # only use the last model for evaluation to save time
    if args.final_eval:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=False,
            evaluation_strategy="no",
            logging_strategy="steps",
            logging_steps=20, 
            save_strategy="epoch",
            save_total_limit = 2,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=args.weight_decay,
            num_train_epochs=args.epoch,
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            #resume_from_checkpoint=args.resume_from_checkpoint,
            report_to="none",
            deepspeed=deepspeed_config,
            bf16=args.bf16
        )
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    # evaluate at each epoch
    else:
        if args.evaluate_dir is None:
            training_args = Seq2SeqTrainingArguments(
                save_dir,
                do_train=True if args.evaluate_dir is None else False,
                do_eval=True,
                evaluation_strategy="steps",
                logging_strategy="steps",
                logging_steps=10, 
                save_strategy="steps",
                eval_steps=1000,
                save_steps=1000,
                save_total_limit = 2,
                learning_rate= args.lr,
                eval_accumulation_steps=args.eval_acc,
                per_device_train_batch_size=args.bs,
                per_device_eval_batch_size=args.eval_bs,
                weight_decay=args.weight_decay,
                num_train_epochs=args.epoch,
                metric_for_best_model="accuracy" if args.prompt_format in ["QCMG-A","QCM-A","QCG-A"] else "rougeL",
                predict_with_generate=args.use_generate,
                generation_max_length=args.output_len,
                load_best_model_at_end=True,
                report_to="none",
                deepspeed=deepspeed_config,
                bf16=args.bf16
            )
            training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

        else:
            ##do evaluation only
            training_args = Seq2SeqTrainingArguments(
                save_dir,
                do_train=True if args.evaluate_dir is None else False,
                do_eval=True,
                evaluation_strategy="steps",
                logging_strategy="steps",
                logging_steps=20,
                save_strategy="steps",
                eval_steps=500,
                save_steps=500,
                save_total_limit = 2,
                learning_rate= args.lr,
                eval_accumulation_steps=args.eval_acc,
                per_device_train_batch_size=args.bs,
                per_device_eval_batch_size=args.eval_bs,
                weight_decay=args.weight_decay,
                num_train_epochs=args.epoch,
                metric_for_best_model="accuracy" if args.prompt_format in ["QCMG-A","QCM-A","QCG-A"] else "rougeL",
                predict_with_generate=args.use_generate,
                generation_max_length=args.output_len,
                load_best_model_at_end=True,
                report_to="none",
            )
    
    #convert model to bf16 dtype
    # model = model.half()
    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits[0], dim=-1)
        return pred_ids, labels
    

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics_acc if args.prompt_format in ["QCMG-A","QCM-A","QCG-A"] else compute_metrics_rougel,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if not args.use_generate else None
    )

    if args.evaluate_dir is None:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(save_dir)
        
    metrics = trainer.evaluate(eval_dataset = test_set, max_length=args.output_len)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    predict_results = trainer.predict(test_dataset=test_set, max_length=args.output_len) 
    
    if trainer.is_world_process_zero() or args.evaluate_dir is not None:
        if args.use_generate:
            preds, targets = predict_results.predictions, predict_results.label_ids
        else:
            preds = predict_results.predictions[0]
            targets = predict_results.label_ids

        preds= np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        targets = tokenizer.batch_decode(
            targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        if args.dataset == "ScienceQA":
            results_ans = {}
            results_rationale = {}
            results_reference = {}
            
            num_fail = 0

            extract_pred_list=[]
            for idx, qid in enumerate(test_qids):
                pred = preds[int(idx)]
                ref = targets[int(idx)]
                extract_pred = extract_ans(pred)
                if extract_pred != "FAILED":
                    if extract_pred in args.options:
                        extract_pred = args.options.index(extract_pred)
                    else:
                        extract_pred = random.choice(range(0,len(args.options)))
                else:
                    num_fail += 1
                    extract_pred = random.choice(range(len(args.options))) # random choose one option
                results_ans[str(qid)] = extract_pred
                extract_pred_list.append(extract_pred)
                results_rationale[str(qid)] = pred
                results_reference[str(qid)] = ref

            scores = get_scores(results_ans, results_rationale, results_reference, os.path.join(args.data_root, "ScienceQA/problems.json"))
            preds = [pred.strip() for pred in preds]
            output_data = {
                    "num_fail": num_fail,
                    "scores": scores,
                    "preds": preds,
                    "extract_pred":extract_pred_list,
                    "labels": targets}
            if args.use_generate:
                output_prediction_file = os.path.join(save_dir,"predictions_ans_test.json")
            else: #with teach forcing
                #make dir
                if not os.path.exists(os.path.join(save_dir,"tf_pred")):
                    os.mkdir(os.path.join(save_dir,"tf_pred"))
                output_prediction_file = os.path.join(save_dir,"tf_pred","predictions_ans_test.json")
            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(output_data, indent=4))

        elif args.dataset == "AQuA":
            results_ans = {}
            results_rationale = {}
            results_reference = {}
            
            num_fail = 0
            extract_pred_list=[]
            for idx in range(len(preds)):
                pred = preds[int(idx)]
                ref = targets[int(idx)]
                extract_pred = extract_ans(pred)
                if extract_pred != "FAILED":
                    if extract_pred in args.options:
                        extract_pred = args.options.index(extract_pred)
                    else:
                        extract_pred = random.choice(range(0,len(args.options)))
                else:
                    num_fail += 1
                    extract_pred = random.choice(range(len(args.options)))

                results_ans[idx] = extract_pred
                extract_pred_list.append(extract_pred)
                results_rationale[idx] = pred
                results_reference[idx] = ref

            scores = get_scores_aqua(results_ans, results_rationale, results_reference, os.path.join(args.data_root, "AQuA/test.json"))
            preds = [pred.strip() for pred in preds]
            output_data = {
                    "num_fail": num_fail,
                    "scores": scores,
                    "preds": preds,
                    "extract_pred":extract_pred_list,
                    "labels": targets}
            if args.use_generate:
                output_prediction_file = os.path.join(save_dir,"predictions_ans_test.json")
            else: #with teacher forcing
                #make dir
                if not os.path.exists(os.path.join(save_dir,"tf_pred")):
                    os.mkdir(os.path.join(save_dir,"tf_pred"))
                output_prediction_file = os.path.join(save_dir,"tf_pred","predictions_ans_test.json")
            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(output_data, indent=4))
    
    # generate the rationale for the eval set
    if args.prompt_format in ["QCM-LE", "QCM-E","QC-E"]:
        torch.cuda.empty_cache()
        #del predict_results, preds, targets
        predict_results = trainer.predict(test_dataset=eval_set, max_length=args.output_len) 
        if trainer.is_world_process_zero():
            if args.use_generate:
                preds, targets = predict_results.predictions, predict_results.label_ids
            else:
                preds = predict_results.predictions[0]
                targets = predict_results.label_ids
                #preds = preds.argmax(axis=2)
            preds= np.where(preds != -100, preds, tokenizer.pad_token_id)
            preds = tokenizer.batch_decode(
                preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            targets = tokenizer.batch_decode(
                targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            preds = [pred.strip() for pred in preds]
            output_data = {"preds": preds,
                 "labels": targets}
            if args.use_generate:
                output_prediction_file = os.path.join(save_dir,"predictions_ans_eval.json")
            else: #with teacher forcing
                #make dir
                if not os.path.exists(os.path.join(save_dir,"tf_pred")):
                    os.mkdir(os.path.join(save_dir,"tf_pred"))
                output_prediction_file = os.path.join(save_dir,"tf_pred","predictions_ans_eval.json")

            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(output_data, indent=4))
    

if __name__ == '__main__':

    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )
    
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)
    
    if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

    if args.img_type is not None: #scienceqa dataset
        problems, qids, name_maps, image_features = load_data_img(args) 
        dataframe = {'problems':problems, 'qids':qids, 'name_maps': name_maps, 'image_features': image_features}
    else: #aqua dataset
        problems_train, problems_dev, problems_test = load_data_std_aqua(args)
        dataframe = {'train':problems_train,"dev":problems_dev,"test":problems_test}
        
    T5Trainer(
        dataframe=dataframe,
        args = args
    )
