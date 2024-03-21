import os
from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from utils_prompt import *
import pickle

img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
    "vit": (145, 1024),
}


def load_data_std_aqua(args):
    def load_data(json_path):
        problems=[]
        with open(json_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                problems.append(data)
        return problems
    problems_train = load_data(os.path.join(args.data_root,args.dataset, 'train.json'))
    problems_dev = load_data(os.path.join(args.data_root,args.dataset, 'dev.json'))
    problems_test = load_data(os.path.join(args.data_root,args.dataset, 'test.json'))

    print(f"number of train problems: {len(problems_train)}\n")
    print(f"number of val problems: {len(problems_dev)}\n")
    print(f"number of test problems: {len(problems_test)}\n")
    
    
    return problems_train, problems_dev, problems_test

def load_data_img(args):
    print("loading data...")
    print("data root: ", args.data_root)
    problems = json.load(open(os.path.join(args.data_root, args.dataset,'problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, args.dataset,'pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]
    name_maps = json.load(open(os.path.join(args.data_root, args.dataset,'name_map.json')))

    # check
    if args.img_type == "resnet":
        image_features = np.load('vision_features/resnet.npy')
        image_features = np.expand_dims(image_features, axis=1)
        image_features = image_features.repeat(512, axis=1)
    elif args.img_type == "clip":
        image_features = np.load('vision_features/clip.npy')
    elif args.img_type == "detr":
        image_features = np.load('vision_features/detr.npy')
    elif args.img_type == "vit":
        image_features = torch.load("vision_features/vit.pth")
    else:
        image_features = np.load('vision_features/detr.npy')
    print("img_features size: ", image_features.shape)

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids, name_maps, image_features


class ScienceQADatasetImg_GoT(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, name_maps, tokenizer, source_len, target_len, args, image_features, test_le=None
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        self.image_ids = []
        self.got_input_text_list_final=[]
        self.got_adj_matrix_list_final=[]

        if args.eval_le != None and args.test_le != None:
            with open(os.path.join(args.got_root,'pred_rationale_mc_input_text.pkl'), 'rb') as f:
                self.got_input_text_list=pickle.load(f)
            with open(os.path.join(args.got_root,'pred_rationale_mc_adj_matrix.pkl'), 'rb') as f:
                self.got_adj_matrix_list=pickle.load(f)
            print("!!successfully load GoT from:")
            print(os.path.join(args.got_root,args.dataset,'pred_rationale_mc_adj_matrix.pkl'),os.path.join(args.got_root,'pred_rationale_got_input_text.pkl'))
            #assert False
        elif args.eval_le == None and args.test_le == None:
            with open(os.path.join(args.got_root,args.dataset,'mc_input_text.pkl'), 'rb') as f:
                self.got_input_text_list=pickle.load(f)
            with open(os.path.join(args.got_root,args.dataset,'mc_adj_matrix.pkl'), 'rb') as f:
                self.got_adj_matrix_list=pickle.load(f)
        else:
            assert False , 'make sure the eval_le and test_le are all given!'        

        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            prompt, target = build_train_pair(problems, qid, args, curr_le_data)
            self.target_text.append(target)
            self.source_text.append(prompt)
            self.got_input_text_list_final.append(self.got_input_text_list[int(qid)-1])
            self.got_adj_matrix_list_final.append(self.got_adj_matrix_list[int(qid)-1])

            if str(qid) in name_maps:
                i_vectors = image_features[int(name_maps[str(qid)])]
                self.image_ids.append(i_vectors)
            else:
                shape = img_shape[args.img_type]
                self.image_ids.append(np.zeros(shape))

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        image_ids = self.image_ids[index]

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        got_input_text=self.got_input_text_list_final[index]
        got_adj_matrix=self.got_adj_matrix_list_final[index]
        got_adj_matrix=torch.tensor(got_adj_matrix)

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        encoded_got_input_text = self.tokenizer.batch_encode_plus(
            got_input_text,
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()

        image_ids = torch.tensor(image_ids).squeeze()

        encoded_got_input_text_ids=encoded_got_input_text["input_ids"].squeeze()
        encoded_got_input_text_mask=encoded_got_input_text["attention_mask"].squeeze()        
        
        # print("source_text:",source_text)
        # print("target_text:",target_text)
        # print("got_input_text:",got_input_text)
        # assert False
        # print("got_adj_matrix:",got_adj_matrix.shape)
        # print("got_input_ids:",encoded_got_input_text_ids.shape)
        # print("got_mask:",encoded_got_input_text_mask.shape)
        # print("###########################")
        #print("image_ids:",image_ids)

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids,
            "got_adj_matrix":got_adj_matrix,
            "got_input_ids":encoded_got_input_text_ids,
            "got_mask":encoded_got_input_text_mask,
        }


class AQuADatasetGoT(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems,split, tokenizer, source_len, target_len, args, test_le=None
    ):
        self.tokenizer = tokenizer
        self.data = problems#{qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []

        self.got_input_text_list_final=[]
        self.got_adj_matrix_list_final=[]
        if args.eval_le != None and args.test_le != None:
            with open(os.path.join(args.got_root,split,'pred_rationale_mc_input_text.pkl'), 'rb') as f:
                self.got_input_text_list=pickle.load(f)
            with open(os.path.join(args.got_root,split,'pred_rationale_mc_adj_matrix.pkl'), 'rb') as f:
                self.got_adj_matrix_list=pickle.load(f)
            print("!!successfully load GoT")
            #assert False
        elif args.eval_le == None and args.test_le == None:
            with open(os.path.join(args.got_root,args.dataset,split,'mc_input_text.pkl'), 'rb') as f:
                self.got_input_text_list=pickle.load(f)
            with open(os.path.join(args.got_root,args.dataset,split,'mc_adj_matrix.pkl'), 'rb') as f:
                self.got_adj_matrix_list=pickle.load(f)
        else:
            assert False , 'make sure the eval_le and test_le are all given!'

        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0



        for qid,prob in enumerate(self.data):
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            prompt, target = build_train_pair_aqua(prob, args, curr_le_data)
            self.target_text.append(target)
            self.source_text.append(prompt)
            self.got_input_text_list_final.append(self.got_input_text_list[qid])
            self.got_adj_matrix_list_final.append(self.got_adj_matrix_list[qid])

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        got_input_text=self.got_input_text_list_final[index]
        got_adj_matrix=self.got_adj_matrix_list_final[index]
        got_adj_matrix=torch.tensor(got_adj_matrix)

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # print("!!!!&&&&&&&&&&&&")
        # print("source_text:",source_text)
        # print("target_text:",target_text)
        # print("got_input_text:",got_input_text)
        # assert False

        encoded_got_input_text = self.tokenizer.batch_encode_plus(
            got_input_text,
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()

        encoded_got_input_text_ids=encoded_got_input_text["input_ids"].squeeze()
        encoded_got_input_text_mask=encoded_got_input_text["attention_mask"].squeeze()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
            "got_adj_matrix":got_adj_matrix,
            "got_input_ids":encoded_got_input_text_ids,
            "got_mask":encoded_got_input_text_mask,
        }
