import neuralcoref
#from openie import StanfordOpenIE
import string
import re
import spacy
import numpy as np
import json
import pickle
import argparse
import os
import stanza
from stanza.server import CoreNLPClient
from tqdm import tqdm

stanza.install_corenlp()
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp) # Add neural coref to SpaCy's pipe
punc = string.punctuation
alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"
max_nodes=150


def coreference(s):
    doc = nlp(s)
    return doc._.coref_clusters

# def sentence_tokenize(text):
#     text = " " + text + "  "
#     text = text.replace("\n", " ")
#     text = re.sub(prefixes, "\\1<prd>", text)
#     text = re.sub(websites, "<prd>\\1", text)
#     if "Ph.D" in text:
#         text = text.replace("Ph.D.", "Ph<prd>D<prd>")
#     text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
#     text = re.sub(acronyms+" "+starters, "\\1<stop> \\2", text)
#     text = re.sub(alphabets + "[.]" + alphabets + "[.]" +
#                   alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
#     text = re.sub(alphabets + "[.]" + alphabets +
#                   "[.]", "\\1<prd>\\2<prd>", text)
#     text = re.sub(" "+suffixes+"[.] "+starters, " \\1<stop> \\2", text)
#     text = re.sub(" "+suffixes+"[.]", " \\1<prd>", text)
#     text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)

#     text = re.sub("([0-9])" + "[.]" + "([0-9])", "\\1<prd>\\2", text)

#     if "..." in text:
#         text = text.replace("...", "<prd><prd><prd>")
#     if "”" in text:
#         text = text.replace(".”", "”.")
#     if "\"" in text:
#         text = text.replace(".\"", "\".")
#     if "!" in text:
#         text = text.replace("!\"", "\"!")
#     if "?" in text:
#         text = text.replace("?\"", "\"?")

#     text = text.replace(".", ".<stop>")
#     text = text.replace("?", "?<stop>")
#     text = text.replace("!", "!<stop>")
#     text = text.replace("<prd>", ".")

#     sentences = text.split("<stop>")
#     sentences = sentences[:-1]
#     sentences = [s.strip() for s in sentences]
#     return sentences

# def compress_triple(triples,coref):
#     temp_set = []
#     for i in range(0, len(triples)):
#         cur = triples[i]
#         cur_subject = cur['subject'].lower()
#         cur_relation = cur['relation'].lower()
#         cur_object = cur['object'].lower()

#         for cluster in coref:
#             span=[w.text.lower() for w in cluster.mentions]
#             if cur_subject in span:
#                 cur_subject=cluster.main.text.lower()
#             if cur_object in span:
#                 cur_object=cluster.main.text.lower()
        
#         if len(temp_set) == 0:
#             temp_set.append([cur_subject, cur_relation, cur_object])
#         else:
#             flag = 0
#             #print(temp_set)
#             for j in range(0, len(temp_set)):
#                 ###save the longest when have two same entities
#                 if temp_set[j][0] == cur_subject and temp_set[j][1] == cur_relation:
                    
#                     if len(cur_object) > len(temp_set[j][2]):
#                         temp_set[j][2] = cur_object
#                     flag = 1
                    
#                 elif temp_set[j][0] == cur_subject and temp_set[j][2] == cur_object:
#                     if len(cur_relation) > len(temp_set[j][1]):
#                         temp_set[j][1] = cur_relation
#                     flag = 1
                    
#                 elif temp_set[j][2] == cur_object and temp_set[j][1] == cur_relation:
#                     if len(cur_subject) > len(temp_set[j][0]):
#                         temp_set[j][0] = cur_subject
#                     flag = 1
                    
            
#             if flag == 0:
#                 ##if no editing, then it is a new triplet, add to temp
#                 temp_set.append([cur_subject, cur_relation, cur_object])
                    
                        
#     return temp_set
def extract_triples(document):
    
    triples=[]
    for sent in document.sentence:
        for triple in sent.openieTriple:
            subject=getattr(triple,'subject')
            relation=getattr(triple,'relation')
            object=getattr(triple,'object')

            triples.append({'subject':subject,'relation':relation,'object':object})
    return triples


def compress_triple(annotate_result,coref):
    #print(coref)
    #print(annotate_result)
    triples=extract_triples(annotate_result)
    #print(len(triples))
    
    #assert False
    temp_set = []
    for i in range(0, len(triples)):
        cur = triples[i]
        cur_subject = cur['subject'].lower()
        cur_relation = cur['relation'].lower()
        cur_object = cur['object'].lower()

        for cluster in coref:
            span=[w.text.lower() for w in cluster.mentions]
            if cur_subject in span:
                cur_subject=cluster.main.text.lower()
            if cur_object in span:
                cur_object=cluster.main.text.lower()
        
        if len(temp_set) == 0:
            temp_set.append([cur_subject, cur_relation, cur_object])
        else:
            flag = 0
            #print(temp_set)
            for j in range(0, len(temp_set)):
                ###save the longest when have two same entities
                if temp_set[j][0] == cur_subject and temp_set[j][1] == cur_relation:
                    
                    if len(cur_object) > len(temp_set[j][2]):
                        temp_set[j][2] = cur_object
                    flag = 1
                    
                elif temp_set[j][0] == cur_subject and temp_set[j][2] == cur_object:
                    if len(cur_relation) > len(temp_set[j][1]):
                        temp_set[j][1] = cur_relation
                    flag = 1
                    
                elif temp_set[j][2] == cur_object and temp_set[j][1] == cur_relation:
                    if len(cur_subject) > len(temp_set[j][0]):
                        temp_set[j][0] = cur_subject
                    flag = 1
                    
            
            if flag == 0:
                ##if no editing, then it is a new triplet, add to temp
                temp_set.append([cur_subject, cur_relation, cur_object])
                    
                        
    return temp_set

def get_mind_chart(mc_context,max_nodes,client):
    """get mind chart

    Args:
        mc_context (string): the context to construct mind chart (question+" "+context+" "+lecture+" "+solution+" "+choice)

    Returns:
        triples(list of triplets list): [[I, love, NLP],[NLP,is,fun]]
        action_input(list):["I</s><s>love</s><s>NLP</s><s>NLP</s><s>is</s><s>fun"]
        action_adj(list): [adjecent matrix] 
    """
    mc_context = mc_context.replace("\n", " ")
    coref=coreference(mc_context)
    # triples = []
    # sentences = sentence_tokenize(mc_context)
    # for sent in sentences:

        # triple =  client.annotate(sent)

        # print(sent)
        # print(triple)
        # #assert False
        # #if len(triple) > 0:
        # triples.extend(compress_triple(triple,coref))

    mc_context = mc_context.replace("\n", " ")
    annotate_result =  client.annotate(mc_context)
    triples =  compress_triple(annotate_result,coref)
    #print("!!compressed triples!!")
    #print(len(triples))
    action_input = []
    #action_adj = []


    id2node = {}
    node2id = {}
    adj_temp = np.zeros([max_nodes, max_nodes])
    index = 0
    if len(triples) == 0:
        action_input.append('<pad>')
    else:
        temp_text = '<s>'
        for u in triples:
            if u[0] not in node2id:
                node2id[u[0]] = index
                id2node[index] = u[0]
                if index<max_nodes:
                    if temp_text == '<s>':
                        temp_text = temp_text + u[0]  
                    else:
                        temp_text = temp_text + '</s><s> ' + u[0] 
                    
                    index = index + 1
                else:
                    break
            if u[1] not in node2id:
                node2id[u[1]] = index
                id2node[index] = u[1]
                
                if index<max_nodes:
                    if  temp_text == '<s>':
                        temp_text =  temp_text +u[1] 
                    else:
                        temp_text = temp_text + '</s><s> ' + u[1] 
                    index = index + 1
                else:
                    break
                    
            if u[2] not in node2id:
                node2id[u[2]] = index
                id2node[index] = u[2]
                
                if index<max_nodes:
                    if temp_text == '<s>':
                        temp_text = temp_text + u[2] 
                    else:
                        temp_text = temp_text + '</s><s> ' + u[2] 
                    index = index + 1
                else:
                    break
            


            adj_temp[node2id[u[0]]][node2id[u[0]]] = 1
            adj_temp[node2id[u[1]]][node2id[u[1]]] = 1
            adj_temp[node2id[u[2]]][node2id[u[2]]] = 1
            
            adj_temp[node2id[u[0]]][node2id[u[1]]] = 1
            adj_temp[node2id[u[1]]][node2id[u[0]]] = 1
            
            adj_temp[node2id[u[1]]][node2id[u[2]]] = 1
            adj_temp[node2id[u[2]]][node2id[u[1]]] = 1
            
            
    
        action_input.append(temp_text)
    #action_adj.append(adj_temp)

    return action_input,adj_temp

def get_context_text(problem, use_caption=True):
    txt_context = problem['hint']
    img_context = problem['caption'] if use_caption else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context

def get_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    #print(choice_txt)
    return choice_txt

def get_lecture_text(problem):
    # \\n: GPT-3 can generate the lecture with more tokens.
    lecture = problem['lecture']#.replace("\n", "\\n")
    return lecture

def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['solution']#.replace("\n", "\\n")
    return solution


def load_data(args):

    problems = json.load(open(os.path.join(args.data_root, 'ScienceQA/problems.json')))
    captions = json.load(open(args.caption_file))["captions"]
    pid_splits = json.load(open(os.path.join(args.data_root, 'ScienceQA/pid_splits.json')))
    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    qids={"train":train_qids,"dev":val_qids,"test":test_qids}
    if args.generate_pred!="":
        dev_le_path=os.path.join(args.generate_pred,"predictions_ans_eval.json")
        dev_le_data =json.load(open(dev_le_path))["preds"]
        test_le_path=os.path.join(args.generate_pred,"predictions_ans_test.json")
        test_le_data =json.load(open(test_le_path))["preds"]
        for id,qid in enumerate(test_qids):
            problems[qid]["pred_le_data"]=test_le_data[id]
        for id,qid in enumerate(val_qids):
            problems[qid]["pred_le_data"]=dev_le_data[id]

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    return problems,qids




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--caption_file', type=str, default='./data/ScienceQA/instruct_captions.json')
    parser.add_argument('--use_caption', type=bool, default=True, help='use image captions or not')
    parser.add_argument('--generate_pred', type=str, default="", help='only for construct mind chart for pred rationale')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])
    parser.add_argument('--output_dir', type=str, default='GoT_output/ScienceQA')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    probs,qids = load_data(args)
    mc_input_text_list=[]
    mc_adj_matrix_list=[]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with CoreNLPClient(
            annotators=["ner","openie","coref"],
            memory='4G', 
            endpoint='http://localhost:9090',
            be_quiet=True) as client:
        for prob_id in tqdm(probs):
            #print(prob_id)
            prob=probs[prob_id]
            question = prob['question']
            context= get_context_text(prob)
            choice=get_choice_text(prob,args.options)
            lecture = get_lecture_text(probs[prob_id])
            solution = get_solution_text(probs[prob_id])
            if args.generate_pred!="":
                #max_nodes=250
                ##used to generate answer
                if "pred_le_data" in prob:
                    ##test or dev
                    assert prob_id in qids["dev"] or prob_id in qids["test"]
                    mc_context_text = question+" "+context+" "+choice+" "+prob["pred_le_data"]
                else:
                    ##train
                    assert prob_id in qids["train"]
                    mc_context_text = question+" "+context+" "+choice+" "+lecture+" "+solution
            else:
                mc_context_text = question+" "+context+" "+choice

            mc_input_text,mc_adj_matrix=get_mind_chart(mc_context_text,max_nodes,client)
            mc_input_text_list.append(mc_input_text)
            mc_adj_matrix_list.append(mc_adj_matrix)


          
        if args.generate_pred=="":
            mc_input_text_path='mc_input_text.pkl'
            mc_adj_matrix_path='mc_adj_matrix.pkl'
        else:
            mc_input_text_path='pred_rationale_mc_input_text.pkl'
            mc_adj_matrix_path='pred_rationale_mc_adj_matrix.pkl'
        
        if args.output_dir != "":
            mc_input_text_path=os.path.join(args.output_dir,mc_input_text_path)
            mc_adj_matrix_path=os.path.join(args.output_dir,mc_adj_matrix_path)
            
        with open(mc_input_text_path, 'wb') as f:
            pickle.dump(mc_input_text_list, f)
        
        with open(mc_adj_matrix_path, 'wb') as f:
            pickle.dump(mc_adj_matrix_list, f)   
        client.stop()  
