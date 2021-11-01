#import termcolor
#import itertools
#from termcolor import colored
import torch
import transformers
#from tqdm import tqdm
import pandas as pd
#import gc
#import os
#import numpy as np
#import copy
#import sys
#import time
#import random
import torch.nn.functional as F
#from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from helpers import *
import argparse
from closs import evaluate_list
from helpers import *



parser = argparse.ArgumentParser(description='Generates text counterfactuals using the CLOSS method.')


parser.add_argument("--dataset", help="Target dataset; options are \'qnli\', \'imdb\', \'imdb_long\'.")
parser.add_argument("--beamwidth", help="Beam width for beam search.", type=int)
parser.add_argument("--w", help="Target number of evaluations per substitution.", type=int)
parser.add_argument("--K", help="Maximum number of substitutions sampled per salient location.", type=int)
parser.add_argument("--evaluation", help="The evaluation to perform; options are \'closs\', \'closs-sv\', \'closs-eo\', \'closs-sv+r\'. Note that CLOSS BB is run by using \'closs-eo\' for evaluation, \'default\' for lm_head and \'mask\' as the saliency method.")
parser.add_argument("--model", help="Model to use; options are \'bert\', \'roberta\'.")
parser.add_argument("--retrain_epochs", help="Number of retraining epoch for the LM head; default is 10.", type=int)
parser.add_argument("--lm_head", help="Language modeling head to use; options are \'default\', \'retrained\'.")
parser.add_argument("--saliency_method", help="Saliency method to use; options are \'mask\', \'norm_grad\'.")
parser.add_argument("--log_note", help="Extra information you want included in the log file's name.")
#parser.add_argument("--gpu_device", help="GPU device number.", type=int)

args = parser.parse_args()




# Note: there's an outstanding bug where a small amount of memory is always assigned to gpu device 0.
# This causes issues if you choose a non-zero gpu device because it forces the method to transfer
# data between gpu devices, which is slow.
# As such, running on a non-zero device requires we invoke the method like so:
# CUDA_VISIBLE_DEVICES=n python3 closs.py [arguments]
gpu_device_num = 0#args.gpu_device
use_imdb = False
use_imdb_long = False
use_qnli = False

if args.dataset == 'imdb':
    use_imdb = True
elif args.dataset == 'qnli':
    use_qnli = True
elif args.dataset in ['imdb_long', 'long_imdb']:
    use_imdb_long = True
else:
    print("Error: unknown dataset:", args.dataset)

runid = args.log_note

use_hotflip_only = (str.lower(args.evaluation) in ['hotflip_only', 'hotflip'])
use_grad_only = (str.lower(args.evaluation) in ['closs-sv', 'grad-only', 'grad_only'])
no_opt_lmh = (str.lower(args.evaluation) in ['closs-eo', 'no_opt', 'no-opt'])
random_logit_matrix = (str.lower(args.evaluation) in ['closs-sv+r', 'random'])
use_SVs = (str.lower(args.evaluation) in ['closs', 'sv', 'svs'])
test_acc = (str.lower(args.evaluation) in ['test'])
lms = 'prediction'

if no_opt_lmh or args.lm_head == 'default':
    lmh_data_source = 'default'
elif use_imdb_long or use_imdb:
    lmh_data_source = 'texts:imdb_seperate_' + str(args.retrain_epochs)
else:
    lmh_data_source = 'texts:qnli_seperate_' + str(args.retrain_epochs)


use_bert = str.lower(args.model) == 'bert'
use_roberta = str.lower(args.model) == 'roberta'

model_used_str =str.lower(args.model)

print("use_hotflip_only   :", use_hotflip_only)
print("use_SVs            :", use_SVs)
print("use_grad_only      :", use_grad_only)
print("no_opt_lmh         :", no_opt_lmh)
print("random_logit_matrix:", random_logit_matrix)
print("use_bert           :", use_bert)
print("use_roberta        :", use_roberta)
print("test_acc           :", test_acc)

if use_hotflip_only:
    substitution_evaluation_method = 'hotflip_only'
    substitution_gen_method = 'hotflip_only'
    method_str = 'hotflip_only'
elif use_SVs:
    substitution_evaluation_method = 'SVs'
    substitution_gen_method = 'logits'
    method_str = 'grounded_rand_SVs'
elif use_grad_only:
    substitution_evaluation_method = 'grad_only'
    method_str = 'grad_sort_only'
    substitution_gen_method = 'logits'
elif no_opt_lmh:
    substitution_evaluation_method = 'SVs'
    method_str = 'no_opt_grounded_rand_SVs'
    substitution_gen_method = 'no_opt_lmh'
elif random_logit_matrix:
    substitution_evaluation_method = 'SVs'
    method_str = 'random_grounded_rand_SVs'
    substitution_gen_method = 'random'
elif not test_acc:
    print("Error: no method chosen")

if use_roberta:
    if use_imdb or use_imdb_long:
        tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-imdb")
        sentiment_model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-imdb")
        #sentiment_model.lm_head = torch.load('textattack_roberta_imdb/lm_heads/default_lm_head.pth')
    elif use_qnli:
        tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-QNLI")
        sentiment_model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-QNLI")
        #sentiment_model.lm_head = torch.load('textattack_roberta_QNLI/lm_heads/default_lm_head.pth')
    
    if no_opt_lmh:
        LM_model = transformers.RobertaForMaskedLM.from_pretrained("roberta-base")
    else:
        LM_model = sentiment_model

elif use_bert:
    if use_imdb or use_imdb_long:
        tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
        sentiment_model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
    elif use_qnli:
        tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-QNLI")
        sentiment_model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-QNLI")
   
    if no_opt_lmh:
        LM_model = transformers.BertForMaskedLM.from_pretrained("bert-base-uncased")
        LM_model.lm_head = LM_model.cls
    else:
        LM_model = sentiment_model


retrain_data = []
if use_imdb or use_imdb_long:
    if use_imdb:
        same_texts_as_baselines = pd.read_csv("roberta-base-imdb-short-1000-log.csv")
    else:
        same_texts_as_baselines = pd.read_csv("bert-base-uncased-long-imdb-1000-log.csv")
    
    if lmh_data_source[:6] == 'texts:':
        retrain_data = process_dataframe(pd.read_csv("retrain_1000_data_different_from_roberta-base-imdb-short-1000-log.csv"), sentiment_model, tokenizer, test_acc)
    same_texts_as_baselines_list = process_dataframe(same_texts_as_baselines, sentiment_model, tokenizer, test_acc)


elif use_qnli:
    same_texts_as_baselines = pd.read_csv("roberta-base-qnli-new-1000-textfooler-log.csv")

    if lmh_data_source[:6] == 'texts:':
        retrain_data = process_dataframe(pd.read_csv("retrain_1000_data_different_from_roberta-base-qnli-new-1000-textfooler-log.csv"), sentiment_model, tokenizer, test_acc)
    same_texts_as_baselines_list = process_dataframe(same_texts_as_baselines, sentiment_model, tokenizer, test_acc)


sentiment_model.cuda(gpu_device_num)
sentiment_model.eval()

LM_model.cuda(gpu_device_num)
LM_model.eval()

for p in LM_model.parameters():
    p.requires_grad = False
    
for p in sentiment_model.parameters():
    p.requires_grad = False


lm_head = get_lm_head(LM_model, lmh_data_source, args.retrain_epochs, tokenizer, retrain_data, gpu_device_num)
lm_head.cuda(gpu_device_num)
lm_head.eval()
for p in lm_head.parameters():
    p.requires_grad = False

LM_model.lm_head = lm_head
sentiment_model.lm_head = lm_head





if use_hotflip_only:
    all_word_embeddings = torch.zeros((tokenizer.vocab_size, 768)).detach().cuda(gpu_device_num)

    for i in range(tokenizer.vocab_size):
        input_tensor = torch.tensor(i).view(1, 1).cuda(gpu_device_num)
        word_embedding = get_word_embeddings(sentiment_model, input_tensor)
        all_word_embeddings[i, :] = word_embedding
    all_word_embeddings = all_word_embeddings.detach().requires_grad_(False)
    print("Computed embeddings")
else:
    all_word_embeddings = None


print("use_imdb:", use_imdb)
print("use_imdb_long:", use_imdb_long)
print("use_qnli:", use_qnli)


arglist = args.dataset + '_' + str(args.beamwidth) + '_' + str(args.w) + '_' + str(args.K) + '_' + args.evaluation + '_' + str(args.retrain_epochs) + '_' + args.lm_head + '_' + args.saliency_method + '_' + args.log_note

print("substitution_gen_method=", substitution_gen_method, "; substitution_evaluation_method=", substitution_evaluation_method)
print("saliency_method=", args.saliency_method)
print("GPU device num:", gpu_device_num)


if not test_acc:
    evaluate_list(same_texts_as_baselines_list, sentiment_model, LM_model, args.retrain_epochs, 1, tokenizer, hs_lr=0.005, group_tokens=False, root_reg=['squared'], l=[1], extra_lasso=False, max_opt_steps=1, n_samples=args.K, topk=args.K,  substitutions_after_loc=0.15, substitutions_after_SVs=10, min_substitutions_after_SVs=50, use_hard_scoring=True, min_substitutions=15, use_random_n_SV_substitutions=False, min_run_sample_size=4, use_grad_for_loc=True, max_SV_loc_evals=0, slowly_focus_SV_samples=True, min_SV_samples_per_sub=args.w, SV_samples_per_eval_after_location=0.5, logit_matix_source=lms, use_SVs=True, use_exact=False, n_branches=args.beamwidth, tree_depth=0.15, beam_width=args.beamwidth, prob_left_early_stopping=0.499999, substitution_gen_method=substitution_gen_method, substitution_evaluation_method=substitution_evaluation_method, saliency_method=args.saliency_method, empty_cache_every_text=True, logname='closs_log_' + arglist + '__' + runid, data_len_str=args.dataset)

print('beamwidth:', args.beamwidth, 'w:', args.w, 'K:', args.K, 'lognotes:', runid)
