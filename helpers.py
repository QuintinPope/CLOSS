import math
import termcolor
import itertools
from termcolor import colored
import torch
import transformers
from tqdm import tqdm
import pandas as pd
import gc
import os
import numpy as np
import scipy
import copy
import sys
import time
import random
import torch.nn.functional as F
import nltk
from nltk.corpus import stopwords
from termcolor import colored
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def l1_nearest_neighbor(embedding, all_word_embeddings):
    l1_distances = torch.abs(all_word_embeddings - embedding)
    magnitudes = torch.sum(l1_distances, dim=1)
    return magnitudes

def l2_nearest_neighbor(embedding, all_word_embeddings):
    l1_distances = torch.abs(all_word_embeddings - embedding)
    magnitudes = torch.sum(l1_distances ** 2, dim=1) ** (1/2)
    return magnitudes

def isBert(model):
    return type(model) in [transformers.models.bert.modeling_bert.BertForSequenceClassification, transformers.models.bert.modeling_bert.BertForMaskedLM]

def isRoberta(model):
    return type(model) in [transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification, transformers.models.roberta.modeling_roberta.RobertaForMaskedLM]

# Adapted from /content/pytorch-pretrained-BERT/pytorch_pretrained_bert/modeling.py
def onwards(averaged_activation, layer_id, model, gpu_device_num):
    logits = model(inputs_embeds=averaged_activation).logits
    return logits




def onwards_token_predict(averaged_activation, layer_id, model, flip_target, gpu_device_num):
    final_hidden = model(inputs_embeds=averaged_activation, output_hidden_states=True).hidden_states[-1]
    predictions = model.lm_head(final_hidden)
    return predictions


def onwards_token_train(averaged_activation, model, gpu_device_num):
    outputs = model(inputs_embeds=averaged_activation, output_hidden_states=True)
    final_hidden = outputs.hidden_states[-1]
    logits = outputs.logits
    return final_hidden


def probability_positive(tokenizer, sentiment_model, text, gpu_device_num):
    if type(text) == torch.Tensor:
        ids = text.view(1, -1).cuda(gpu_device_num)
    elif type(text) == list:
        if type(text[0]) == int:
            ids = torch.tensor(text).view(1, -1).cuda(gpu_device_num)
        else:
            ids = torch.tensor(tokenizer.convert_tokens_to_ids(text)).view(1, -1).cuda(gpu_device_num)
    else:
        ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True, truncation=True)).view(1, -1).cuda(gpu_device_num)
    #embedding = get_embeddings(sentiment_model, ids, gpu_device_num)
    #logits = sentiment_model(inputs_embeds=embedding).logits
    #print("probability_positive IO:", logits)
    #prob_positive = torch.nn.functional.softmax(logits, dim=1).T[1].item()
    
    
    
    if isBert(sentiment_model):
        outputs = sentiment_model(ids.cuda(gpu_device_num), token_type_ids=get_token_type_ids(ids).cuda(gpu_device_num))
    elif isRoberta(sentiment_model):
        outputs = sentiment_model(ids.cuda(gpu_device_num))
    loss = outputs.loss
    logits = outputs.logits
        
    #ppe = probability_positive(sentiment_model, tokens, gpu_device_num)
    prob_positive = torch.nn.functional.softmax(logits, dim=1)[0][1].item()
    return prob_positive

def get_word_embeddings(model, ids):
    if isRoberta(model):
        word_embedding = model.roberta.embeddings.word_embeddings(ids)
    elif isBert(model):
        word_embedding = model.bert.embeddings.word_embeddings(ids)
    else:
        print("Error: unknown model type:", type(model))
    return word_embedding



def get_embeddings(model, ids, gpu_device_num):
    if isRoberta(model):
        word_embedding = model.roberta.embeddings(ids)
    elif isBert(model):
        token_type_ids = get_token_type_ids(ids).cuda(gpu_device_num)
        #token_type_embeddings = model.bert.embeddings.token_type_embeddings(token_type_ids)
        word_embedding = model.bert.embeddings(input_ids=ids, token_type_ids=token_type_ids)
    else:
        print("Error: unknown model type:", type(model))
    return word_embedding

def pp_to_pg(flip_target, ops, pp):
    if flip_target == 1:
        return pp - ops
    return ops - pp

def pp_to_pl(flip_target, pp):
    if flip_target == 1:
        return 1 - pp
    return pp

def get_token_type_ids(tokens):
    passed_SEP = False
    if type(tokens) == torch.Tensor:
        tokens = tokens.view(-1).tolist()
    token_type_ids = [0 for _ in range(len(tokens))]
    #print(tokens)
    if type(tokens[0]) == str:
        for i in range(len(tokens)):
            if passed_SEP:
                token_type_ids[i] = 1
            if tokens[i] == '[SEP]':
                passed_SEP = True
    else:
        for i in range(len(tokens)):
            if passed_SEP:
                token_type_ids[i] = 1
            if tokens[i] == 102:
                passed_SEP = True
    return torch.tensor([token_type_ids])

def print_token_importances(sentiment_model, locational_scores, tokens, n_tokens, desc, gpu_device_num):
    paired_location_and_score = [[i, 0] for i in range(0, n_tokens)]
    for i in range(0, n_tokens):
        if i == 0 or i == n_tokens - 1:
            paired_location_and_score[i][1] = 0
        else:
            paired_location_and_score[i][1] = locational_scores[i].item()
    
    paired_location_and_score = sorted(paired_location_and_score, key=lambda x: x[1], reverse=True)
    black_cutoff = paired_location_and_score[int(0.5 * n_tokens)][1]
    red_cutoff = paired_location_and_score[int(0.3 * n_tokens)][1]
    yellow_cutoff = paired_location_and_score[int(0.2 * n_tokens)][1]
    green_cutoff = paired_location_and_score[int(0.15 * n_tokens)][1]
    blue_cutoff = paired_location_and_score[int(0.05 * n_tokens)][1]
    if desc == None:
        print('100 - 50%')
        print(colored('50  - 30%', 'red'))
        print(colored('30  - 20%', 'yellow'))
        print(colored('20  - 10%', 'green'))
        print(colored('10  -  5%', 'blue'))
        print(colored('5   -  0%', 'magenta'))
    else:
        #pass
        print(desc)
    importance_string = ''
    for tok_loc in range(n_tokens):
        
        old_tok = tokens[tok_loc]
        spacechar = ' '
        if isRoberta(sentiment_model):
            spacechar = ''
        elif (len(old_tok) > 2 and old_tok[0] == '#' and old_tok[1] == '#'):
            spacechar = ''
            old_tok = old_tok[2:]
        
        if tok_loc == 0 or tok_loc == n_tokens - 1:
            loc_importance_score = 0
        else:
            loc_importance_score = locational_scores[tok_loc].item()
        
        if loc_importance_score <= black_cutoff:
            importance_string += spacechar + old_tok
        elif loc_importance_score <= red_cutoff:
            importance_string += spacechar + colored(old_tok, 'red')
        elif loc_importance_score <= yellow_cutoff:
            importance_string += spacechar + colored(old_tok, 'yellow')
        elif loc_importance_score <= green_cutoff:
            importance_string += spacechar + colored(old_tok, 'green')
        elif loc_importance_score <= blue_cutoff:
            importance_string += spacechar + colored(old_tok, 'blue')
        else:
            importance_string += spacechar + colored(old_tok, 'magenta')
    print(importance_string.replace('Ġ', ' '))

def compute_substitution_scores(all_word_embeddings, sentiment_model, tokenizer, loss_fct, flip_target, tokens, gpu_device_num):
    
    ids_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).cuda(gpu_device_num).view(1,-1)
    embedding = get_embeddings(sentiment_model, ids_tensor, gpu_device_num).cuda(gpu_device_num).requires_grad_(True)
    
    embedding_opt = torch.optim.Adam([embedding], lr=1e-4)
    initial_outputs = onwards(embedding, 0, sentiment_model, gpu_device_num)
    #if random.randint(0, 100) == 4:
    #    print(torch.nn.functional.softmax(initial_outputs, dim=1))
    #prob_pos = torch.nn.functional.softmax(initial_outputs, dim=1).T[1].item()
    prob_pos = probability_positive(tokenizer, sentiment_model, tokens, gpu_device_num)
    
    loss = loss_fct(initial_outputs, torch.tensor([1 - flip_target]).cuda(gpu_device_num).long())
    loss.backward()
    embedding_grad = embedding.grad[0]
    #print("embedding_grad     :", embedding_grad.size())
    #print("all_word_embeddings:", all_word_embeddings.size())
    #print("embedding          :", embedding.size())
    
    #token_derivatives = torch.sum(torch.pow(embedding_grad * embedding[0], 2), dim=1)
    token_derivatives = torch.sum(embedding_grad * embedding[0], dim=1)
    if random.randint(0, 1000) == 3:
        print(embedding[0].size(), embedding_grad.size(), (embedding_grad * embedding[0]).size(), token_derivatives.size())
        print_token_importances(sentiment_model, token_derivatives, tokens, len(tokens), "Token derivs", gpu_device_num)
    
    #print(embedding_grad.size(), all_word_embeddings.size())
    #vocab_derivatives = torch.zeros((len(tokens), tokenizer.vocab_size)).cuda(gpu_device_num)
    #for i in range(len(tokens)):
    #    vocab_derivatives[i][:] = torch.sum(torch.pow(embedding_grad[i] * all_word_embeddings, 2), dim=1)
    
    vocab_derivatives = torch.matmul(embedding_grad, all_word_embeddings.T)
    #print("vocab_derivatives  :", vocab_derivatives.size())
    #substitution_scores = (token_derivatives - vocab_derivatives.T).cuda(gpu_device_num)
    substitution_scores = (vocab_derivatives.T - token_derivatives).cuda(gpu_device_num)
    #substitution_scores = torch.rand_like(substitution_scores)
    
    #torch.sum(torch.pow(all_word_embeddings * embedding_grad, 2), dim=2)
    return substitution_scores.T, prob_pos
    
def hotflip_beamsearch(all_word_embeddings, sentiment_model, tokenizer, loss_fct, beam_width, tree_depth, prob_left_early_stopping, topk, flip_target, prob_pos, tokens, n_tokens, gpu_device_num):
    beam = [[tokens, 0, []]]
    extra_evals = 0
    for i in range(int(n_tokens * tree_depth)):
        #print([[b[1], b[2]] for b in beam])
        final_result = []
        new_beam = []
        potential_substitutions = []
        for c_num in range(len(beam)):
            c = beam[c_num]
            c_tokens = c[0]
            c_score = c[1]
            c_indexes_modified = c[2]
            substitution_scores, candidate_prob_pos = compute_substitution_scores(all_word_embeddings, sentiment_model, tokenizer, loss_fct, flip_target, c_tokens, gpu_device_num)
            extra_evals += 1
            #print(candidate_prob_pos)

            candidate_prob_left = pp_to_pl(flip_target, candidate_prob_pos)
            #print("Candidate prob left:", candidate_prob_left)
            if candidate_prob_left < prob_left_early_stopping:
                print("CF FOUND!!!!!!!!!!!!!!")
                return c_tokens, extra_evals
            sub_scores, sub_ids = torch.topk(substitution_scores, topk, dim=1)
            for j in range(1, n_tokens - 1):
                if (not j in c_indexes_modified) and (not tokens[j] in ['[SEP]', '[CLS]', '</s>', '<s>']):
                    for l in range(topk):
                    
                        id_of_token_to_insert = sub_ids[j][l].item()
                        sub_score = sub_scores[j][l].item()
                    
                        potential_substitutions.append([(j, id_of_token_to_insert, tokenizer.convert_ids_to_tokens(id_of_token_to_insert)), c_num, sub_score, sub_score + c_score])
            
        potential_substitutions = sorted(potential_substitutions, key=lambda x: x[3], reverse=True)[:beam_width]
        #print(potential_substitutions)
        #print('\n', potential_substitutions)
        for s in potential_substitutions:
            c_num = s[1]
            parent_in_beam = beam[c_num]
            parent_tokens = parent_in_beam[0]
            parent_score = parent_in_beam[1]
            parent_indexes_modified = parent_in_beam[2].copy()
            
            index_to_modify = s[0][0]
            id_of_token_to_insert = s[0][1]
            token_to_insert = tokenizer.convert_ids_to_tokens(id_of_token_to_insert)
            
            child_tokens = parent_tokens.copy()
            child_tokens[index_to_modify] = token_to_insert
            parent_indexes_modified.append(index_to_modify)
            child_score = parent_score + s[2]
            
            new_beam.append([child_tokens, child_score, parent_indexes_modified])
        beam = new_beam

    return tokens, extra_evals


def get_lm_head(model, head_data, n_epochs, tokenizer, text_list, gpu_device_num):
    if head_data[:6] == 'texts:':
        head_data = head_data[6:]
        use_provided_data = True
        data = text_list
        target_tokens = 1000000000
    else:
        data = []

    if isRoberta(model):
        lm_head_loc = 'cache_roberta/lm_heads/'
    else:
        lm_head_loc = 'cache_bert/lm_heads/'
    lm_head_path = lm_head_loc + head_data + '_lm_head.pth'
    lm_head_already_exists = True
    try:
        print("Loadeding lm head from:", lm_head_path)
        lm_head = torch.load(lm_head_path)
        print("lm head found!")
        return lm_head
    except:
        lm_head_already_exists = False
    #if use_wiki_data:
    #    wiki_data = pd.read_csv("wiki_2000000_words.tsv")
    #    wiki_data = wiki_data.sample(frac=1, random_state=0)
    #    target_tokens = int(head_data[5:])
    #    data = wiki_data['text'].values
    try:
        lm_head = torch.load(lm_head_loc + 'default_lm_head.pth')
    except:
        print("Failed to load default LM head from cache. Loading from pretrained model.")
        if isBert(model):
            m = transformers.BertForMaskedLM.from_pretrained("bert-base-uncased")
            lm_head = m.cls
        else:
            m = transformers.RobertaForMaskedLM.from_pretrained("roberta-base")
            lm_head = m.lm_head
        torch.save(lm_head, lm_head_loc + 'default_lm_head.pth')
        if head_data == 'default':
            return lm_head

    lm_head.cuda(gpu_device_num)
    model.lm_head = lm_head
    loss_fct = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(list(model.lm_head.parameters()))

    n_texts = len(data)
    id_tensor_list = []
    tokens_found = 0
    for text_loc in range(n_texts):
        text = data[text_loc]
        id_list = tokenizer.encode(text, add_special_tokens=True, truncation=True)
        id_tensor = torch.tensor([id_list]).cuda(gpu_device_num)
        tokens_added = len(id_list)
        if tokens_found + tokens_added <= target_tokens:
            id_tensor_list.append(id_tensor)
            tokens_found += tokens_added
        else:
            break
    print("Training LM head for", n_epochs, "epochs on", tokens_found, "tokens.")
    n_train_texts = len(id_tensor_list)
    
    for epoch in range(n_epochs):
        permute = np.random.permutation([i for i in range(n_train_texts)])
        avg_loss = 0
        n_toks_passed_to_LMH = 0
        for i in range(n_train_texts):
            t = permute[i]
            id_tensor = id_tensor_list[t]
            n_toks_passed_to_LMH += len(id_tensor[0])
            
            hidden_state = get_embeddings(model, id_tensor, gpu_device_num)
            final_hidden = onwards_token_train(hidden_state, model, gpu_device_num)
            predictions = model.lm_head(final_hidden)
            
            masked_lm_loss = loss_fct(predictions.view(-1, tokenizer.vocab_size), id_tensor.view(-1))
            masked_lm_loss.backward()
            opt.step()
            opt.zero_grad()
            avg_loss += masked_lm_loss.item()
    
        print("Epoch", epoch, "avg. loss per token:", avg_loss / n_toks_passed_to_LMH)
    torch.save(lm_head, lm_head_path)
    return lm_head

def get_saliency(sentiment_model, tokenizer, prob_pos, flip_target, tokens, ids, method, loss_fct, gpu_device_num):
    if method in ['norm_grad', 'norm_grad+', 'grad']:
        new_embeddings = get_embeddings(sentiment_model, ids, gpu_device_num).cuda(gpu_device_num).requires_grad_(True)
        embedding_opt = torch.optim.Adam([new_embeddings])
        initial_outputs = onwards(new_embeddings, 0, sentiment_model, gpu_device_num)
        loss = loss_fct(initial_outputs, torch.tensor([flip_target]).cuda(gpu_device_num).long())
        loss.backward()
        embedding_grad = new_embeddings.grad
        new_ids_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).cuda(gpu_device_num)
        word_embedding = get_word_embeddings(sentiment_model, new_ids_tensor).cuda(gpu_device_num)
        if method == 'norm_grad':
            logit_grads = torch.sum(torch.pow(word_embedding * embedding_grad, 2), dim=2)
        if method == 'norm_grad+':
            logit_grads = torch.sum(torch.pow(torch.max(word_embedding * embedding_grad, torch.tensor(0.0)), 2), dim=2)
        if method == 'grad':
            logit_grads = torch.sum(torch.pow(word_embedding * embedding_grad, 2), dim=1)
        return logit_grads, 1
    
    elif method in ['unk', 'del', 'zero', 'mask']:
        extra_evals = 0
        saliencies = torch.zeros_like(ids).float() - 10
        for i in range(1, len(ids[0]) - 1):
            mod_ids = ids.clone()
            if method == 'mask':
                mod_ids[0][i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
            else:
                print("Error: unknown saliency method:", method)
            
            pp_i = probability_positive(tokenizer, sentiment_model, mod_ids, gpu_device_num)
            extra_evals += 1
            pg_i = pp_to_pg(flip_target, prob_pos, pp_i)
            #print(pg_i)
            saliencies[0][i] = pg_i
        return saliencies, extra_evals
    elif method in ['random', 'rand']:
        saliencies = torch.randn(ids.size()).cuda(gpu_device_num)
        saliencies[0][0] = - 10.0
        saliencies[0][-1] = -10.0
        return saliencies, 0
    else:
        print("Error: unknown saliency method:", method)

def process_dataframe(df, sentiment_model, tokenizer, test_acc):
    df_list = []
    oC = 0
    n = 0
    for i in range(len(df)):
        text_o = df.iloc[i]['original_text']
        try:
            text_o = text_o.replace('[[[[Question]]]]: ', '')
            text_o = text_o.replace('[[[[Sentence]]]]: ', '')
            if isRoberta(sentiment_model):
                text_o = text_o.replace('>>>>', '</s></s>')
            elif isBert(sentiment_model):
                text_o = text_o.replace('>>>>', '[SEP]')
            text_o = text_o.replace('[[', '')
            text_o = text_o.replace(']]', '')
        except:
            pass
        
        if test_acc:
            trueL = df.iloc[i]['ground_truth_output']
            n_words += len(text_o.split(' '))
            id_list = tokenizer.encode(text_o, add_special_tokens=True, truncation=True)
            tokens = tokenizer.convert_ids_to_tokens(id_list)
            ppe = probability_positive(tokenizer, sentiment_model, tokens, gpu_device_num)
            
            if abs(trueL - ppe) < 0.5:
                oC += 1
            else:
                print(ppe, trueL)
                print(tokens)
            n += 1
        
        df_list.append(text_o)
    if test_acc:
        print(oC / n, n)
        print(n_words / len(df_list))
    return df_list


def hotflip_beamsearch_substitutions(all_word_embeddings, sentiment_model, tokenizer, loss_fct, beam_width, tree_depth, prob_left_early_stopping, flip_target, prob_pos, tokens, n_tokens, substitutions_dict, gpu_device_num):
    beam = [[tokens, 0, []]]
    extra_evals = 0
    for i in range(int(n_tokens * tree_depth)):
        final_result = []
        new_beam = []
        potential_substitutions = []
        for c_num in range(len(beam)):
            c = beam[c_num]
            c_tokens = c[0]
            c_score = c[1]
            c_indexes_modified = c[2]
            substitution_scores, candidate_prob_pos = compute_substitution_scores(all_word_embeddings, sentiment_model, tokenizer, loss_fct, flip_target, c_tokens, gpu_device_num)
            extra_evals += 1

            candidate_prob_left = pp_to_pl(flip_target, candidate_prob_pos)
            if candidate_prob_left < prob_left_early_stopping:
                return c_tokens, extra_evals
            for j in range(1, n_tokens - 1):
                if not j in c_indexes_modified:
                    for k in range(len(substitutions_dict[j]['substitutions'])):
                        
                        kth_sub_token = substitutions_dict[j]['substitutions'][k]
                        kth_sub_id = tokenizer.convert_tokens_to_ids(kth_sub_token)
                        kth_sub_score = substitution_scores[j][kth_sub_id].item()
                        
                        potential_substitutions.append([(j, kth_sub_id), c_num, kth_sub_score, kth_sub_score + c_score])
            
        potential_substitutions = sorted(potential_substitutions, key=lambda x: x[3], reverse=True)[:beam_width]
        for s in potential_substitutions:
            c_num = s[1]
            parent_in_beam = beam[c_num]
            parent_tokens = parent_in_beam[0]
            parent_score = parent_in_beam[1]
            parent_indexes_modified = parent_in_beam[2].copy()
            
            index_to_modify = s[0][0]
            id_of_token_to_insert = s[0][1]
            token_to_insert = tokenizer.convert_ids_to_tokens(id_of_token_to_insert)
            
            child_tokens = parent_tokens.copy()
            child_tokens[index_to_modify] = token_to_insert
            parent_indexes_modified.append(index_to_modify)
            child_score = parent_score + s[2]
            
            new_beam.append([child_tokens, child_score, parent_indexes_modified])
        beam = new_beam

    return tokens, extra_evals



