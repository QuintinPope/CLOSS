#import math
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
#import scipy
import copy
import sys
import time
import random
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from termcolor import colored
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from helpers import *
import argparse


def generate_flip_l_g_a(sentiment_model, LM_model, tokenizer, tokens, text, layer, hs_lr, group_tokens, root_reg,  l, extra_lasso, max_opt_steps, n_samples, topk, substitutions_after_loc, substitutions_after_SVs, min_substitutions_after_SVs, use_hard_scoring, min_substitutions, use_random_n_SV_substitutions, min_run_sample_size, use_grad_for_loc, max_SV_loc_evals, slowly_focus_SV_samples, min_SV_samples_per_sub, SV_samples_per_eval_after_location, logit_matix_source, use_SVs, use_exact, n_branches, tree_depth, beam_width, prob_left_early_stopping, substitution_gen_method, substitution_evaluation_method, saliency_method, gpu_device_num):

    start_time = time.time() ###
    model_evals = 0

    loss_fct = torch.nn.CrossEntropyLoss()
    ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).cuda(gpu_device_num).view(1,-1)
    n_tokens = len(tokens)
    max_SV_evals = int(min_SV_samples_per_sub * topk / SV_samples_per_eval_after_location)
    bias = F.one_hot(ids, tokenizer.vocab_size)
    
    hidden_state = get_embeddings(LM_model, ids, gpu_device_num).cuda(gpu_device_num)
    original_hidden_state = hidden_state.clone().detach()
    model_evals += 1
    prob_pos = probability_positive(tokenizer, sentiment_model, tokens, gpu_device_num)
    #print(prob_pos)
    found_flip = False
    restart = False
    if prob_pos > 0.5:
        flip_target = 0
    else:
        flip_target = 1
    
    if substitution_gen_method == 'no_opt_lmh':
        forward_prediction_target = - 1
    else:
        forward_prediction_target = flip_target
    opt_start_time = time.time()
    substitutions_dict = {}
    substitutions_locs_values = [[i, -2] for i in range(0, n_tokens)]
    for i in range(0, n_tokens):
        substitutions_dict[i] = {
            "index" : i,
            "loc_score" : 0,
            "substitutions" : [],
            "replacement_scores" : []
            }
        
    if substitution_gen_method in ['logits', 'no_opt_lmh', 'no_opt_e', 'random']:
        candidate_hidden_state = hidden_state.detach().requires_grad_(True)
        if substitution_gen_method == 'logits':
            opt = torch.optim.Adam([candidate_hidden_state], lr=hs_lr)
        
        for sample_n in range(n_samples):
            if substitution_gen_method == 'logits':
                for i in range(0, max_opt_steps):
                    outputs = onwards(candidate_hidden_state, layer, sentiment_model, gpu_device_num)
                    model_evals += 1
                    loss = loss_fct(outputs, torch.tensor([flip_target]).cuda(gpu_device_num).long())
                    loss_root = 0

                    numerical_stability_tensor = 0.0000001 * (0.5 - torch.rand_like(candidate_hidden_state))
                    if extra_lasso and 'start' in root_reg:
                        abs_diff_tensor = torch.sum(torch.abs(candidate_hidden_state - original_hidden_state + numerical_stability_tensor), axis=2)
                        loss_root += l * torch.sum(torch.sqrt(abs_diff_tensor)) / len(tokens)
                    elif 'squared' in root_reg:
                        squared_diff_tensor = \
                        torch.sum(torch.pow(candidate_hidden_state - original_hidden_state + numerical_stability_tensor, 2), axis=2)
                        if group_tokens:
                            loss_root = 0
                            word_loss = 0
                            for j in range(len(squared_diff_tensor)):
                                if tokens[j][0] == 'Ġ' or tokens[j] in ['<s>', '</s>']:
                                    # The token we're looking at is the start of a new word.
                                    if word_loss > 0:
                                        loss_root += torch.sqrt(word_loss)
                                    word_loss = squared_diff_tensor[j]
                                else:
                                    word_loss += squared_diff_tensor[j]
                        loss_root += l[0] * torch.sum(torch.sqrt(squared_diff_tensor)) / len(tokens)
                    elif 'start' in root_reg:
                        abs_diff_tensor = torch.abs(candidate_hidden_state - original_hidden_state + numerical_stability_tensor)
                        loss_root += l[0] * torch.sum(torch.sqrt(abs_diff_tensor)) / len(tokens)
                    if 'end' in root_reg:
                        new_token_logits = onwards_token_predict(candidate_hidden_state, layer, LM_model, forward_prediction_target, gpu_device_num)[0]
                        model_evals += 1
                        numerical_stability_tensor = 0.000001 * (0.5 - torch.rand_like(new_token_logits))
                        new_token_probs = torch.functional.F.softmax(new_token_logits, dim=1)
                        id_token_probs = new_token_probs.gather(1, ids)


                        loss_root += l[1] * torch.sum(torch.abs(0.05 + 1 - id_token_probs))
                        #loss_root += l * torch.sum(torch.sqrt(torch.abs(new_token_logits - initial_token_logits + numerical_stability_tensor)))
                    if 'cubic' in root_reg:
                        abs_diff_tensor = torch.abs(candidate_hidden_state - original_hidden_state + numerical_stability_tensor)
                        loss_root += l[0] * torch.sum(torch.pow(abs_diff_tensor, 1/3))
                   
                    if loss_root.item() > 0 and not torch.isnan(loss_root):
                        loss += loss_root
                    if torch.isnan(loss) or torch.isnan(loss_root):
                        print(loss_root, loss)
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Error: NaN loss!\n\n\nRestarting:")
                        return generate_flip(sentiment_model, LM_model, tokenizer, -1, target, layer, root_reg, l, extra_lasso, gpu_device_num)
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    with torch.no_grad():
                        candidate_hidden_state[:, 0, :] = original_hidden_state[:, 0, :]
                        candidate_hidden_state[:, -1, :] = original_hidden_state[:, -1, :]
            if logit_matix_source == 'embeddings' or logit_matix_source == 'embedding':
                nn_embedding = (candidate_hidden_state - position_embedding)[0].detach().requires_grad_(False)
                token_distances = torch.zeros((len(ids), tokenizer.vocab_size))
                for token_loc in range(len(ids)):
                    nn_magnitudes = l1_nearest_neighbor(nn_embedding[token_loc], all_word_embeddings)
                    token_distances[token_loc, :] = nn_magnitudes
                sample_logits = -token_distances.cuda(gpu_device_num)
            elif substitution_gen_method == 'random':
                sample_logits = torch.rand((n_tokens, tokenizer.vocab_size))
            else:
                sample_logits = onwards_token_predict(candidate_hidden_state, layer, LM_model, forward_prediction_target, gpu_device_num)[0]


            if topk > 0:
                _, topk_alternatives = torch.topk(sample_logits, k=1+topk, sorted=True)
                topk_alternatives = topk_alternatives.tolist()
                topk_substitutions = [tokenizer.convert_ids_to_tokens(topk_alternatives[i]) for i in range(len(topk_alternatives))]
            
            for i in range(1, n_tokens - 1):
                for j in range(1+topk):
                    replacement_from_topk = topk_substitutions[i][j]
                    if (replacement_from_topk == tokens[i]) or (replacement_from_topk in substitutions_dict[i]['substitutions']):
                        continue
                    else:
                        substitutions_dict[i]['substitutions'].append(replacement_from_topk)
                        #                                           8 avg value of samples that include this substitution
                        #                                                          7 SV estimate for this substitution  |
                        #                                        6 number of samples that include this substitution  |  |
                        #                            5 list of values of samples that include this substitution   |  |  |
                        #                                                              4 local mod prob gain  |   |  |  |
                        #                                                              3 avg sv loc score  |  |   |  |  |
                        #                                                                2 n loc evals  |  |  |   |  |  |
                        #                                                            1 sv loc score  |  |  |  |   |  |  |
                        substitutions_dict[i]['replacement_scores'].append([replacement_from_topk, 0, 0, 0, 0, [], 0, 0, 0])
                        break
    opt_end_time = time.time()    
    substitution_start_time = time.time()
    extra_evals = 0
    
    if substitution_evaluation_method in ['hotflip_only']:
        best_candidate_tokens, extra_evals = hotflip_beamsearch(all_word_embeddings, sentiment_model, tokenizer, loss_fct, beam_width, tree_depth, prob_left_early_stopping, topk, flip_target, prob_pos, tokens, n_tokens, gpu_device_num)
        model_evals += extra_evals
        substitution_end_time = time.time()
        
    if substitution_evaluation_method in ['hotflip']:
        best_candidate_tokens, extra_evals = hotflip_beamsearch_substitutions(all_word_embeddings, sentiment_model, tokenizer, loss_fct, beam_width, tree_depth, prob_left_early_stopping, flip_target, prob_pos, tokens, n_tokens, substitutions_dict, gpu_device_num)
        model_evals += extra_evals
        substitution_end_time = time.time()
    print("Extra Evals:", extra_evals)
    
    if substitution_evaluation_method in ['SVs', 'grad-only', 'grad_only']:
                                
        
        logit_grads, extra_evals = get_saliency(sentiment_model, tokenizer, prob_pos, flip_target, tokens, ids, saliency_method, loss_fct, gpu_device_num)
        model_evals += extra_evals

        with torch.no_grad():
            if use_grad_for_loc:
                print_token_importances(sentiment_model, logit_grads[0], tokens, n_tokens, "grad loc importances:", gpu_device_num)
                for i in range(1, n_tokens - 1):
                    replacement_options = substitutions_dict[i]['replacement_scores']
                    if tokens[i] in ['[SEP]', '[CLS]', '</s>', '<s>']:
                        substitutions_dict[i]['loc_score'] = -10
                        substitutions_locs_values[i][1] = -10
                    else:
                        substitutions_dict[i]['loc_score'] = logit_grads[0][i].item()
                        substitutions_locs_values[i][1] = substitutions_dict[i]['loc_score']
                SV_loc_prob_gain = 0
                
            n_evals_all_substitutions = 0
            all_substitutions_total_value = 0
            total_SV = []
                

            n_substitutions_after_location_SVs = max(int(substitutions_after_loc * n_tokens), 2)
            n_substitutions_after_location_SVs = min(n_substitutions_after_location_SVs, len(substitutions_locs_values) - 2)
            substitutions_locs_values = sorted(substitutions_locs_values, key=lambda x: x[1], reverse=True)
            location_score_cutoff = substitutions_locs_values[n_substitutions_after_location_SVs][1]
            
            # Compute approximate SV values for each substitution:

            important_replacement_locs = [substitutions_locs_values[i][0] for i in range(n_substitutions_after_location_SVs)]
            print(important_replacement_locs)
            location_sampling_weights = [len(substitutions_dict[i]['replacement_scores']) for i in important_replacement_locs]
            n_subs = sum(location_sampling_weights)
            location_sampling_weights = [w / n_subs for w in location_sampling_weights]
            
            tokens_sampled = int(n_substitutions_after_location_SVs * SV_samples_per_eval_after_location)
            if substitution_evaluation_method in ['SVs']:
                for i in range(max_SV_evals):
                    run_sample_size = tokens_sampled
                    sample = list(np.random.choice(important_replacement_locs, size=run_sample_size, p=location_sampling_weights))
                    eval_tokens = tokens.copy()
                    replacement_inner_indicies = []
                    for s in sample:
                        if s == 0 or s >= n_tokens - 1:
                            replacement_inner_indicies.append(-1)
                            print("Error: sampled end tokens")
                            continue
                        
                        replacement_options = substitutions_dict[s]['replacement_scores']
                        inner_index = random.randint(0, len(replacement_options) - 1)
                        
                        replacement_inner_indicies.append(inner_index)
                        eval_tokens[s] = replacement_options[inner_index][0]
                    SV_eval_prob_pos = probability_positive(tokenizer, sentiment_model, eval_tokens, gpu_device_num)
                    model_evals += 1
                    SV_eval_prob_gain = pp_to_pg(flip_target, prob_pos, SV_eval_prob_pos)
                    for j in range(len(replacement_inner_indicies)):
                        s = sample[j]
                        if s == 0 or s >= n_tokens - 1:
                            continue
                        inner_index = replacement_inner_indicies[j]
                        substitutions_dict[s]['replacement_scores'][inner_index][5].append(SV_eval_prob_gain)
                        substitutions_dict[s]['replacement_scores'][inner_index][6] += 1

                for i in range(1, n_tokens - 1):
                    replacement_options = substitutions_dict[i]['replacement_scores']
                    for j in range(len(replacement_options)):
                        if replacement_options[j][6] != 0:
                            replacement_options[j][8] = sum(replacement_options[j][5])
                            all_substitutions_total_value += replacement_options[j][8]
                            n_evals_all_substitutions += replacement_options[j][6]
                for i in range(1, n_tokens - 1):
                    replacement_options = substitutions_dict[i]['replacement_scores']
                    for j in range(len(replacement_options)):
                        if replacement_options[j][6] != 0:
                            substitution_total_value = replacement_options[j][8]
                            n_evals_with_substitution = replacement_options[j][6]
                            n_evals_without_substitution = n_evals_all_substitutions - n_evals_with_substitution
                            
                            avg_value_with_sub = substitution_total_value / n_evals_with_substitution
                            avg_value_without_sub = (all_substitutions_total_value - substitution_total_value) / n_evals_without_substitution
                            
                            replacement_options[j][7] = avg_value_with_sub - avg_value_without_sub
                            total_SV.append(replacement_options[j][7])
            
            substitutions = []
            
            cutoff = min(len(total_SV), int(n_tokens * substitutions_after_SVs))
            total_SV = sorted(total_SV, reverse=True)[:cutoff]
            
            print("\ntotal SVs   =", sum(total_SV))

            for i in range(1, n_tokens - 1):
                if substitutions_dict[i]['loc_score'] <= location_score_cutoff:
                    continue
                replacement_options = substitutions_dict[i]['replacement_scores']
                for j in range(len(replacement_options)):
                    if substitution_evaluation_method in ['grad-only', 'grad_only']:
                        sv_score = substitutions_dict[i]['loc_score']
                    elif substitution_evaluation_method in ['SVs']:
                        sv_score = replacement_options[j][7]
                    substitutions.append([i, replacement_options[j][0], sv_score])
            substitutions = sorted(substitutions, key=lambda x: x[2], reverse=True)
            l = min(8, len(substitutions) - 1)
            
            print("Top scoring substitutions by Shapley value:")
            for r in substitutions[:l]:
                print(r)
            substitution_end_time = time.time()
            
            greedy_start_time = time.time()
            
            best_candidate_tokens = []
            best_candidate_score = 0
            
            
            if tree_depth > 0:
                substitutions_already_tested = []
                counterfactuals_generated = []
                #             counterfactual tokens-. sorted substitutions used in counterfactual-.  sorted indexes changed in counterfactual  score
                #                                   |                                             |   |                                        |
                #                                   v                                             v   v                                        v
                current_level_counterfactuals = [[tokens,                                         [], [],                                      1]]
                substitutions_sorted_by_score = sorted(substitutions, key=lambda x: x[2], reverse=True)
                early_stop = False
                
                max_treesearch_substitutions = max(3, int(n_tokens * tree_depth))
                
                for depth in range(max_treesearch_substitutions):
                    if early_stop:
                        break
                    next_level_counterfactuals = []
                    for node_n in range(len(current_level_counterfactuals)):
                        if early_stop:
                            break
                        parent_node = current_level_counterfactuals[node_n]
                        parent_node_tokens = parent_node[0]
                        parent_node_substitutions = parent_node[1]
                        parent_node_indexes_modified = parent_node[2]
                        n_branches_added_to_node_n = 0
                        
                        for possible_substitution_n in range(len(substitutions)):
                            child_node_substitutions = parent_node_substitutions.copy()
                            child_node_substitutions.append(possible_substitution_n)
                            child_node_substitutions = sorted(child_node_substitutions)
                            if child_node_substitutions in substitutions_already_tested:
                                #print("\t\t\tskip due to already seen")
                                continue
                            
                            possible_substitution = substitutions_sorted_by_score[possible_substitution_n]
                            index_that_would_be_modified_by_possible_substitution = possible_substitution[0]
                            token_being_substituted_into_child = possible_substitution[1]
                            if index_that_would_be_modified_by_possible_substitution in parent_node_indexes_modified:
                                #print("\t\t\tskip due to index already changed")
                                continue
                            child_node_indexes_modified = parent_node_indexes_modified.copy()
                            child_node_indexes_modified.append(index_that_would_be_modified_by_possible_substitution)
                            child_node_indexes_modified = sorted(child_node_indexes_modified)
                            
                            child_node_tokens = parent_node_tokens.copy()
                            child_node_tokens[index_that_would_be_modified_by_possible_substitution] = token_being_substituted_into_child
                            
                            n_tokens_changed = len(child_node_indexes_modified)
                            child_prob_pos = probability_positive(tokenizer, sentiment_model, child_node_tokens, gpu_device_num)
                            model_evals += 1
                            child_prob_gain = pp_to_pg(flip_target, prob_pos, child_prob_pos)
                            child_prob_left = pp_to_pl(flip_target, child_prob_pos)
                            child_score = child_prob_gain + 1 - n_tokens_changed / n_tokens
                            
                            child_node = [child_node_tokens, child_node_substitutions, child_node_indexes_modified, child_score]
                            
                            substitutions_already_tested.append(child_node_substitutions)
                            next_level_counterfactuals.append(child_node)
                            counterfactuals_generated.append(child_node)
                            if child_prob_left < prob_left_early_stopping:
                                early_stop = True
                                break
                            
                            n_branches_added_to_node_n += 1
                            
                            if n_branches_added_to_node_n >= n_branches:
                                break
                    cutoff = min(len(next_level_counterfactuals), beam_width)
                    next_level_counterfactuals = sorted(next_level_counterfactuals, key=lambda x: x[3], reverse=True)[:cutoff]
                    current_level_counterfactuals = next_level_counterfactuals        
            
                best_node = []
                for generated_counterfactual in counterfactuals_generated:
                    modified_toks = generated_counterfactual[0]
                    candidate_score = generated_counterfactual[3]
                    if candidate_score > best_candidate_score:
                        best_candidate_tokens = modified_toks
                        best_candidate_score = candidate_score
                        best_node = generated_counterfactual
                
            greedy_end_time = time.time()
    
                    
    if not substitution_evaluation_method in ['SVs']:
        greedy_start_time = greedy_end_time = 0
    input_tokens_prob_pos = probability_positive(tokenizer, sentiment_model, best_candidate_tokens, gpu_device_num)
    print("Final eval prob pos:", input_tokens_prob_pos)
    model_evals += 1
    if flip_target == 1:
        input_tokens_prob_gain = input_tokens_prob_pos - prob_pos
        prob_left = 1 - input_tokens_prob_pos
    else:
        input_tokens_prob_gain = prob_pos - input_tokens_prob_pos
        prob_left = input_tokens_prob_pos
    sameness_list = [0 for i in range(n_tokens)]
    
    for i in range(n_tokens):
        if best_candidate_tokens[i] == tokens[i]:
            sameness_list[i] = 1
            
    frac_tokens_same = sum(sameness_list) / n_tokens
    print(sum(sameness_list), n_tokens)

    old_tokens_string = ''
    new_tokens_string = ''
    for tok_loc in range(n_tokens):
        old_tok = tokens[tok_loc]
        new_tok = best_candidate_tokens[tok_loc]
        seperator_size = max(len(old_tok), len(new_tok))
        spacechar = ' '
        skippainttok = False
        if isRoberta(sentiment_model):
            spacechar = ''
        elif (len(new_tok) > 2 and new_tok[0] == '#' and new_tok[1] == '#'):
            spacechar = ''
            new_tok = new_tok[2:]
            skippainttok = True
        if old_tok == new_tok or skippainttok:
            old_tokens_string += spacechar + old_tok.ljust(seperator_size, ' ')
            new_tokens_string += spacechar + new_tok.ljust(seperator_size, ' ')
        else:
            old_tokens_string += spacechar + colored(old_tok.ljust(seperator_size, ' '), 'red')
            new_tokens_string += spacechar + colored(new_tok.ljust(seperator_size, ' '), 'red')
    print("Old tokens           :", old_tokens_string.replace('Ġ', ' ').replace('##', ''))
    print("New tokens           :", new_tokens_string.replace('Ġ', ' ').replace('##', ''))
    print("Best prob gain       :", round(input_tokens_prob_gain, 3))
    print("Fraction toks same   :", round(frac_tokens_same, 3))

    if prob_left < 0.5:
        found_flip = True
    return sameness_list, found_flip, frac_tokens_same, -1, -1, tokenizer.convert_tokens_to_string(best_candidate_tokens), tokens, best_candidate_tokens, [0, 0, opt_end_time - opt_start_time, substitution_end_time - substitution_start_time, greedy_end_time - greedy_start_time], model_evals


def evaluate_list(text_list, sentiment_model, LM_model, n_epochs, attempts, tokenizer, hs_lr, group_tokens, root_reg, l, extra_lasso, max_opt_steps, n_samples, topk, substitutions_after_loc, substitutions_after_SVs, min_substitutions_after_SVs, use_hard_scoring, min_substitutions, use_random_n_SV_substitutions, min_run_sample_size, use_grad_for_loc, max_SV_loc_evals, slowly_focus_SV_samples, min_SV_samples_per_sub, SV_samples_per_eval_after_location, logit_matix_source, use_SVs, use_exact, n_branches, tree_depth, beam_width, prob_left_early_stopping, substitution_gen_method, substitution_evaluation_method, saliency_method, empty_cache_every_text, logname, data_len_str):

    result_log = []
    all_results = []
    all_embeddings = []
    setup_time = gradient_time = opt_time = substitution_time = greedy_time = total_attempts = texts_tried = total_final_perplexity = total_final_bleu = total_initial_perplexity = total_flips_found = 0
    n_texts = len(text_list)
    #            p2n n2p n2n p2p
    CM_flips  = [0,  0,  0,  0]
    CM_bleu   = [0,  0,  0,  0]
    CM_f_perp = [0,  0,  0,  0]
    CM_i_perp = [0,  0,  0,  0]
    CM_change = [0,  0,  0,  0]
    CM_evals  = [0,  0,  0,  0]
    
    all_flip_diameters = []

    for text_loc in tqdm(range(n_texts)):
        text = text_list[text_loc]
        try:
            initial_perplexity = 1#perplexity(text)
            total_initial_perplexity += initial_perplexity
        except:
            print("Error on", text)
            initial_perplexity = 0
        
        
        skiptxt = False
        total_attempts = 0
        
        id_list = tokenizer.encode(text, add_special_tokens=True, truncation=True)
        tokens = tokenizer.convert_ids_to_tokens(id_list)

        #print("Initial perplexity   :", round(initial_perplexity, 3))
        initial_PP = probability_positive(tokenizer, sentiment_model, tokens, gpu_device_num)
        print("Initial prob positive:", round(initial_PP, 5))
        results_list = []
        tmp_opt_time = tmp_substitution_time = tmp_greedy_time = tmp_setup_time = tmp_gradient_time = 0
        for i in range(attempts):
            
            change_indexes, found_flip, frac_tokens_same, frac_words_same, embedding, new_text, old_tokens, new_tokens, all_times, model_evals = generate_flip_l_g_a(sentiment_model, LM_model, tokenizer, tokens, text, 0, hs_lr, group_tokens, root_reg, l, extra_lasso, max_opt_steps, n_samples, topk, substitutions_after_loc, substitutions_after_SVs, min_substitutions_after_SVs, use_hard_scoring, min_substitutions, use_random_n_SV_substitutions, min_run_sample_size, use_grad_for_loc, max_SV_loc_evals, slowly_focus_SV_samples, min_SV_samples_per_sub, SV_samples_per_eval_after_location, logit_matix_source, use_SVs, use_exact, n_branches, tree_depth, beam_width, prob_left_early_stopping, substitution_gen_method, substitution_evaluation_method, saliency_method, gpu_device_num)
            
        
            if empty_cache_every_text:
                torch.cuda.empty_cache()
            tmp_setup_time += all_times[0]
            tmp_gradient_time += all_times[1]
            tmp_opt_time += all_times[2]
            tmp_substitution_time += all_times[3]
            tmp_greedy_time += all_times[4]
            
            bleu = sentence_bleu([old_tokens], new_tokens)
            new_PP = probability_positive(tokenizer, sentiment_model, new_tokens, gpu_device_num)
            if initial_PP > 0.5 and new_PP < 0.5 or initial_PP < 0.5 and new_PP > 0.5:
                flip_found = True
            else:
                flip_found = False
            results_list.append([change_indexes, found_flip, frac_tokens_same, [embedding, initial_PP], new_text, bleu])

            total_attempts += 1
            if found_flip and frac_tokens_same > 0.9:
                break
        if skiptxt:
            continue

        first_flip_loc = last_flip_loc = 0
        n_tokens = len(new_tokens)
        for i in range(n_tokens):
            if new_tokens[i] != tokens[i]:
                first_flip_loc = i
                break
        for i in range(1, n_tokens):
            if new_tokens[n_tokens - i] != tokens[n_tokens - i]:
                last_flip_loc = n_tokens - i
                break
        all_flip_diameters.append([first_flip_loc, last_flip_loc, n_tokens, int(found_flip), round(frac_tokens_same, 6), model_evals + 2])
        
        texts_tried += 1
        setup_time += tmp_setup_time / total_attempts
        gradient_time += tmp_gradient_time / total_attempts
        opt_time += tmp_opt_time / total_attempts
        substitution_time += tmp_substitution_time / total_attempts
        greedy_time += tmp_greedy_time / total_attempts

        results_list = sorted(results_list, key=lambda x: x[2] * x[1], reverse=True)
        all_results.append(results_list[0])
        found_flip = results_list[0][1]
        final_perplexity = -1#perplexity(tokenizer.convert_tokens_to_string(new_tokens[1:-1]))
                             # more efficient to compute perplexity later; no need to keep GPT model in memory
        final_bleu = results_list[0][5]
        if found_flip:
            total_final_bleu += final_bleu
            total_final_perplexity += final_perplexity
            total_flips_found += 1
        
        if initial_PP > 0.5 and new_PP < 0.5:
            CM_flips[0] += 1
            CM_bleu[0] += bleu
            CM_f_perp[0] += final_perplexity
            CM_i_perp[0] += initial_perplexity
            CM_change[0] += frac_tokens_same
            CM_evals[0] += model_evals + 2
        if initial_PP < 0.5 and new_PP > 0.5:
            CM_flips[1] += 1
            CM_bleu[1] += bleu
            CM_f_perp[1] += final_perplexity
            CM_i_perp[1] += initial_perplexity
            CM_change[1] += frac_tokens_same
            CM_evals[1] += model_evals + 2
        if initial_PP < 0.5 and not found_flip:
            CM_flips[2] += 1
            CM_bleu[2] += bleu
            CM_f_perp[2] += final_perplexity
            CM_i_perp[2] += initial_perplexity
            CM_change[2] += frac_tokens_same
            CM_evals[2] += model_evals + 2
        if initial_PP > 0.5 and not found_flip:
            CM_flips[3] += 1
            CM_bleu[3] += bleu
            CM_f_perp[3] += final_perplexity
            CM_i_perp[3] += initial_perplexity
            CM_change[3] += frac_tokens_same
            CM_evals[3] += model_evals + 2
        
        CM_bleu_print   = [0,  0,  0,  0]
        CM_f_perp_print = [0,  0,  0,  0]
        CM_i_perp_print = [0,  0,  0,  0]
        CM_change_print = [0,  0,  0,  0]
        CM_evals_print  = [0,  0,  0,  0]
        
        for i in range(4):
            CM_bleu_print[i] = round(CM_bleu[i] / max(1, CM_flips[i]), 3)
            CM_f_perp_print[i] = round(CM_f_perp[i] / max(1, CM_flips[i]), 3)
            CM_i_perp_print[i] = round(CM_i_perp[i] / max(1, CM_flips[i]), 3)
            CM_change_print[i] = round(CM_change[i] / max(1, CM_flips[i]), 3)
            CM_evals_print[i] = round(CM_evals[i] / max(1, CM_flips[i]), 3)
        
        print("Results by flip type:")
        print(CM_flips)
        print(CM_bleu_print)
        print(CM_f_perp_print)
        print(CM_i_perp_print)
        print(CM_change_print)
        print(CM_evals_print)
        
        print("Changed perplexity   :", round(final_perplexity, 3))
        print("Changed BLEU score   :", round(final_bleu, 3))


        print("Flip found:", found_flip)
        print()
        print("Total perplexity     :", round(total_final_perplexity, 3))
        print("Total bleu           :", round(total_final_bleu, 3))
        print("flips found / texts tried: " + str(total_flips_found) + " / " + str(texts_tried) + " : " + str(round(total_flips_found / texts_tried, 4)))
        print('\n\n')
        
        result_log.append([text, new_text, n_tokens, tmp_gradient_time, tmp_opt_time, tmp_substitution_time, tmp_greedy_time, model_evals + 2, bleu, frac_tokens_same, frac_words_same, found_flip, initial_PP, new_PP, first_flip_loc, last_flip_loc])
        

    print("\n####################################################################################################################\n")
    print("\n####################################################################################################################\n")
    print("\n####################################################################################################################\n")
    print(all_flip_diameters)

    avg_sameness = 0
    n_texts_changed = 0
    total_bleu = 0
    for r in all_results:
        if r[1]:
            all_embeddings.append(r[3])
            n_texts_changed += 1
            avg_sameness += r[2]
            total_bleu += r[5]
    print("Attempted change for", texts_tried, "texts.")
    print("Changed sentiment for", n_texts_changed, "texts.")
    print("Average token match frac among changed texts =", round(avg_sameness / n_texts_changed, 3))
    print("Average perplexity for original input texts = ", round(total_initial_perplexity / n_texts_changed, 3))
    print("Average perplexity for generated counterfactuals =", round(total_final_perplexity / n_texts_changed, 3))
    print("Average BLEU for generated counterfactuals =", round(total_bleu / n_texts_changed, 3))
    #for r in all_results:
    #    print(r[:2])
    total_time = setup_time + gradient_time + opt_time + substitution_time + greedy_time
    print("Avg. setup time   :", round(setup_time / texts_tried, 3))
    print("Avg. gradient time:", round(gradient_time / texts_tried, 3))
    print("Avg. opt time     :", round(opt_time / texts_tried, 3))
    print("Avg. subst. time  :", round(substitution_time / texts_tried, 3))
    print("Avg. greedy time  :", round(greedy_time / texts_tried, 3))
    print("Avg. total time   :", round(total_time / texts_tried, 3))
    print("Total time        :", round(total_time, 3))
    tot_e = 0
    for i in all_flip_diameters:
        tot_e += i[5]
    print("Avg. evals             :", round(tot_e / 1000, 3))
    f = open("text_logs/" + logname + ".txt", 'w')
    f.write("Attempted change for " + str(texts_tried) + " texts.")
    f.write("\nChanged sentiment for " + str(n_texts_changed) + " texts.")
    f.write("\nAverage token match frac among changed texts = " + str(round(avg_sameness / n_texts_changed, 3)))
    f.write("\nAverage perplexity for original input texts =  " + str(round(total_initial_perplexity / n_texts_changed, 3)))
    f.write("\nAverage perplexity for generated counterfactuals = " + str(round(total_final_perplexity / n_texts_changed, 3)))
    f.write("\nAverage BLEU for generated counterfactuals = " + str(round(total_bleu / n_texts_changed, 3)))
    #for r in all_results:
    #    print(r[:2])
    f.write("\nAvg. setup time    : " + str(round(setup_time / texts_tried, 3)))
    f.write("\nAvg. gradient time : " + str(round(gradient_time / texts_tried, 3)))
    f.write("\nAvg. opt time      : " + str(round(opt_time / texts_tried, 3)))
    f.write("\nAvg. subst. time   : " + str(round(substitution_time / texts_tried, 3)))
    f.write("\nAvg. greedy time   : " + str(round(greedy_time / texts_tried, 3)))
    f.write("\nAvg. total time    : " + str(round(total_time / texts_tried, 3)))
    f.write("\nTotal time         : " + str(round(total_time, 3)))
    f.write("\nAvg. evals         : " + str(round(tot_e / 1000, 3)))
    
    output_string = "{" + str(beam_width) + ", " + str(min_SV_samples_per_sub) + ", " + str(topk) + ", {" + str(round(total_time / texts_tried, 5)) + ", " + str(round(tot_e / texts_tried, 5)) + ", " + str(round(total_initial_perplexity / n_texts_changed, 5)) + ", " + str(round(total_final_perplexity / n_texts_changed, 5)) + ", " + str(round((total_final_perplexity / total_initial_perplexity) / n_texts_changed, 5)) + ", " + str(round(total_bleu / n_texts_changed, 5)) + ", " + str(round(avg_sameness / n_texts_changed, 5)) + ", " + str(round(n_texts_changed / texts_tried, 5))  + "}, \"" + data_len_str + "_" + substitution_gen_method + "_" + substitution_evaluation_method + "\"}"
            
    python_string = output_string.replace('{', '[').replace('}', ']')
    
    print(output_string)
    print(python_string)

    output_df = pd.DataFrame(result_log, columns=["Original_text", "New_text", "N_tokens", "Grad_time", "Opt_time", "Subst_time", "Beamsearch_time", "N_evals", "Bleu", "Frac_tokens_same", "Frac_words_same", "Found_flip", "Original_prob_positive", "New_prob_positive", "First_flip_loc", "Last_flip_loc"])
    output_df.to_csv("tsv_logs/" + logname + ".tsv", sep='\t')
    
    f.write("\n" + output_string + "\n" + python_string + '\n')
    f.close()
    
    return


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
