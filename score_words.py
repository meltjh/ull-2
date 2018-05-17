from operator import itemgetter
from scipy import spatial
import torch
from torch.autograd import Variable
import functions as fn
import sys

# Score the candidates based on the skip gram model.
def get_scores_sg(word_vectors, lst_dataset, candidates, word_to_id):
    all_scores_sorted = dict()

    num_sentence = 0
    for sentence_id, data in lst_dataset.items():
        num_sentence += 1
        sys.stdout.write('\rGoing through sentence {}/{} ({}%)'.format(num_sentence, len(lst_dataset), round(num_sentence/len(lst_dataset)*100, 2)))
        sys.stdout.flush()
        
        all_scores = dict()
        target_word_raw = data['target_word_raw']
        target_word = data['target_word']
        context_words = data['context_words']
        
        # Map target word to 'UNK' if it is not in the top words.
        if target_word not in word_to_id.keys():
            target_word_id = word_to_id['UNK']
        else:
            target_word_id = word_to_id[target_word]
            
        target_word_vector = word_vectors[target_word_id]
        target_word_candidates = candidates[target_word_raw]
        
        context_word_vectors = dict()
        for cw in context_words:
            cw_id = word_to_id[cw]
            cw_vector = word_vectors[cw_id]
            context_word_vectors[cw_id] = cw_vector
            
        for candidate in target_word_candidates:
            # Map candidate word to 'UNK' if it is not in the top words.
            if candidate in word_to_id:
                candidate_id = word_to_id[candidate]
                candidate_vector = word_vectors[candidate_id]
            else:
                unk_id = word_to_id['UNK']
                candidate_vector = word_vectors[unk_id]
            
            score = 0
            # Compute the cosine similarity between the target word and the candidate word.
            cos_candidate_target = 1 - spatial.distance.cosine(candidate_vector, target_word_vector)
            
            cos_candidate_context = 0
            # Compute the cosine similarity between the context words and the candidate word.
            for cw, cw_vector in context_word_vectors.items():
                cos_candidate_context += 1 - spatial.distance.cosine(candidate_vector, cw_vector)
            
            score = (cos_candidate_target + cos_candidate_context) / (len(context_word_vectors.keys()) + 1)
            all_scores[candidate] = score
        
        # Sort and store the scores.
        sorted_scores = sort_scores(all_scores)
        all_scores_sorted[(target_word_raw, sentence_id)] = sorted_scores
    return all_scores_sorted

# Score the candidates based on the Bayesian skip gram model.
def get_scores_bsg(modules, lst_dataset, candidates, word_to_id):
    mod_encoder = modules[0]
    mod_prior_mu = modules[1]
    lookup_prior_mu = mod_prior_mu.L.weight.data
    mod_prior_sigma = modules[2]
    lookup_prior_sigma = mod_prior_sigma.softplus(mod_prior_sigma.S.weight.data)
    
    all_scores_sorted = dict()
    num_sentence = 0
    for sentence_id, data in lst_dataset.items():
        num_sentence += 1
        sys.stdout.write('\rGoing through sentence {}/{} ({}%)'.format(num_sentence, len(lst_dataset), round(num_sentence/len(lst_dataset)*100, 2)))
        sys.stdout.flush()
        all_scores = dict()
        target_word_raw = data['target_word_raw']
        target_word = data['target_word']
        context_words = data['context_words']
        
        # Map target word to 'UNK' if it is not in the top words.
        if target_word not in word_to_id.keys():
            target_word_id = word_to_id['UNK']
        else:
            target_word_id = word_to_id[target_word]
        
        target_word_id = Variable(torch.LongTensor([target_word_id]))
        target_word_id_rep = target_word_id.repeat(2)

        context_word_ids = []
        for cw in context_words:
            cw_id = word_to_id[cw]
            context_word_ids.append(cw_id)

        context_word_ids = Variable(torch.LongTensor([context_word_ids]))
        context_word_ids_rep = context_word_ids.repeat(2, 1)
        
        # Get the posterior mu and sigma for the target word.
        posterior_mu_target, posterior_sigma_target = mod_encoder(target_word_id_rep, context_word_ids_rep)
        posterior_mu_target = posterior_mu_target[0]
        posterior_sigma_target = posterior_sigma_target[0]

        target_word_candidates = candidates[target_word_raw]
        # Map candidate word to 'UNK' if it is not in the top words.
        for candidate in target_word_candidates:
            if candidate in word_to_id:
                candidate_id = word_to_id[candidate]
            else:
                candidate_id = word_to_id['UNK']
            
            # Get the prior mu and sigma for the candidate word.
            prior_mu_candidate = lookup_prior_mu[candidate_id]
            prior_sigma_candidate = lookup_prior_sigma[candidate_id]
            
            # Compute the KL loss between the posterior of the target and the prior of the candidate.
            kl = fn.compute_kl(posterior_mu_target, posterior_sigma_target, prior_mu_candidate, prior_sigma_candidate)
            all_scores[candidate] = kl.item()

        # Sort and store the scores (ascending).
        sorted_scores = sort_scores(all_scores, False)
        all_scores_sorted[(target_word_raw, sentence_id)] = sorted_scores
    return all_scores_sorted

# Score the candidates based on the Embed Align model.
def get_scores_ea(modules, lst_dataset, candidates, word_to_id):
    mod_encoder = modules[0]
    mod_posterior_mu = modules[1]
    mod_posterior_sigma = modules[2]
    
    all_scores_sorted = dict()
    
    num_sentence = 0
    for sentence_id, data in lst_dataset.items():
        num_sentence += 1
        sys.stdout.write('\rGoing through sentence {}/{} ({}%)'.format(num_sentence, len(lst_dataset), round(num_sentence/len(lst_dataset)*100, 2)))
        sys.stdout.flush()

        all_scores = dict()
        target_word_raw = data['target_word_raw']
        target_word = data['target_word']
        sentence = data['sentence']
        position = data['position']
      
        # Map target word to 'UNK' if it is not in the top words.
        if target_word not in word_to_id.keys():
            target_word_id = word_to_id['UNK']
        else:
            target_word_id = word_to_id[target_word]
        
        # Get the ids of the words in the sentence.
        target_sentence = []
        for i in range(len(sentence)):
            if i == position:
                word_id = target_word_id
            else:
                word_id = word_to_id[sentence[i]]
            target_sentence.append(word_id)
            
        target_sentence = Variable(torch.LongTensor([target_sentence]))
        stacked_sentences = target_sentence[:]
        
        # Use the target word's sentence, but replace the target word by candidate's id.
        target_word_candidates = candidates[target_word_raw]
        candidate_str = []
        for candidate in target_word_candidates:
            candidate_str.append(candidate)
            if candidate in word_to_id:
                candidate_id = word_to_id[candidate]
            else:
                candidate_id = word_to_id['UNK']
            
            candidate_sentence = target_sentence[:]
            candidate_sentence[:, position] = candidate_id
            
            # Stack target sentence ids and candidates sentence ids so that it can be used as one batch.
            stacked_sentences = torch.cat([stacked_sentences, candidate_sentence], 0)
        
        # Get the posterior mus and sigmas for the target and candidate.
        h = mod_encoder(stacked_sentences)
        posterior_mu = mod_posterior_mu(h)
        posterior_sigma = mod_posterior_sigma(h)
    
        post_mu_target = posterior_mu[0]
        post_sigma_target = posterior_sigma[0]
        
        # Compute the KL divergence between the target and each candidate.
        for i in range(1, h.shape[0]):
            post_mu_candidate = posterior_mu[i]
            post_sigma_candidate = posterior_sigma[i]
            kl = fn.compute_kl(post_mu_target, post_sigma_target, post_mu_candidate, post_sigma_candidate)
            all_scores[candidate_str[i-1]] = kl.item()

        # Sort and store the scores (ascending).
        sorted_scores = sort_scores(all_scores, False)
        all_scores_sorted[(target_word_raw, sentence_id)] = sorted_scores
    return all_scores_sorted
        
# Sort the scores.
def sort_scores(sub_word_scores, reverse=True):
    return sorted(sub_word_scores.items(), key=itemgetter(1), reverse=reverse)