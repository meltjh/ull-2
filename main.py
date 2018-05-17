import skipgram_data as sk_d
import skipgram_model as sk_m
from skipgram_model import skip_gram
import score_words as sw
import functions as fn
import bayesian_skipgram_model as bsk_m
import bayesian_skipgram_data as bsk_d
import embed_align_data as ea_d
import embed_align_model as ea_m
import lst_data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# %% ===== READ IN DATA
#min_occurence = 25
#en, fr, top_words_en, top_words_fr = fn.read_data('hansards/training', min_occurence)

# %% ==== SKIP GRAM MODEL: train model ====


#context_window = 2
#epochs = 19
#batch_size = 500
#learning_rate = 0.001
#embedding_dim = 64
#load_model = False
#
## Preprocess data
#context, num_pairs, word_to_id, id_to_word = sk_d.get_pairs(en, context_window)
#
## Train the model
#model_file_name = "sg_model_final_{}_{}_{}_{}".format(num_pairs, batch_size, learning_rate, embedding_dim)
#model, epoch_data = sk_m.train_model(context, word_to_id, id_to_word, top_words_en, num_pairs, model_file_name, load_model, epochs, batch_size, learning_rate, embedding_dim)

## Visualize loss per epoch
#losses, times = zip(*epoch_data)
#
#plt.plot(losses)
#plt.show()
#
#plt.plot(times)
#plt.show()

# %% ==== SKIP GRAM MODEL: load existing model & perform LST ====
#
filename = "sg_model_final_8414748_500_0.001_64"
context_window = 2

checkpoint = torch.load(filename)
start_epoch = checkpoint['epoch']
vocab_size = checkpoint['vocab_size']
embedding_dim = checkpoint['embedding_dim']
model = skip_gram(vocab_size, embedding_dim, vocab_size)
model.load_state_dict(checkpoint['state_dict'])
epoch_data = checkpoint['epoch_data']
word_to_id = checkpoint['word_to_id']
id_to_word = checkpoint['id_to_word']
top_words_en = checkpoint['top_words_en']
skipgram_word_vectors = model.embedding.weight.data

lst_dataset = lst_data.read_lst_data_sg(context_window, top_words_en)
candidates, _ = lst_data.get_candidates()
scores = sw.get_scores_sg(skipgram_word_vectors, lst_dataset, candidates, word_to_id)
lst_data.write_scores('lst/lst.out_sg', scores)

# %% === BAYESIAN GRAM MODEL: train model ====

#context_window = 2
#epochs = 50
#batch_size = 500
#learning_rate = 0.001
#embedding_dim = 64
#load_model = False

# Preprocess data
#context, num_context, word_to_id, id_to_word = bsk_d.get_context(en, context_window)
#
## Train the model
#model_file_name = "bgs_model_{}_{}_{}_{}".format(num_context, batch_size, learning_rate, embedding_dim)  
#modules, epoch_data = bsk_m.train_model(context, word_to_id, id_to_word, top_words_en, num_context, model_file_name, load_model, epochs, batch_size, learning_rate, embedding_dim)

## Visualize the loss per epoch
#losses, times = zip(*epoch_data)
#
#plt.plot(losses)
#plt.show()
#
#plt.plot(times)
#plt.show()

# %% ==== SKIP GRAM MODEL: load existing model & perform LST ====

filename = "bgs_model_3918401_500_0.001_64"
context_window = 2

checkpoint = torch.load(filename)
start_epoch = checkpoint['epoch']
epoch_data = checkpoint['epoch_data']
word_to_id = checkpoint['word_to_id']
id_to_word = checkpoint['id_to_word']
top_words_en = checkpoint['top_words_en']
num_context = checkpoint['num_context']
embedding_dim = checkpoint['embedding_dim']
vocab_size = checkpoint['vocab_size']

# Initialize the models
mod_encoder = bsk_m.bsg_encoder(vocab_size, embedding_dim, num_context)
mod_prior_mu = bsk_m.bsg_prior_mu(vocab_size, embedding_dim)
mod_prior_sigma = bsk_m.bsg_prior_sigma(vocab_size, embedding_dim)
mod_decoder = bsk_m.bsg_decoder(vocab_size, embedding_dim)

# Combine them into the modules
modules = nn.ModuleList()
modules.append(mod_encoder)
modules.append(mod_prior_mu) 
modules.append(mod_prior_sigma) 
modules.append(mod_decoder)  
    
modules.load_state_dict(checkpoint['state_dict'])

lst_dataset = lst_data.read_lst_data_sg(context_window, top_words_en)
candidates, _ = lst_data.get_candidates()
scores = sw.get_scores_bsg(modules, lst_dataset, candidates, word_to_id)
lst_data.write_scores('lst/lst.out_bsg', scores)

# %% ==== EMBED ALIGN MODEL: train model ====

#epochs = 10
#batch_size = 500
#learning_rate = 0.01    
#embedding_dim = 64
#max_max_length = 20
#load_model = False
#
## Preprocess data
#context, num_pairs, word_to_id1, id_to_word1, word_to_id2, id_to_word2 = ea_d.get_context(en, fr, max_max_length)
#
## Train the model
#model_file_name = "ea_model_final20_{}_{}_{}_{}".format(num_pairs, batch_size, learning_rate, embedding_dim)  
#modules, epoch_data = ea_m.train_model(context, word_to_id1, id_to_word1, word_to_id2, id_to_word2, num_pairs, model_file_name, load_model, epochs, batch_size, learning_rate, embedding_dim)
#
## Visualize the loss per epoch
#losses, times = zip(*epoch_data)
#
#plt.plot(losses)
#plt.show()
#
#plt.plot(times)
#plt.show()

# %% ==== EMBED ALIGN MODEL: load existing model & perform LST ====

filename = "ea_model_final231164_50_0-1.01_64"

checkpoint = torch.load(filename)
start_epoch = checkpoint['epoch']
epoch_data = checkpoint['epoch_data']        
word_to_id1 = checkpoint['word_to_id1']
word_to_id2 = checkpoint['word_to_id2']
id_to_word1 = checkpoint['id_to_word1']
id_to_word2 = checkpoint['id_to_word2']
vocab_size1 = checkpoint['vocab_size1']
vocab_size2 = checkpoint['vocab_size2']
max_length_L1 = checkpoint['max_length_L1']
num_pairs = checkpoint['num_pairs']
embedding_dim = checkpoint['embedding_dim']

# Initialize the models
mod_encoder = ea_m.ea_encoder(vocab_size1, embedding_dim)
mod_posterior_mu = ea_m.ea_posterior_mu(embedding_dim)
mod_posterior_sigma = ea_m.ea_posterior_sigma(embedding_dim)
mod_decoder_L1 = ea_m.ea_decoder(vocab_size1, embedding_dim)
mod_decoder_L2 = ea_m.ea_decoder(vocab_size2, embedding_dim)

# Combine them into the modules
modules = nn.ModuleList()
modules.append(mod_encoder)
modules.append(mod_posterior_mu) 
modules.append(mod_posterior_sigma) 
modules.append(mod_decoder_L1)       
modules.append(mod_decoder_L2)
    
modules.load_state_dict(checkpoint['state_dict'])
top_words = word_to_id1.keys()
lst_dataset = lst_data.read_lst_data_ea(top_words, max_length_L1)
candidates, _ = lst_data.get_candidates()
scores = sw.get_scores_ea(modules, lst_dataset, candidates, word_to_id1)
lst_data.write_scores('lst/lst.out_ea_all', scores)
