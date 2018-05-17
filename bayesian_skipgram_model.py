import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import itertools
import os
import sys
import functions as fn

torch.manual_seed(0)
random.seed(0)

# Generator that returns (target_word_hv, context_word_hv) tuples.
def gen_data_id(train_data, word_to_id):
    data_pairs = []
    for target_word, contexts in train_data.items():
        target_word_id = word_to_id[target_word]
        for single_context in contexts:
            single_context_data = []
            for c_word in single_context:
                context_word_id = word_to_id[c_word]
                single_context_data.append(context_word_id)
            data_pairs.append((target_word_id, single_context_data))
    random.shuffle(data_pairs)

    for dp in data_pairs:
        yield dp

class bsg_encoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_window):
        super(bsg_encoder, self).__init__()
         
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        
        self.R = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.M = nn.Linear(2*self.embedding_dim, 2*self.embedding_dim)
        
        self.relu = nn.ReLU()
        
        self.U = nn.Linear(2*embedding_dim, embedding_dim)
        self.W = nn.Linear(2*embedding_dim, embedding_dim)
        self.softplus = nn.Softplus()
        
    def forward(self, target_w_id, context_w_ids):
        tw_embeds = self.R(target_w_id)  
        
        h = 0
        for cw_id in torch.squeeze(context_w_ids.transpose(0, 1)):
            context_w_id = cw_id
            cw_embeds = self.R(context_w_id)
            stacked = torch.cat([cw_embeds, tw_embeds], 1)
            projection = self.M(stacked)
            relu = self.relu(projection)

            h += relu

        posterior_mu = self.U(h)
        posterior_sigma = self.softplus(self.W(h))
        return (posterior_mu, posterior_sigma)

class bsg_prior_mu(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(bsg_prior_mu, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.L = nn.Embedding(self.vocab_size, self.embedding_dim)
         
    def forward(self, target_w_id):
        prior_mu = self.L(target_w_id)
        return prior_mu
  
      
class bsg_prior_sigma(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(bsg_prior_sigma, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.S = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.softplus = nn.Softplus()
        
    def forward(self, target_w_id):
        prior_sigma = self.softplus(self.S(target_w_id))
        
        return prior_sigma

class bsg_decoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(bsg_decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.Z_lin = nn.Linear(self.embedding_dim, self.vocab_size)
        self.softmax = nn.Softmax(dim=1)
 
    def forward(self, sampled_z):
        probs = self.softmax(self.Z_lin(sampled_z))
        return probs
    
def train_model(train_data, word_to_id, id_to_word, top_words_en, num_context, filename, load_model, epochs, batch_size, learning_rate, embedding_dim):
    vocab_size = len(word_to_id.keys())
    
    # Initialize the models
    mod_encoder = bsg_encoder(vocab_size, embedding_dim, num_context)
    mod_prior_mu = bsg_prior_mu(vocab_size, embedding_dim)
    mod_prior_sigma = bsg_prior_sigma(vocab_size, embedding_dim)
    mod_decoder = bsg_decoder(vocab_size, embedding_dim)
    
    # Combine them into the modules
    modules = nn.ModuleList()
    modules.append(mod_encoder)
    modules.append(mod_prior_mu) 
    modules.append(mod_prior_sigma) 
    modules.append(mod_decoder)        
    
    optimizer = optim.Adam(modules.parameters(), lr = learning_rate)    
    norm_dist = torch.distributions.normal.Normal(0, 1)
    
    # contains the losses and time per epoch
    epoch_data = []
    start_epoch = 0
    if load_model and os.path.isfile(filename):
        print("Loading checkpoint")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        modules.load_state_dict(checkpoint['state_dict'])
        epoch_data = checkpoint['epoch_data']
        optimizer.load_state_dict(checkpoint['optimizer'])
        word_to_id = checkpoint['word_to_id']
        id_to_word = checkpoint['id_to_word']
        top_words_en = checkpoint['top_words_en']
        num_context = checkpoint['num_context']
        embedding_dim = checkpoint['embedding_dim']
        vocab_size = checkpoint['vocab_size']

    print(modules)
    print("==========\nepochs: {}\nbatch_size: {}\nlearning_rate: {}\nvocab_size: {}\nembedding_dim: {}\nnum_context: {}\n==========\n\n".format(epochs, batch_size, learning_rate, vocab_size, embedding_dim, num_context))
    
    # For each epoch
    for epoch_i in range(start_epoch + 1, epochs + 1):    
        train_loss = 0
        print("\tEpoch: {}".format(epoch_i))
        
        data_generator = gen_data_id(train_data, word_to_id)
        start_time = time.time()      
        
        # For each batch
        amount_batches = int((num_context + batch_size - 1) / batch_size)
        for batch_i in range(amount_batches):
            modules.zero_grad()  
                    
            # It will also return the less than batchsize part at the end
            data_subset = list(itertools.islice(data_generator, batch_size))
            target_w_id, context_w_ids = zip(*data_subset)
        
            target_w_id = Variable(torch.LongTensor([target_w_id]))
            target_w_id = torch.squeeze(target_w_id)
            
            context_w_ids = torch.squeeze(Variable(torch.LongTensor([context_w_ids])))
            context_w_ids = torch.squeeze(context_w_ids)
 
            # Forward pass
            posterior_mu, posterior_sigma = mod_encoder(target_w_id, context_w_ids)
            prior_mu = mod_prior_mu(target_w_id)
            prior_sigma = mod_prior_sigma(target_w_id)
            
            # Sample noise
            cur_batch_size = posterior_sigma.shape[0]
            noise = norm_dist.sample((cur_batch_size, embedding_dim))
            
            # Sample z
            sampled_z = posterior_mu + torch.mul(posterior_sigma, noise)
            probs = mod_decoder(sampled_z)
            gat = probs.gather(1, context_w_ids)

            # Compute loss
            likelihood = torch.sum(torch.log(gat))
            kl = fn.compute_kl(posterior_mu, posterior_sigma, prior_mu, prior_sigma)
            loss = kl - likelihood

            # Backward pass
            loss.backward()
    
            # Update weights
            optimizer.step()
        
            train_loss += loss
            
            if (amount_batches > 100) and (batch_i%100) == 0:
                sys.stdout.write('\rLearning batch: {}/{} ({}%)'.format(batch_i, amount_batches, round(batch_i/amount_batches*100, 2)))
                sys.stdout.flush()
            
        time_diff = time.time()-start_time
        
        avg_loss = float(train_loss)/amount_batches
        print("Epoch %r: Train loss/sent=%.4f, Time=%.2fs" % 
              (epoch_i, avg_loss, time_diff))
        
        epoch_data.append((avg_loss, time_diff))
        
        # Save the model
        if os.path.isfile(filename):
            os.remove(filename)
        torch.save({
            'epoch': epoch_i,
            'state_dict': modules.state_dict(),
            'epoch_data': epoch_data,
            'optimizer': optimizer.state_dict(),
            'word_to_id': word_to_id,
            'id_to_word': id_to_word,
            'top_words_en': top_words_en,
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'num_context': num_context,
        }, filename)
    
    return modules, epoch_data