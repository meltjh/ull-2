import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import itertools
import sys
import os
import functions as fn

torch.manual_seed(0)
random.seed(0)

# Generator that returns (sentence_L1, sentence_L2) tuples.
def gen_data_id(train_data, word_to_id1, word_to_id2):
    data_pairs = []
    
    for sentence_L1, sentence_L2, length_L1, length_L2 in train_data:
        sentence_ids1 = sentence_to_ids(sentence_L1, word_to_id1)
        sentence_ids2 = sentence_to_ids(sentence_L2, word_to_id2)

        data_pairs.append((sentence_ids1, sentence_ids2, length_L1, length_L2))
    
    random.shuffle(data_pairs)

    for dp in data_pairs:
        yield dp

# Get the id sequence of the sentence.
def sentence_to_ids(sentence, word_to_id):
    sentence_ids = []
    for word in sentence:
        word_id = word_to_id[word]
        sentence_ids.append(word_id)
    return sentence_ids

class ea_encoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(ea_encoder, self).__init__()
         
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        # Gives 2*embedding_dim output.
        self.bilstm = nn.LSTM(self.embedding_dim, self.embedding_dim, bidirectional=True)
        
    def forward(self, sentences):
        snt_embeds = self.embedding(sentences)

        # Gives 2*embedding_dim output.
        snt_embeds, _ = self.bilstm(snt_embeds)
        h = snt_embeds[:, :, 0:self.embedding_dim] + snt_embeds[:, :, self.embedding_dim:2*self.embedding_dim]

        return h

class ea_posterior_mu(nn.Module):

    def __init__(self, embedding_dim):
        super(ea_posterior_mu, self).__init__()

        self.embedding_dim = embedding_dim
        
        self.affine_1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.relu = nn.ReLU()
        self.affine_2 = nn.Linear(self.embedding_dim, self.embedding_dim)
         
    def forward(self, h):
        a1 = self.relu(self.affine_1(h))
        posterior_mu = self.affine_2(a1)
        
        return posterior_mu
  
      
class ea_posterior_sigma(nn.Module):

    def __init__(self, embedding_dim):
        super(ea_posterior_sigma, self).__init__()

        self.embedding_dim = embedding_dim
    
        self.affine_1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.relu = nn.ReLU()
        self.affine_2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.softplus = nn.Softplus()
         
    def forward(self, h):
        a1 = self.relu(self.affine_1(h))
        posterior_sigma = self.softplus(self.affine_2(a1))
        
        return posterior_sigma

class ea_decoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(ea_decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.a1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.relu = nn.ReLU()
        self.a2 = nn.Linear(self.embedding_dim, self.vocab_size)
        self.softmax = nn.Softmax(dim=2)
 
    def forward(self, z):
        a1 = self.relu(self.a1(z))
        probs = self.softmax(self.a2(a1))
        return probs
    
def train_model(train_data, word_to_id1, id_to_word1, word_to_id2, id_to_word2, num_pairs, filename, load_model, epochs, batch_size, learning_rate, embedding_dim):
    vocab_size1 = len(word_to_id1.keys())
    vocab_size2 = len(word_to_id2.keys())    
    max_length_L1 = len(train_data[0][0])
    
    # Initialize the models.
    mod_encoder = ea_encoder(vocab_size1, embedding_dim)
    mod_posterior_mu = ea_posterior_mu(embedding_dim)
    mod_posterior_sigma = ea_posterior_sigma(embedding_dim)
    mod_decoder_L1 = ea_decoder(vocab_size1, embedding_dim)
    mod_decoder_L2 = ea_decoder(vocab_size2, embedding_dim)
    
    # Combine them into the modules.
    modules = nn.ModuleList()
    modules.append(mod_encoder)
    modules.append(mod_posterior_mu) 
    modules.append(mod_posterior_sigma) 
    modules.append(mod_decoder_L1)       
    modules.append(mod_decoder_L2)       
    
    optimizer = optim.Adam(modules.parameters(), lr = learning_rate)    
    norm_dist = torch.distributions.normal.Normal(0, 1)
    
    # Contains the losses and time per epoch.
    epoch_data = []
    start_epoch = 0
    if load_model and os.path.isfile(filename):
        print("Loading checkpoint")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        modules.load_state_dict(checkpoint['state_dict'])
        epoch_data = checkpoint['epoch_data']
        optimizer.load_state_dict(checkpoint['optimizer'])
        word_to_id1 = checkpoint['word_to_id1']
        word_to_id2 = checkpoint['word_to_id2']
        id_to_word1 = checkpoint['id_to_word1']
        id_to_word2 = checkpoint['id_to_word2']
        vocab_size1 = checkpoint['vocab_size1']
        vocab_size2 = checkpoint['vocab_size2']
        max_length_L1 = checkpoint['max_length_L1']
        num_pairs = checkpoint['num_pairs']
        embedding_dim = checkpoint['embedding_dim']

    print(modules)
    print("==========\nepochs: {}\nbatch_size: {}\nlearning_rate: {}\nvocab_size1: {}\nvocab_size2: {}\nembedding_dim: {}\nnum_pairs: {}\n==========\n\n".format(epochs, batch_size, learning_rate, vocab_size1, vocab_size2, embedding_dim, num_pairs))

    # For each epoch
    for epoch_i in range(start_epoch + 1, epochs + 1):    
        train_loss = 0
        print("\tEpoch: {}".format(epoch_i))
        
        data_generator = gen_data_id(train_data, word_to_id1, word_to_id2)
        start_time = time.time()      
        
        # For each batch.
        amount_batches = int((num_pairs + batch_size - 1) / batch_size) # +1 so the overfloating part is also used.
        for batch_i in range(amount_batches):
            modules.zero_grad()  
                    
            # It will also return the less than batchsize part at the end.
            data_subset = list(itertools.islice(data_generator, batch_size))
            
            # If the batch is exactly 1, we will get problems due to singleton 
            # dimensions. We did include batches smaller than the batchsize to 
            # use most data as possible, but the 1 batch size is problematic and 
            # not that important and also it almost never occurs. So, if it does
            # it will be ignored.
            if len(data_subset) == 1:
                print('Warning, batch of 1 element has been ignored')
                continue

            sentence_ids1, sentence_ids2, lengths_L1, lengths_L2 = zip(*data_subset)
            
            sentence_ids1 = torch.squeeze(Variable(torch.LongTensor([sentence_ids1])))
            sentence_ids2 = torch.squeeze(Variable(torch.LongTensor([sentence_ids2])))
            lengths_L1 = torch.squeeze(Variable(torch.FloatTensor([lengths_L1])))
            lengths_L2 = torch.squeeze(Variable(torch.FloatTensor([lengths_L2])))
    
            # Forward passes.
            h = mod_encoder(sentence_ids1)
            posterior_mu = mod_posterior_mu(h)
            posterior_sigma = mod_posterior_sigma(h)
            
            # Sample noise.
            cur_batch_size = posterior_sigma.shape[0]
            noise = norm_dist.sample((cur_batch_size, max_length_L1, embedding_dim))
            
            # Sample z and get the probabilities.
            sampled_z = posterior_mu + torch.mul(posterior_sigma, noise)
            probs_L1 = mod_decoder_L1(sampled_z)
            probs_L2 = mod_decoder_L2(sampled_z)

            # Compute loss.
            sentence_ids1 = torch.unsqueeze(sentence_ids1, 2)
            likelihood_L1 = torch.sum(torch.log(probs_L1.gather(2, sentence_ids1)), 1)
            
            # Instead of having L1sentence x L2vocsize, 
            # we want to create L1sentence x L2sentence combinations.
            sentence_ids2_unsqueezed = torch.unsqueeze(sentence_ids2, 1)
            sentence_ids2_rep = sentence_ids2_unsqueezed.repeat(1, max_length_L1, 1)
            probs_L1_L2 = probs_L2.gather(2, sentence_ids2_rep)
            sum_L1 = probs_L1_L2.sum(1)
            
            # batch x 1.
            lengths_L1_unsqueezed = torch.unsqueeze(lengths_L1, 1)
            normalized_L1 = torch.div(sum_L1, lengths_L1_unsqueezed)
            log_L1 = torch.log(normalized_L1)
            sum_L1 = sum(log_L1, 1)
       
            prior_mu = torch.zeros(posterior_mu.shape)
            prior_sigma = torch.ones(posterior_sigma.shape)
            kl = fn.compute_kl(posterior_mu, posterior_sigma, prior_mu, prior_sigma)
            loss = -(torch.sum(likelihood_L1) + torch.sum(sum_L1) - kl)

            # Backward pass.
            loss.backward()
    
            # Update weights.
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

        # Save the model.
        if os.path.isfile(filename):
            os.remove(filename)
        torch.save({
            'epoch': epoch_i,
            'state_dict': modules.state_dict(),
            'epoch_data': epoch_data,
            'optimizer': optimizer.state_dict(),
            'word_to_id1': word_to_id1,
            'word_to_id2': word_to_id2,
            'id_to_word1': id_to_word1,
            'id_to_word2': id_to_word2,
            'vocab_size1': vocab_size1,
            'vocab_size2': vocab_size2,
            'embedding_dim': embedding_dim,
            'max_length_L1': max_length_L1,
            'num_pairs': num_pairs,
        }, filename)
    
    return modules, epoch_data