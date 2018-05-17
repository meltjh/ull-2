from collections import defaultdict
import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import itertools
import torch.nn.functional as F
import sys
import os

torch.manual_seed(0)
random.seed(0)

# Generator that returns (target_word_hv, context_word_hv) tuples.
def gen_data_id(train_data, word_to_id):
    data_pairs = []
    for target_word, context_words in train_data.items():
        target_word_id = word_to_id[target_word]
        for c_word in context_words:
            context_word_id = word_to_id[c_word]
            dp = (target_word_id, context_word_id)
            data_pairs.append(dp)
            
    random.shuffle(data_pairs)
    
    for dp in data_pairs:
        yield dp


class skip_gram(nn.Module):

    def __init__(self, input_dim, embedding_dim, output_dim):
        super(skip_gram, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)
 
    def forward(self, inputs):
        embed = self.embedding(inputs)
        lin = self.linear(embed)
        outputs = self.log_softmax(lin)
        return outputs

def train_model(train_data, word_to_id, id_to_word, top_words_en, num_pairs, filename, load_model, epochs, batch_size, learning_rate, embedding_dim):
    vocab_size = len(word_to_id.keys())

    model = skip_gram(vocab_size, embedding_dim, vocab_size)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    loss_function = nn.NLLLoss()

    # contains the losses and time per epoch
    epoch_data = []
    start_epoch = 0
    if load_model and os.path.isfile(filename):
        print("Loading checkpoint")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        epoch_data = checkpoint['epoch_data']
        optimizer.load_state_dict(checkpoint['optimizer'])
        word_to_id = checkpoint['word_to_id']
        id_to_word = checkpoint['id_to_word']
        top_words_en = checkpoint['top_words_en']
        vocab_size = checkpoint['vocab_size']
        num_pairs = checkpoint['num_pairs']
        embedding_dim = checkpoint['embedding_dim']

    print(model)
    print("==========\nEpochs: {}\nBatch_size: {}\nLearning_rate: {}\nVocab_size: {}\nEmbedding_dim: {}\nnum_pairs: {}\n==========\n\n".format(epochs, batch_size, learning_rate, vocab_size, embedding_dim, num_pairs))
    
    for epoch_i in range(start_epoch + 1, epochs + 1):        
        train_loss = 0
        print("\tEpoch: {}".format(epoch_i))
        
        data_generator = gen_data_id(train_data, word_to_id)
        start_time = time.time()
    
        # For each batch
        amount_batches = int((num_pairs + batch_size - 1) / batch_size)
        for batch_i in range(amount_batches): 
            model.zero_grad()
            
            # It will also return the less than batchsize part at the end
            data_subset = list(itertools.islice(data_generator, batch_size))
            l_in_id, l_out_id = zip(*data_subset)
            
            # Forward pass
            target_id = Variable(torch.LongTensor(l_in_id))
            output = model(target_id)

            # Compute loss
            context_id = Variable(torch.LongTensor(l_out_id))
            loss = loss_function(output, context_id)
    
            # Backward pass
            loss.backward()
    
            # Update weights
            optimizer.step()
            
            train_loss = train_loss+loss
            
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
            'state_dict': model.state_dict(),
            'epoch_data': epoch_data,
            'optimizer': optimizer.state_dict(),
            'word_to_id': word_to_id,
            'id_to_word': id_to_word,
            'top_words_en': top_words_en,
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'num_pairs': num_pairs,
        }, filename)   
        
    return model, epoch_data