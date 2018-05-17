import numpy as np
from collections import defaultdict
import sys
import functions as fn
from functions import get_word_id_mappings

def get_pairs(dataset, context_window):
    context, num_pairs = get_context(dataset, context_window)
    word_to_id, id_to_word = fn.get_word_id_mappings(dataset, False)
    return context, num_pairs, word_to_id, id_to_word
    
def get_context(dataset, context_window):
    len_dataset = len(dataset)
    context = defaultdict(lambda: [])
    progress = 0
    num_pairs = 0
    for sentence in dataset:
        progress += 1
        for i in range(len(sentence)):
            target_word = sentence[i]
            if target_word == 'UNK':
                continue

            for j in range(1, context_window+1):
                if i - j >= 0:
                    context_word_before = sentence[i - j]
                    context[target_word].append(context_word_before)
                    num_pairs += 1
                    
                if i + j < len(sentence):
                    context_word_after = sentence[i + j]
                    context[target_word].append(context_word_after)
                    num_pairs += 1
                    
        if (progress%1000) == 0:
            sys.stdout.write('\rObtaining context words: {}/{} ({}%)'.format(progress, len_dataset, round(progress/len_dataset*100, 2)))
            sys.stdout.flush()
            
    sys.stdout.write('\rFinished obtaining context words: {}/{}'.format(progress, len_dataset))
    return context, num_pairs