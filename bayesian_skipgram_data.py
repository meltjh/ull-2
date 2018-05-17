from collections import defaultdict
import sys
import functions as fn
    
def get_context(dataset, context_window):
    len_dataset = len(dataset)
    context = defaultdict(lambda: [])
    progress = 0
    num_context = 0
    for sentence in dataset:
        progress += 1
        for i in range(len(sentence)):
            sentence_context = []
            target_word = sentence[i]
            
            # Get the context words before and after the target model.
            # in range of the context window.
            for j in range(1, context_window+1):
                if i - j >= 0:
                    context_word_before = sentence[i - j]
                    sentence_context.append(context_word_before)
                else:
                    sentence_context.append('PAD')
                    
                if i + j < len(sentence):
                    context_word_after = sentence[i + j]
                    sentence_context.append(context_word_after)
                else:
                    sentence_context.append('PAD')

            if len(sentence_context) > 0:
                context[target_word].append(sentence_context)
                num_context += 1
                        
        if (progress%1000) == 0:
            sys.stdout.write('\rObtaining context words: {}/{} ({}%)'.format(progress, len_dataset, round(progress/len_dataset*100, 2)))
            sys.stdout.flush()

    sys.stdout.write('\rFinished obtaining context words: {}/{}'.format(progress, len_dataset))
    # Get the word2id and id2word mappings.
    word_to_id, id_to_word = fn.get_word_id_mappings(dataset, True)
    
    return context, num_context, word_to_id, id_to_word