import functions as fn
    
def get_context(dataset1, dataset2, max_max_length):
    context = list(zip(dataset1, dataset2))
    
    max_size_L1 = 0
    max_size_L2 = 0
    for sentence_L1, sentence_L2 in context:
        sentence_L1 = sentence_L1[:max_max_length]
        sentence_L2 = sentence_L2[:max_max_length]
        
        size_L1 = len(sentence_L1)
        size_L2 = len(sentence_L2)
        
        # Keep track of the maximum sentence length for the padding.
        if size_L1 > max_size_L1:
            max_size_L1 = size_L1
        if size_L2 > max_size_L2:
            max_size_L2 = size_L2
            
    padded_sentences_L1, lengths_L1 = pad_sentences(dataset1, max_size_L1, max_max_length)
    padded_sentences_L2, lengths_L2 = pad_sentences(dataset2, max_size_L2, max_max_length)
    context_padded = list(zip(padded_sentences_L1, padded_sentences_L2, lengths_L1, lengths_L2))

    # Get the word2id and id2word mappings.
    word_to_id1, id_to_word1 = fn.get_word_id_mappings(dataset1, True)
    word_to_id2, id_to_word2 = fn.get_word_id_mappings(dataset2, True)
    return context_padded, len(context_padded), word_to_id1, id_to_word1, word_to_id2, id_to_word2

# Extend the sentences to the maximum sentence length by using a padding word.
def pad_sentences(sentences, max_length, max_max_length):
    new_sentences = []
    sentence_lengths = []
    for sentence in sentences:
        n_sentence = sentence[:max_max_length]
        length = len(sentence)
        shortage = max_length - length
        if shortage > 0:
            padding = ['PAD'] * shortage
            n_sentence.extend(padding)
        new_sentences.append(n_sentence)
        sentence_lengths.append(length)
    return new_sentences, sentence_lengths