from collections import defaultdict

# Reads the data from st_test.preprocessed and returns dict
# sentence_id:{tw:tw, cw:[cw,...], tw_raw: tw.postag}        
def read_lst_data_sg(context_window, top_words):
    filename = 'lst/lst_test.preprocessed'
    data = dict()
    
    with open(filename) as f:
        for line in f:
            # Split line by tabs.
            content = line.split('\t')
            # Obtain main parts.
            tw_raw = str(content[0])
            id_sentence = int(content[1])
            position = int(content[2])
            sentence = str(content[3])
            
            # Split a few inner parts.
            tw = str(tw_raw.split('.')[0]) # remove the .pos part.
            
            sentence_words = sentence.split()
            
            # Get the context words.
            sentence_len = len(sentence_words)
            contextwords = []
            
            # Check the left context window.
            pos_left_fulfilled = 0
            pos_left = position
            # While not all of the context words are found.
            while pos_left_fulfilled < context_window:
                # Get position to the left.
                pos_left = pos_left - 1
                # Break if invalid pos_front.
                if pos_left < 0:
                    break
                # Obtain the word and continue if the word should be ignored.
                word = sentence_words[pos_left]
                if word not in top_words:
                    word = 'UNK'
                
                # Add the word
                contextwords.append(word)
                pos_left_fulfilled = pos_left_fulfilled + 1
            
            # Check the right context window.
            pos_right_fulfilled = 0
            pos_right = position
            # While not all of the context words are found.
            while pos_right_fulfilled < context_window:
                # Get position to the right.
                pos_right = pos_right + 1
                # Break if invalid pos_front.
                if pos_right >= sentence_len:
                    break
                # Obtain the word and continue if the word should be ignored.
                word = sentence_words[pos_right]
                if word not in top_words:
                    word = 'UNK'
                
                # Add the word
                contextwords.append(word)
                pos_right_fulfilled = pos_right_fulfilled + 1
                
            # Create the dict and put in in the data dict.      
            data[id_sentence] = {'target_word': tw, 'context_words': contextwords, 'target_word_raw': tw_raw}
    return data

# Reads the data from st_test.preprocessed and returns dict
# sentence_id:{tw:tw, cw:[cw,...], tw_raw: tw.postag}        
def read_lst_data_ea(top_words, max_length):
    filename = 'lst/lst_test.preprocessed'
    data = dict()
    
    with open(filename) as f:
        for line in f:
            # Split line by tabs.
            content = line.split('\t')
            # Obtain main parts.
            tw_raw = str(content[0])
            id_sentence = int(content[1])
            position = int(content[2])
            sentence = str(content[3])
            
            # Split a few inner parts.
            tw = str(tw_raw.split('.')[0]) # remove the .pos part.
            
            sentence_words = sentence.split()
            preprocessed_sentence = []
            for word in sentence_words:
                if word not in top_words:
                    word = 'UNK'
                preprocessed_sentence.append(word)
            
            # Add padding.
            temp_length = len(preprocessed_sentence)
            diff = max_length - temp_length
            if diff > 0:
                preprocessed_sentence.extend(['PAD'] * diff)
            
            # Create the dict and put in in the data dict          
            data[id_sentence] = {'target_word': tw, 'sentence': preprocessed_sentence, 
                'target_word_raw': tw_raw, 'position': position}
    return data

def get_candidates():
    filename = 'lst/lst.gold.candidates'
    candidates_dict = defaultdict(lambda: [])
    all_words = []
    with open(filename) as f:
        for line in f:
            target_raw, candidates_string = line.split('::')
            
            # Remove \n and split by ;
            candidates = candidates_string.rstrip().split(';')
            
            # Extend the existing list with the new one.
            candidates_dict[target_raw] += candidates
        
    # Remove duplicates per target_raw by casting all candidates in a set.
    for target_raw, candidates in candidates_dict.items():
        set_candidates = set(candidates)
        candidates_dict[target_raw] = set_candidates
        all_words += set_candidates

    return candidates_dict, list(set(all_words))

# Write the scores to a file.
def write_scores(filename, results):
    with open(filename, "w+") as f:
        for key, scores in results.items():
            target_word = key[0]
            sentence_id = key[1]
            
            string = "RANKED\t{} {}".format(target_word, sentence_id)
            for (candidate_word, score) in scores:
                string += "\t{} {}".format(candidate_word, score)
            string += "\n"
            f.write(string)