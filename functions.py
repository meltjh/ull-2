from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import torch
from lst_data import get_candidates

def get_stopwords(language):
    stop_words = stopwords.words(language)
    stop_words = stop_words + [',', '.', ';', ':', '?', '"', "'", '-', '!', ')', '(']
    stop_words = set(stop_words)
    return stop_words

def read_data(temp_filename, min_occurence):
    stopwords_english = get_stopwords('english')
    stopwords_french = get_stopwords('french')

    en, cnt_en_words_original, cnt_en_words_filtered = read_data_single_lang(temp_filename + '.en', min_occurence, stopwords_english)
    fr, cnt_fr_words_original, cnt_fr_words_filtered = read_data_single_lang(temp_filename + '.fr', min_occurence, stopwords_french)

    # Plot the word frequencies.
#    plot_word_frequencies([cnt_en_words_original, cnt_fr_words_original], ['English', 'French', 'Minimum amount of occurences'],'word_frequencies_original.png', min_occurence)
#    plot_word_frequencies([cnt_en_words_filtered, cnt_fr_words_filtered], ['English', 'French'],'word_frequencies_filtered.png')
 
    top_words_en = list(cnt_en_words_filtered.keys())
    top_words_fr = list(cnt_fr_words_filtered.keys())
    
    return en, fr, top_words_en, top_words_fr

def read_data_single_lang(filename, min_occurence, stop_words):
    sentences = [] 
    all_words = []

    # Obtain all the sentences and all words.
    with open(filename) as f:
        for line in f:
            sentence_words = line.split()
            sentence_words = [word.lower() for word in sentence_words]
            sentences.append(sentence_words)
            all_words += sentence_words
                
    # Count the words before removing words.
    cnt_words_original = Counter(all_words)
    
    words_allowed = dict()

    # Removing least occuring words.
    for word, occur in cnt_words_original.items():
        if (occur >= min_occurence) and (word not in stop_words):
            words_allowed[word] = True
        
    # Also add the words that are used to evaluate.
    _, all_candidates = get_candidates()
    for word in all_candidates:
         words_allowed[word] = True
    
    # Replace all words that are not in the allowed words by 'UNK' in the sentences.
    filtered_sentences = []
    for sentence in sentences:
        filtered_sentence = []
        for word in sentence:
            if word in words_allowed:
                filtered_sentence.append(word)
            else:
                filtered_sentence.append("UNK")
        filtered_sentences.append(filtered_sentence)
    
    # Plot the filtered words occurences, excluding UNK.
    cnt_words_filtered = Counter([word for sentence in filtered_sentences for word in sentence])
    del cnt_words_filtered['UNK']
    
    return filtered_sentences, cnt_words_original, cnt_words_filtered

# Plot the word frequencies.
def plot_word_frequencies(cnt_words_languages, languages, filename, threshold=None):
    for cnt_words in cnt_words_languages:
        cnt_words = cnt_words.most_common()
        _, values = zip(*cnt_words)
        plt.plot(values)
        
    if threshold != None:
        plt.axhline(y=threshold, color='g', linestyle='--')
    
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Occurences')
    plt.legend(languages)
    plt.tight_layout()
    plt.savefig(filename, dpi = 200)
    plt.show()

# Returns the mapping of the word to id and id to word.
def get_word_id_mappings(sentences, padding):
    word_to_id = dict()
    id_to_word = dict()
    unique_words = list(set([word for sentence in sentences for word in sentence]))
    num_unique_words = len(unique_words)

    # Add padding when using BSG and EA.
    diff = 0
    if padding:
        word_to_id['PAD'] = 0
        id_to_word[0] = 'PAD'
        diff = 1

    for i in range(diff, num_unique_words + diff):
        cur_word = unique_words[i - diff]
        word_to_id[cur_word] = i
        id_to_word[i] = cur_word
    return word_to_id, id_to_word

# Compute the KL divergence.
def compute_kl(posterior_mu, posterior_sigma, prior_mu, prior_sigma):    
    a = torch.log(torch.div(prior_sigma, posterior_sigma))
    b = posterior_sigma**2 + (posterior_mu - prior_mu)**2
    c = 2*(prior_sigma**2)
    d = 0.5
    kl = torch.sum(a+torch.div(b, c)-d)
    return kl