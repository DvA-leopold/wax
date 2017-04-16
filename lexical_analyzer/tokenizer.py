import nltk
from typing import Dict

from lexical_analyzer import vocabulary_size, UNKNOWN_TOKEN


# split text into sentences and sentences into words


def tokenize(text_for_tokenize: [str]) -> [str]:
    return [nltk.word_tokenize(sentence) for sentence in nltk.sent_tokenize(text_for_tokenize)]


def word_frequency(tokenized_words: [str], with_frequency=True):
    word_dist = nltk.FreqDist(tokenized_words)  # TODO check what is in the word_freq
    print('%d unique words' % len(word_dist.elements()))
    if with_frequency:
        return word_dist.most_common(vocabulary_size - 1)
    else:
        return [word_freq[0] for word_freq in word_dist.most_common(vocabulary_size - 1)]


def index_text(text_to_index, word_index_dict: Dict[str, int]) -> None:
    for word in text_to_index:
        word_index_dict.get(word, UNKNOWN_TOKEN)

'''
def replace_with_index(all_tokenized_sentences, word_index_dict) -> None:
    for i, sentence in enumerate(all_tokenized_sentences):
        sentence_tokenized_list = [START_SENTENCE] + [word_index_dict.get(word, UNKNOWN_TOKEN) for word in sentence]
        sentence_tokenized_list.append(END_SENTENCE)
        all_tokenized_sentences[i] = sentence_tokenized_list
'''