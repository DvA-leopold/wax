import itertools
import nltk


vocabulary_size = 10000
unknown_token = 'UNKNOWN_TOKEN'


def tokenize(text_for_tokenize):
    sentences = nltk.sent_tokenize(text_for_tokenize)
    tokenized_sentences_generator = (nltk.word_tokenize(sentence) for sentence in sentences)
    tokenized_sentences = list(tokenized_sentences_generator)
    return tokenized_sentences, sentences


def word_frequency(tokenized_sentences):
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print('%d unique words' % len(word_freq.items()))
    vocabulary = word_freq.most_common(vocabulary_size - 1)
    print('vocabulary:', vocabulary)
    return word_freq

