from typing import Dict

import nltk

from lexical_analysis import vocabulary_size, unknown_token, start_sentence, end_sentence


def tokenize(text_for_tokenize: [str]):
    return [nltk.word_tokenize(sentence) for sentence in nltk.sent_tokenize(text_for_tokenize)]


def word_frequency(tokenized_words, with_frequency=True):
    word_dist = nltk.FreqDist(tokenized_words)  # TODO check what is in the word_freq
    print('%d unique words' % len(word_dist.items()))
    if with_frequency:
        return word_dist.most_common(vocabulary_size - 1)
    else:
        return [word_freq[0] for word_freq in word_dist.most_common(vocabulary_size - 1)]


def replace_with_index(all_tokenized_sentences, word_index_dict) -> None:
    for i, sentence in enumerate(all_tokenized_sentences):
        sentence_tokenized_list = [start_sentence] + [word_index_dict.get(word, unknown_token) for word in sentence]
        sentence_tokenized_list.append(end_sentence)
        all_tokenized_sentences[i] = sentence_tokenized_list


def words_indexer(common_tokenized_vocabulary, reverse=False) -> Dict[str, int]:
    if reverse:
        return {index: word for index, word in enumerate(common_tokenized_vocabulary)}
    else:
        return {word: index for index, word in enumerate(common_tokenized_vocabulary)}
