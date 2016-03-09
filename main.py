import itertools

from lexical_analysis import lingspam_public_bare
from lexical_analysis.message_parser import MessageParser
from lexical_analysis.tokenizer import tokenize, word_frequency, replace_with_index, words_indexer

if __name__ == '__main__':
    message_parser = MessageParser()
    message_parser.init_sub_files(lingspam_public_bare)
    all_messages_in_string = message_parser.message_bodies_as_string()

    tokenized_sentences_by_words = tokenize(all_messages_in_string)
    common_vocabulary = word_frequency(itertools.chain(*tokenized_sentences_by_words), False)
    word_index_map = words_indexer(common_vocabulary)
    replace_with_index(tokenized_sentences_by_words, word_index_map)
    # for sentence in tokenized_sentences_by_words:
    #     print(sentence)

