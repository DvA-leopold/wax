import itertools

from theano.gradient import np

from lexical_analysis import lingspam_public_bare, vocabulary_size
from lexical_analysis.message_parser import MessageParser
from lexical_analysis.tokenizer import tokenize, word_frequency, replace_with_index, words_indexer
from rnn.gradient_theano import RNNTheano
from rnn.gradient_numpy import RNNNumpy
from rnn.sgd_train import train_with_sgd

if __name__ == '__main__':
    message_parser = MessageParser()
    message_parser.init_sub_files(lingspam_public_bare)
    all_messages_in_string = message_parser.message_bodies_as_string()

    tokenized_sentences_by_words = tokenize(all_messages_in_string)
    common_vocabulary = word_frequency(itertools.chain(*tokenized_sentences_by_words), False)
    word_index_map = words_indexer(common_vocabulary)
    replace_with_index(tokenized_sentences_by_words, word_index_map)

    X_train = np.asarray([sent[:-1] for sent in tokenized_sentences_by_words])
    Y_train = np.asarray([sent[1:] for sent in tokenized_sentences_by_words])

    np.random.seed(10)
    # Train on a small subset of the data to see what happens
    model = RNNTheano(vocabulary_size)
    losses = train_with_sgd(model, X_train[:1000], Y_train[:1000], nepoch=100, evaluate_loss_after=1)

    # model = RNNNumpy(vocabulary_size)
    # print('Expected random loss: %f' % np.log(vocabulary_size))
    # print('Actual loss: %f' % model.calculate_loss(X_train[:500], Y_train[:500]))

    # for sentence in tokenized_sentences_by_words:
    #     print(sentence)

