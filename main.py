from math import sqrt

from keras.preprocessing import sequence

from lexical_analyzer import lingspam_public_bare, lingspam_public_lemm
from lexical_analyzer.message_parser import MessageParser
from spam_analyzer import *
from utils.math import math_expectation
from utils.math import dispersion

if __name__ == '__main__':
    mp_train = MessageParser(lingspam_public_bare, spam_msg_limit=1000, clear_msg_limit=1000)
    train_samples, train_result = mp_train.index_by_message()

    mp_test = MessageParser(lingspam_public_lemm, spam_msg_limit=500, clear_msg_limit=500)
    test_samples, test_results = mp_test.index_by_message()

    math_exp_train, math_exp_test = math_expectation(train_samples), math_expectation(test_samples)
    disp_train = dispersion(train_samples, math_exp_train)
    disp_test = dispersion(test_samples, math_exp_test)

    pad_train_seq = int(math_exp_train + sqrt(disp_train) * 2)
    pad_test_seq = int(math_exp_test + sqrt(disp_test) * 2)
    max_pad = max(pad_test_seq, pad_train_seq)

    test_samples = sequence.pad_sequences(test_samples, max_pad, padding='post', truncating='post')
    train_samples = sequence.pad_sequences(train_samples, max_pad, padding='post', truncating='post')

    print('math expectation train-{}, res-{}: '.format(math_exp_train, math_exp_test))
    print('math deviation train-{}, res-{}: '.format(sqrt(disp_train), sqrt(disp_test)))
    print('max msg len test-{}, train-{}: '.format(mp_test.max_message_len, mp_train.max_message_len))

    # print('-------------------------------------------------------------------------------')
    # prepared_model, history = prepare_rnn_model(train_samples, train_result, max_pad)
    # score, accuracy = prepared_model.evaluate(test_samples, test_results, verbose=0)
    #
    # print('metrics: ', prepared_model.metrics_names)
    # print("Score: {}%, Accuracy: {}%".format(round(score * 100, 2), round(accuracy * 100, 2)))

    print('-------------------------------------------------------------------------------')
    prepared_model, history = prepare_lstm_model(train_samples, train_result, max_pad)
    score, accuracy = prepared_model.evaluate(test_samples, test_results, verbose=0)

    print('metrics: ', prepared_model.metrics_names)
    print("Score: {}%, Accuracy: {}%".format(round(score * 100, 2), round(accuracy * 100, 2)))

    # print('-------------------------------------------------------------------------------')
    # prepared_model, history = prepare_gru_model(train_samples, train_result, max_pad)
    # score, accuracy = prepared_model.evaluate(test_samples, test_results, verbose=0)
    #
    # print('metrics: ', prepared_model.metrics_names)
    # print("Score: {}%, Accuracy: {}%".format(round(score * 100, 2), round(accuracy * 100, 2)))