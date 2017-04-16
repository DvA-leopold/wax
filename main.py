from math import sqrt

from keras.preprocessing import sequence

from lexical_analyzer import lingspam_public_bare
from lexical_analyzer.message_parser import MessageParser
from spam_analyzer import prepare_model

if __name__ == '__main__':
    message_parser = MessageParser(lingspam_public_bare, spam_msg_limit=200, clear_msg_limit=200)
    samples, results = message_parser.index_by_message()
    math_exp, dispersion = message_parser.calc_math(samples)
    pad_seq_size = int(math_exp + sqrt(dispersion) * 2)
    samples = sequence.pad_sequences(samples, pad_seq_size, padding='post', truncating='post')

    print('math expectation: ', math_exp)
    print('math deviation: ', sqrt(dispersion))
    print('max msg len: ', message_parser.max_message_len)

    prepared_model, history = prepare_model(samples, results, pad_seq_size)
    scores = prepared_model.evaluate(samples, results, verbose=0)
    print("Accuracy: {}%".format(scores[1] * 100))
