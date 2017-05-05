from typing import Union, List

from keras.callbacks import History, TensorBoard
from keras.layers import LSTM, Embedding, Dense, GRU, SimpleRNN, Conv1D, MaxPooling1D
from keras.models import Sequential

from lexical_analyzer import vocabulary_size


def prepare_lstm_model(train: List[List[int]], results: List[int], max_message_len: int) -> Union[Sequential, History]:
    model = Sequential()
    model.add(Embedding(vocabulary_size + 1, output_dim=64, input_length=max_message_len))
    model.add(Conv1D(filters=32, kernel_size=16, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(128, kernel_initializer='he_normal', return_sequences=True))
    model.add(LSTM(64, kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train, results, epochs=10, batch_size=64,
                        callbacks=[TensorBoard(log_dir='./logs/conv_256_64/lstm', write_graph=True)])
    return model, history


def prepare_gru_model(train: List[List[int]], results: List[int], max_message_len: int) -> Union[Sequential, History]:
    model = Sequential()
    model.add(Embedding(vocabulary_size + 1, output_dim=64, input_length=max_message_len))
    model.add(Conv1D(filters=32, kernel_size=16, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(GRU(256, kernel_initializer='he_normal', return_sequences=True))
    model.add(GRU(64, kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(train, results, epochs=10, batch_size=64,
                        callbacks=[TensorBoard(log_dir='./logs/conv_256_64/gru', write_graph=True)])
    return model, history


def prepare_rnn_model(train: List[List[int]], results: List[int], max_message_len: int) -> Union[Sequential, History]:
    model = Sequential()
    model.add(Embedding(vocabulary_size + 1, output_dim=64, input_length=max_message_len))
    model.add(Conv1D(filters=32, kernel_size=16, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(SimpleRNN(256, kernel_initializer='he_normal', return_sequences=True))
    model.add(SimpleRNN(64, kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(train, results, epochs=10, batch_size=64,
                        callbacks=[TensorBoard(log_dir='./logs/conv_256_64/rnn', write_graph=True)])
    return model, history
