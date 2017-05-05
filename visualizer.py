from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, GRU, SimpleRNN
from keras.models import Sequential
from keras.utils import plot_model

from lexical_analyzer import vocabulary_size


def prepare_lstm_model():
    model = Sequential()
    model.add(Embedding(vocabulary_size + 1, output_dim=64))
    model.add(Conv1D(filters=32, kernel_size=16, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(128, kernel_initializer='he_normal', return_sequences=True))
    model.add(LSTM(128, kernel_initializer='he_normal', return_sequences=True))
    model.add(LSTM(64, kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, 'pics/conv128-128-64/lstm.png', show_shapes=True, show_layer_names=True)


def prepare_gru_model():
    model = Sequential()
    model.add(Embedding(vocabulary_size + 1, output_dim=64))
    model.add(Conv1D(filters=32, kernel_size=16, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(GRU(128, kernel_initializer='he_normal', return_sequences=True))
    model.add(GRU(128, kernel_initializer='he_normal', return_sequences=True))
    model.add(GRU(64, kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, 'pics/conv128-128-64/gru.png', show_shapes=True, show_layer_names=True)


def prepare_rnn_model():
    model = Sequential()
    model.add(Embedding(vocabulary_size + 1, output_dim=64))
    model.add(Conv1D(filters=32, kernel_size=16, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(SimpleRNN(128, kernel_initializer='he_normal', return_sequences=True))
    model.add(SimpleRNN(128, kernel_initializer='he_normal', return_sequences=True))
    model.add(SimpleRNN(64, kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, 'pics/conv128-128-64/rnn.png', show_shapes=True, show_layer_names=True)


if __name__ == '__main__':
    prepare_lstm_model()
    prepare_gru_model()
    prepare_rnn_model()
