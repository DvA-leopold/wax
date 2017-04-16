from keras.layers import LSTM, Embedding, Dense
from keras.models import Sequential

from lexical_analyzer import vocabulary_size


def prepare_model(x_train, y_train, max_message_len: int):
    model = Sequential()
    model.add(Embedding(vocabulary_size + 1, output_dim=32, input_length=max_message_len))  # 5000
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(x_train, y_train, epochs=10, batch_size=64)
    return model, history
