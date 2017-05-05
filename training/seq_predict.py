import numpy as np

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense


def baseline_model():
    model = Sequential()
    model.add(Dense(1, input_dim=1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model


def baseline_model_5():
    model = Sequential()
    model.add(Dense(5, input_dim=1, activation='linear'))
    model.add(Dense(1, input_dim=5, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model


def baseline_model_ReLU():
    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu'))
    model.add(Dense(1, input_dim=10, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model


def baseline_model_msgd_ReLU():
    model = Sequential()
    model.add(Dense(10000, input_dim=1, activation='relu'))
    model.add(Dense(1, input_dim=10000, activation='linear'))

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model


def baseline_model_msgd_tanh():
    model = Sequential()
    model.add(Dense(100, input_dim=1, activation='tanh'))
    model.add(Dense(1, input_dim=100, activation='linear'))
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model


def baseline_model_msgd_tanh_init_weight():
    model = Sequential()
    # init='glorot', init='he_normal'
    model.add(Dense(20, input_dim=1, activation='tanh', kernel_initializer='he_normal'))
    model.add(Dense(1, input_dim=20, activation='linear', kernel_initializer='he_normal'))
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model


def baseline_model_msgd_deep_tanh():
    model = Sequential()
    model.add(Dense(20, input_dim=1, activation='tanh', kernel_initializer='he_normal'))
    model.add(Dense(20, input_dim=20, activation='tanh', kernel_initializer='he_normal'))
    model.add(Dense(1, input_dim=20, activation='linear', kernel_initializer='he_normal'))
    model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
    return model


def f(x):
    return x * np.sin(x * 2 * np.pi) if x < 0 else -x * np.sin(x * np.pi) + np.exp(x / 2) - np.exp(0)


if __name__ == '__main__':
    x = np.linspace(-3, 3, 1000).reshape(-1, 1)
    print(x)

    f = np.vectorize(f)
    y = f(x)
    # model = baseline_model_msgd_deep_tanh()
    # model.fit(x, y, epochs=1000, verbose=0)

    # plt.scatter(x, y, color='black', antialiased=True)
    # plt.plot(x, model.predict(x), color='magenta', linewidth=2, antialiased=True)
    # plt.show()

    # for layer in model.layers:
    #     weights = layer.get_weights()
    #     print(weights)
