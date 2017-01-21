import operator

import theano
import numpy as np
import theano.tensor as tensor


class RNNTheano:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        print('init rnn')
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.theano = {}
        self.__theano_build__()
        print('init rnn finished')

    def __theano_build__(self):
        U, V, W = self.U, self.V, self.W
        x = tensor.ivector('x')
        y = tensor.ivector('y')

        def forward_prop_step_gru(x_t, s_t1_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # Get the word vector
            x_e = E[:, x_t]

            # GRU Layer
            z_t1 = tensor.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = tensor.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = tensor.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
            s_t1 = (tensor.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = tensor.nnet.softmax(V.dot(s_t1) + c)[0]

            return [o_t, s_t1]

        def forward_prop_step_rnn(x_t, s_t_prev, U, V, W):
            s_t = tensor.tanh(U[:, x_t] + W.dot(s_t_prev))
            o_t = tensor.nnet.softmax(V.dot(s_t))
            return [o_t[0], s_t]

        [o, s], updates = theano.scan(
            forward_prop_step_gru,
            sequences=x,
            outputs_info=[None, dict(initial=tensor.zeros(self.hidden_dim))],
            non_sequences=[U, V, W],
            truncate_gradient=self.bptt_truncate,
            strict=True)

        prediction = tensor.argmax(o, axis=1)
        o_error = tensor.sum(theano.tensor.nnet.categorical_crossentropy(o, y))

        dU = tensor.grad(o_error, U)
        dV = tensor.grad(o_error, V)
        dW = tensor.grad(o_error, W)

        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU, dV, dW])

        learning_rate = tensor.scalar('learning_rate')
        self.sgd_step = theano.function([x, y, learning_rate], [],
                                        updates=[(self.U, self.U - learning_rate * dU),
                                                 (self.V, self.V - learning_rate * dV),
                                                 (self.W, self.W - learning_rate * dW)])

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x, y) for x, y in zip(X, Y)])

    def calculate_loss(self, X, Y):
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X, Y) / float(num_words)

    def gradient_check_theano(self, x, y, h=0.001, error_threshold=0.01):
        self.bptt_truncate = 1000
        bptt_gradients = self.bptt(x, y)
        model_parameters = ['U', 'V', 'W']
        for pidx, pname in enumerate(model_parameters):
            parameter_T = operator.attrgetter(pname)(self)
            parameter = parameter_T.get_value()
            print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                original_value = parameter[ix]
                parameter[ix] = original_value + h
                parameter_T.set_value(parameter)
                gradplus = self.calculate_total_loss([x], [y])
                parameter[ix] = original_value - h
                parameter_T.set_value(parameter)
                gradminus = self.calculate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus) / (2 * h)
                parameter[ix] = original_value
                parameter_T.set_value(parameter)
                backprop_gradient = bptt_gradients[pidx][ix]
                relative_error = np.abs(backprop_gradient - estimated_gradient) / (
                np.abs(backprop_gradient) + np.abs(estimated_gradient))
                if relative_error > error_threshold:
                    print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                    print("+h Loss: %f" % gradplus)
                    print("-h Loss: %f" % gradminus)
                    print("Estimated_gradient: %f" % estimated_gradient)
                    print("Backpropagation gradient: %f" % backprop_gradient)
                    print("Relative Error: %f" % relative_error)
                    return
                it.iternext()
            print("Gradient check for parameter %s passed." % (pname))
