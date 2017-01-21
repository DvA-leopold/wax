import numpy as np
import theano as theano
import theano.tensor as tensor


class GRUTheano:
    def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):
        print('start gru init')
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Initialize the network parameters
        E = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        U = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (6, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (6, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        b = np.zeros((6, hidden_dim))
        c = np.zeros(word_dim)
        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
        self.__theano_build__()
        print('finish gru init')

    def __theano_build__(self):
        E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b, self.c

        x = tensor.ivector('x')
        y = tensor.ivector('y')

        def forward_prop_step(x_t, s_t1_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # Word embedding layer
            x_e = E[:, x_t]

            z_t1 = tensor.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = tensor.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = tensor.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
            s_t1 = (tensor.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = tensor.nnet.softmax(V.dot(s_t1) + c)[0]

            return [o_t, s_t1]

        [o, s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          dict(initial=tensor.zeros(self.hidden_dim))])

        prediction = tensor.argmax(o, axis=1)
        o_error = tensor.sum(tensor.nnet.categorical_crossentropy(o, y))

        # Total cost (could add regularization here)
        cost = o_error

        # Gradients
        dE = tensor.grad(cost, E)
        dU = tensor.grad(cost, U)
        dW = tensor.grad(cost, W)
        db = tensor.grad(cost, b)
        dV = tensor.grad(cost, V)
        dc = tensor.grad(cost, c)

        # Assign functions
        self.predict = theano.function([x], o)
        self.predict_class = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], [dE, dU, dW, db, dV, dc])

        # SGD parameters
        learning_rate = tensor.scalar('learning_rate')
        decay = tensor.scalar('decay')

        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2

        self.sgd_step = theano.function(
            inputs=[x, y, learning_rate, theano.Param(decay, default=0.9)],
            outputs=[],
            updates=[(E, E - learning_rate * dE / tensor.sqrt(mE + 1e-6)),
                     (U, U - learning_rate * dU / tensor.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / tensor.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / tensor.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / tensor.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / tensor.sqrt(mc + 1e-6)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)]
        )

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x, y) for x, y in zip(X, Y)])

    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X, Y) / float(num_words)
