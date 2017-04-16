from datetime import datetime


# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, Y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        if epoch % evaluate_loss_after == 0:
            loss = model.calculate_loss(X_train, Y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate *= 0.5
                print("Setting learning rate to %f" % learning_rate)
                # save_model_parameters_theano("./data/training-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        for i in range(len(Y_train)):
            # for every sentence make sgd step
            model.sgd_step(X_train[i], Y_train[i], learning_rate)
            num_examples_seen += 1
        # print(model.theano)
