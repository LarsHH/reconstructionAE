import theano
from theano import tensor as T
import numpy as np
from nnet_tools import rectify, floatX, init_weights, init_biases, sgd, save_model
from load import mnist
import scipy
from scipy.misc import imsave


def model(input_data, w_h, w_o, b_h, b_y):
    h1 = T.nnet.sigmoid(T.dot(input_data, w_h) + b_h)
    output = T.nnet.sigmoid(T.dot(h1, w_o) + b_y)
    return output

# Define parameters
learning_rate = 0.1
n_hidden_units = 100
n_visible_units = 784
blank_proportion = .1
alpha = .5
batch_size = 100
epochs = 10

X = T.fmatrix('X')

# Initialize Weights
weights_layer_1 = init_weights((n_visible_units, n_hidden_units), name='weights_layer_1')
bias_layer_1 = init_biases(n_hidden_units, name='bias_layer_1')
weights_layer_y = init_weights((n_hidden_units, n_visible_units), name='weights_layer_y')
bias_layer_y = init_biases(n_visible_units, name='bias_layer_y')

# Build Model
X_hat = model(X, weights_layer_1, weights_layer_y, bias_layer_1, bias_layer_y)

# Loss
L = - T.sum(X * T.log(X_hat) + (1 - X) * T.log(1 - X_hat), axis=1)
loss = T.mean(L)

# Parameter Updating
params = [weights_layer_y, bias_layer_y, weights_layer_1, bias_layer_1]
updates = sgd(loss, params, lr=learning_rate)

# Compiling
train = theano.function(inputs=[X], outputs=loss, updates=updates, allow_input_downcast=True, on_unused_input='warn')
predict = theano.function(inputs=[X], outputs=X_hat, allow_input_downcast=True)

# Load the data
trX, teX, trY, teY = mnist(scale_data=False)
n_batches_train = trX.shape[0] / batch_size


# Training
for i in range(epochs):
    cost_per_batch = np.zeros(n_batches_train)
    pred_cost_per_batch = np.zeros(n_batches_train)
    cost = []
    for start, end in zip(range(0, n_batches_train * batch_size, batch_size),
                          range(batch_size, n_batches_train * batch_size, batch_size)):
        cost.append(train(trX[start:end]))

    print "Epoch number {0}".format(i)
    print 'Mean Cost per Batch %s' % str(i), np.mean(cost)

# Prediction
n_imgs = 100
predX = predict(teX[0:n_imgs])
pred_imgs = predX.reshape((-1, 28, 28))
true_imgs = teX[0:n_imgs].reshape((-1, 28, 28))

img_array = np.zeros((28 * n_imgs, 28 * 2))
for i in range(n_imgs):
    img_array[i*28:(i+1)*28, 0:28] = true_imgs[i, :, :]
    img_array[i*28:(i+1)*28, 28:56] = pred_imgs[i, :, :]

scipy.misc.imsave('./predictions/regae.gif', img_array)

# Save model
save_model(path="./regae.pkl", values=[weights_layer_1.eval(), bias_layer_1.eval(),
                                                      bias_layer_y.eval()])

