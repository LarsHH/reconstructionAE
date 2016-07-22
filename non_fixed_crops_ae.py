import theano
from theano import tensor as T
import numpy as np
from nnet_tools import rectify, floatX, init_weights, init_biases, sgd, save_model, RMSprop
from load import mnist
import scipy
from scipy.misc import imsave

rs = np.random.RandomState(1234)
rng = theano.tensor.shared_randomstreams.RandomStreams(rs.randint(999999))

def model(input_data, w_h, w_h2, w_o, b_h, b_h2, b_y):
    h1 = T.nnet.sigmoid(T.dot(input_data, w_h) + b_h)
    h2 = T.nnet.sigmoid(T.dot(h1, w_h2) + b_h2)
    output = T.nnet.sigmoid(T.dot(h2, w_o) + b_y)
    return output


def random_crop(x_i):
    idx0 = rng.random_integers(low=0, high=crop_size, ndim=1)
    idx1 = rng.random_integers(low=784-crop_size, high=784, ndim=1)
    return T.set_subtensor(T.zeros_like(x_i)[idx0:idx1], T.alloc(1., idx1-idx0))


def MASK_blanking(x_i):
    # Find indicies of first and last non-zero value in x_i
    idxs = T.nonzero(x_i)[0][[1, -1]]
    # Diff = no of non zero values
    no_values = idxs[1] - idxs[0]
    # Move index inside by proportion of no of values
    idxs0 = T.cast(T.floor(idxs[0] + no_values * blank_proportion), 'int32')
    idxs1 = T.cast(T.floor(idxs[1] - no_values * blank_proportion), 'int32')
    # Return a vector that has a tighter mask than x_i
    return T.set_subtensor(T.zeros_like(x_i)[idxs0:idxs1], T.alloc(1., idxs1-idxs0))


# Define parameters
learning_rate = 0.001
n_hidden_units = 500
n_visible_units = 784
crop_size = 200
blank_proportion = 0.15
alpha = 0.5
batch_size = 100
epochs = 30

X = T.fmatrix('X')

# Make random crop from X
crop_mask, updates = theano.scan(fn=random_crop, sequences=[X])
crop_X = X * crop_mask

# Creating the masks
reconstruction_dims, updates = theano.scan(fn=MASK_blanking, sequences=[crop_X])
prediction_dims = crop_mask - reconstruction_dims

# Initialize Weights
weights_layer_1 = init_weights((n_visible_units, n_hidden_units), name='weights_layer_1')
bias_layer_1 = init_biases(n_hidden_units, name='bias_layer_1')
weights_layer_2 = init_weights((n_hidden_units, n_hidden_units), name='weights_layer_2')
bias_layer_2 = init_biases(n_hidden_units, name='bias_layer_2')
weights_layer_y = init_weights((n_hidden_units, n_visible_units), name='weights_layer_y')
bias_layer_y = init_biases(n_visible_units, name='bias_layer_y')

# Build Model
model_input = crop_X * reconstruction_dims
X_hat = model(model_input, weights_layer_1, weights_layer_2, weights_layer_y, bias_layer_1, bias_layer_2, bias_layer_y)

# Loss
# reconstruction_L = -T.dot(reconstruction_dims.T, crop_X * T.log(X_hat) + (1 - crop_X) * T.log(1 - X_hat))/T.sum(T.neq(reconstruction_dims, 0.))
reconstruction_L = -T.dot(reconstruction_dims.T, crop_X * T.log(X_hat) + (1 - crop_X) * T.log(1 - X_hat))
prediction_L = -T.dot(prediction_dims.T, crop_X * T.log(X_hat) + (1 - crop_X) * T.log(1 - X_hat))
loss = alpha * T.mean(prediction_L) + (1 - alpha) * T.mean(reconstruction_L)

# Parameter Updating
params = [weights_layer_y, bias_layer_y, weights_layer_1, bias_layer_1, weights_layer_2, bias_layer_2]
updates = RMSprop(loss, params, lr=learning_rate)

# Compiling
train = theano.function(inputs=[X], outputs=[loss, T.mean(reconstruction_L), T.mean(prediction_L), X_hat[-2:]], updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[crop_X], outputs=[model_input, X_hat], allow_input_downcast=True)
crop = theano.function(inputs=[X], outputs=crop_X, allow_input_downcast=True)

# Load the data
trX, teX, trY, teY = mnist(scale_data=False, add_noise=False)
n_batches_train = trX.shape[0] / batch_size



for i in range(epochs):
    cost_per_batch = np.zeros(n_batches_train)
    pred_cost_per_batch = np.zeros(n_batches_train)
    rec_cost_per_batch = np.zeros(n_batches_train)
    batch_num = 0
    for start, end in zip(range(0, n_batches_train * batch_size, batch_size),
                          range(batch_size, n_batches_train * batch_size, batch_size)):

        cost, pred_cost, rec_cost, tr_pred = train(trX[start:end])

        cost_per_batch[batch_num] = cost
        pred_cost_per_batch[batch_num] = pred_cost
        rec_cost_per_batch[batch_num] = rec_cost
        batch_num += 1

    print "----------------------------------------------------------"
    print "Epoch number {0}".format(i)
    print 'Avg Cost per sample per dimension Epoch %s' % str(i), np.mean(cost_per_batch)
    print 'Avg Reconstruction Cost per sample per dimension Epoch %s' % str(i), np.mean(rec_cost_per_batch)
    print 'Avg Prediction Cost per sample per dimension Epoch %s' % str(i), np.mean(pred_cost_per_batch)
    # print "1 Weights"
    # print weights_layer_1.get_value()
    # print "Y Weights"
    # print weights_layer_y.get_value()
    # scipy.misc.imsave('./predictions/tr_pred_1_{0}.gif'.format(i), tr_pred[0].reshape((28, 28)))
    # scipy.misc.imsave('./predictions/tr_pred_2_{0}.gif'.format(i), tr_pred[1].reshape((28, 28)))


# Prediction
n_imgs = 100
crop_teX = crop(teX[0:n_imgs])
mask_teX, pred_teX = predict(crop_teX)
mask_imgs = mask_teX.reshape((-1, 28, 28))
pred_imgs = pred_teX.reshape((-1, 28, 28))
true_imgs = crop_teX.reshape((-1, 28, 28))

img_array = np.zeros((28 * n_imgs, 28 * 3))
for i in range(n_imgs):
    img_array[i*28:(i+1)*28, 0:28] = true_imgs[i, :, :]
    img_array[i*28:(i+1)*28, 28:56] = mask_imgs[i, :, :]
    img_array[i*28:(i+1)*28, 56:84] = pred_imgs[i, :, :]

scipy.misc.imsave('./predictions/predae_h500_30epochs.gif', img_array)


# Save model
save_model(path="./predae.pkl", values=[weights_layer_1.eval(), bias_layer_1.eval(),
                                                      bias_layer_y.eval()])

