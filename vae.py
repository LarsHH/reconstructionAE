import theano
from theano import tensor as T
import numpy as np
from nnet_tools import rectify, floatX, init_weights, init_biases, sgd, save_model, RMSprop
from load import mnist
import scipy
from scipy.misc import imsave

rs = np.random.RandomState(1234)
rng = theano.tensor.shared_randomstreams.RandomStreams(rs.randint(999999))

def model(input_data, epsilon, w_h, mu_w, sig_w, w_h2, w_o, b_h, mu_b, sig_b, b_h2, b_o):
    h = T.tanh(T.dot(input_data, w_h) + b_h)
    mu = T.dot(h, mu_w) + mu_b
    sig = T.exp(0.5 * (T.dot(h, sig_w) + sig_b)) # Why the 0.5 ???
    z = mu + sig * epsilon
    h2 = T.tanh(T.dot(z, w_h2) + b_h2)
    # Log likelihood for decoder p(x|z)
    output = T.nnet.sigmoid(T.dot(h2, w_o) + b_o)
    return output, mu, sig


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
n_z = 20
n_visible_units = 784
blank_proportion = 0.15
alpha = 0.5
batch_size = 100
epochs = 5

X = T.fmatrix('X')
epsilon = T.fmatrix()

# Creating the masks
reconstruction_dims, updates = theano.scan(fn=MASK_blanking, sequences=[X])
prediction_dims = T.neq(X, 0.) - reconstruction_dims

# model(input_data, epsilon, w_h, mu_w, sig_w, w_h2, w_o, b_h, mu_b, sig_b, b_h2, b_o):
# Initialize Weights
weights_layer_1 = init_weights((n_visible_units, n_hidden_units), name='weights_layer_1')
bias_layer_1 = init_biases(n_hidden_units, name='bias_layer_1')
weights_layer_mu = init_weights((n_hidden_units, n_z), name='weights_layer_mu')
bias_layer_mu = init_biases(n_z, name='bias_layer_mu')
weights_layer_sig = init_weights((n_hidden_units, n_z), name='weights_layer_sig')
bias_layer_sig = init_biases(n_z, name='bias_layer_sig')
weights_layer_2 = init_weights((n_z, n_hidden_units), name='weights_layer_2')
bias_layer_2 = init_biases(n_hidden_units, name='bias_layer_2')
weights_layer_y = init_weights((n_hidden_units, n_visible_units), name='weights_layer_y')
bias_layer_y = init_biases(n_visible_units, name='bias_layer_y')


# Build Model
model_input = X * reconstruction_dims
X_hat, mu, sig = model(model_input, epsilon, weights_layer_1, weights_layer_mu, weights_layer_sig, weights_layer_2,
                       weights_layer_y, bias_layer_1, bias_layer_mu, bias_layer_sig, bias_layer_2, bias_layer_y)

# Loss
log_lik = -T.nnet.binary_crossentropy(X_hat, X).sum()

# KL Divergence
D_KL = 0.5 * T.sum(1 + 2*T.log(sig) - mu**2 - sig**2)

# Total cost ( signs correct??? )
L = log_lik + D_KL

# For prediction
reconstruction_mse = T.dot(reconstruction_dims.T, (X - X_hat)**2)/T.sum(T.neq(reconstruction_dims, 0.))
prediction_mse = T.dot(prediction_dims.T, (X - X_hat)**2)/T.sum(T.neq(prediction_dims, 0.))

# Parameter Updating
params = [weights_layer_1, weights_layer_mu, weights_layer_sig, weights_layer_2, weights_layer_y, bias_layer_1,
          bias_layer_mu, bias_layer_sig, bias_layer_2, bias_layer_y]
updates = RMSprop(-L, params, lr=learning_rate)

# Compiling
train = theano.function(inputs=[X, epsilon], outputs=[L, log_lik, D_KL], updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X, epsilon], outputs=[model_input, X_hat, T.mean(reconstruction_mse),
                                                        T.mean(prediction_mse)], allow_input_downcast=True)

# Load the data
trX, teX, trY, teY = mnist(scale_data=False)
n_batches_train = trX.shape[0] / batch_size



for i in range(epochs):
    L_per_batch = np.zeros(n_batches_train)
    log_lik_per_batch = np.zeros(n_batches_train)
    D_KL_per_batch = np.zeros(n_batches_train)
    batch_num = 0
    for start, end in zip(range(0, n_batches_train * batch_size, batch_size),
                          range(batch_size, n_batches_train * batch_size, batch_size)):
        e = np.random.normal(0, 1, (batch_size, n_z))
        batch_L, batch_log_lik, batch_D_KL = train(trX[start:end], e)

        L_per_batch[batch_num] = batch_L
        log_lik_per_batch[batch_num] = batch_log_lik
        D_KL_per_batch[batch_num] = batch_D_KL
        batch_num += 1

    print "----------------------------------------------------------"
    print "Epoch number {0}".format(i)
    print 'Avg Lower Bound Epoch %s' % str(i), np.mean(L_per_batch)
    print 'Avg Log Likelihood Epoch %s' % str(i), np.mean(log_lik_per_batch)
    print 'Avg KL Divergence Epoch %s' % str(i), np.mean(D_KL_per_batch)
    # print "1 Weights"
    # print weights_layer_1.get_value()
    # print "Y Weights"
    # print weights_layer_y.get_value()
    # scipy.misc.imsave('./predictions/tr_pred_1_{0}.gif'.format(i), tr_pred[0].reshape((28, 28)))
    # scipy.misc.imsave('./predictions/tr_pred_2_{0}.gif'.format(i), tr_pred[1].reshape((28, 28)))


# Prediction
n_imgs = 100
e = np.random.normal(0, 1, (teX.shape[0], n_z))
mask_teX, pred_teX, reconstruction_error, prediction_error = predict(teX, e)
mask_imgs = mask_teX[0:n_imgs].reshape((-1, 28, 28))
pred_imgs = pred_teX[0:n_imgs].reshape((-1, 28, 28))
true_imgs = teX[0:n_imgs].reshape((-1, 28, 28))

img_array = np.zeros((28 * n_imgs, 28 * 3))
for i in range(n_imgs):
    img_array[i*28:(i+1)*28, 0:28] = true_imgs[i, :, :]
    img_array[i*28:(i+1)*28, 28:56] = mask_imgs[i, :, :]
    img_array[i*28:(i+1)*28, 56:84] = pred_imgs[i, :, :]

scipy.misc.imsave('./predictions/vae_h500_5epochs_fixed_crops.gif', img_array)

print "Reconstruction test sqrt MSE = {0}".format(np.sqrt(reconstruction_error))
print "Prediction test sqrt MSE = {0}".format(np.sqrt(prediction_error))



# Save model
save_model(path="./predae.pkl", values=[weights_layer_1.eval(), bias_layer_1.eval(),
                                                      bias_layer_y.eval()])
