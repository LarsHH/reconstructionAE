import theano
from theano import tensor as T
import numpy as np
import cPickle


def save_model(path, values):
    return cPickle.dump(values, open(path, "wb"))

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape, array=None, name="", scale=0.01):
    if array is not None:
        return theano.shared(floatX(array), name=name, borrow=True)
    else:
        return theano.shared(floatX(np.random.randn(*shape) * scale), name=name, borrow=True)

def init_biases(shape, array=None, name=""):
    if array is not None:
        return theano.shared(floatX(array), name=name, borrow=True)
    else:
        return theano.shared(floatX(np.zeros(shape)), name=name, borrow=True)

def rectify(X):
    return T.maximum(X, 0.)

def sigmoid(X):
    return 1/(1 + T.exp(-X))

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def ADAM(cost, params, t, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        # Initialize m and v
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        # Updates to moments
        m_new = beta1 * m + (1-beta1) * g
        v_new = beta2 * v + (1-beta2) * g**2
        # Bias corrected estimates
        m_hat = m_new/(1 - beta1**t)
        v_hat = v_new/(1 - beta2**t)
        # update these
        updates.append((m, m_new))
        updates.append((v, v_new))
        updates.append((p, p - alpha * m_hat / (T.sqrt(v_hat) + eps)))
    return updates

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def sgd(loss, params, lr=0.01):
    grads = T.grad(cost=loss, wrt=params)
    updates = []
    for parameter, gradient in zip(params, grads):
        updates.append([parameter, parameter - gradient * lr])
    return updates

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


