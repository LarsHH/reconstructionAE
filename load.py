import numpy as np
import os
import scipy.io
from prepro_tools import center, scale


datasets_dir = '../data/'


def one_hot(x,n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x),n))
    o_h[np.arange(len(x)),x] = 1
    return o_h


def mnist(ntrain=50000, ntest=10000, onehot=True, center_data=False, scale_data=False,
          random_crop=True, add_pattern=True):
    data_dir = os.path.join(datasets_dir,'mnist/')
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28*28)).astype(float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28*28)).astype(float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trX = trX/255.
    teX = teX/255.

    trX = trX[:ntrain]
    trY = trY[:ntrain]

    print "Shuffling training data..."
    np.random.shuffle(trX)
    print "Shuffling done."

    teX = teX[:ntest]
    teY = teY[:ntest]

    # if onehot:
    #     trY = one_hot(trY, 10)
    #     teY = one_hot(teY, 10)
    # else:
    #     trY = np.asarray(trY)
    #     teY = np.asarray(teY)

    add_noise=False
    if add_noise:
        trX += np.abs(np.random.normal(size=trX.shape, scale=0.01))
        trX = trX / np.float(trX.max())
        teX += np.abs(np.random.normal(size=teX.shape, scale=0.01))
        teX = trX / np.float(trX.max())

    if add_pattern:
        print "Adding pattern to images..."
        hi = 0.4
        lo = 0.2
        ones = lo * np.ones((4,4))
        zeros = hi * np.ones((4,4))
        tile1 = np.vstack((np.hstack((ones, zeros)), np.hstack((zeros, ones))))
        tile1 = np.tile(tile1, (4,4))[:28,:28]
        tile1 = tile1.reshape(-1)
        tile2 = np.vstack((np.hstack((ones, zeros)), np.hstack((ones, zeros))))
        tile2 = np.tile(tile2, (4,4))[:28,:28]
        tile2 = tile2.reshape(-1)
        tile3 = np.vstack((np.hstack((ones, ones)), np.hstack((zeros, zeros))))
        tile3 = np.tile(tile3, (4,4))[:28,:28]
        tile3 = tile3.reshape(-1)
        tiles = [tile1, tile2, tile3, tile1, tile2, tile3, tile1, tile2, tile3, tile1]

        # rand_idxs = np.random.randint(0,3, size=trX.shape[0])

        pattern = np.asarray([tiles[idx] for idx in trY])
        trX += pattern
        trX = trX / (1. + hi)

        rand_idxs = np.random.randint(0,3, size=teX.shape[0])
        pattern = np.asarray([tiles[idx] for idx in teY])
        teX += pattern
        teX = teX / (1. + hi)
        print "Adding pattern done."



    if center_data:
        print "Centering images..."
        trX = center(trX, name='train')
        teX = center(teX, name='test')
        print "Centering done."

    if scale_data:
        print "Scaling images..."
        trX = scale(trX, name='train', scaling_type='linear')
        teX = scale(teX, name='test', scaling_type='linear')
        print "Scaling done."

    if random_crop:
        print "Cropping images..."
        crop_size = 200
        for row in trX:
            idx0 = np.random.randint(low=0, high=crop_size, size=1)
            idx1 = np.random.randint(low=784-crop_size, high=784, size=1)
            row[0:idx0] = 0.
            row[idx1:] = 0.
        for row in teX:
            idx0 = np.random.randint(low=0, high=crop_size, size=1)
            idx1 = np.random.randint(low=784-crop_size, high=784, size=1)
            row[0:idx0] = 0.
            row[idx1:] = 0.
        print "Cropping done."

    return trX, teX, trY, teY


def frey(scaled=True, centered=False):
    data_dir = os.path.join(datasets_dir,'frey/')

    data = scipy.io.loadmat(os.path.join(data_dir,'frey_rawface.mat'))
    data = data['ff'].T # original array is 560x1965. We want the samples as rows

    trX = data[:1000]
    teX = data[1000:]
    if scaled:
        trX /= 255.
        teX /= 255.
    if centered:
        trX_mean = trX.mean(axis=0)
        trX = trX - trX_mean
        teX = teX - trX_mean

    return trX, teX
