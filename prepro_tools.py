import numpy as np
import cPickle


def center(X, name='train'):
    if name == 'train':
        X_mean = np.nanmean(X, axis=0)
        X -= X_mean
        f = open('./prepro/train_mean.pkl', 'wb')
        cPickle.dump(X_mean, f)
        f.close()
    else:
        f = open('./prepro/train_mean.pkl', 'rb')
        X_mean = cPickle.load(f)
        f.close()
        X -= X_mean
    return X



def scale(X, name='train', scaling_type='tanh'):
    if scaling_type=='tanh':
        if name == 'train':
            X_abs_max = np.abs(X).max(axis=0)
            X /= X_abs_max
            f = open('./prepro/train_abs_max.pkl', 'wb')
            cPickle.dump(X_abs_max, f)
            f.close()
        else:
            f = open('./prepro/train_abs_max.pkl', 'rb')
            X_abs_max = cPickle.load(f)
            f.close()
            X /= X_abs_max
    else:
        if name == 'train':
            X_std = np.nanstd(X, axis=0)
            X /= X_std
            f = open('./prepro/train_std.pkl', 'wb')
            cPickle.dump(X_std, f)
            f.close()
        else:
            f = open('./prepro/train_std.pkl', 'rb')
            X_std = cPickle.load(f)
            f.close()
            X /= X_std
    return X
