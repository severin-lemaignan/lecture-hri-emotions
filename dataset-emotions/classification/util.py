# -*- coding: utf-8 -*-

import itertools
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from matplotlib import offsetbox

# ensure the font has the required emojis!
import matplotlib
matplotlib.rc('font', family='DejaVu Sans')

def normalize(X, low, high, dtype=None):
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].    
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

def read_image(path, sz=None):
    im = Image.open(path)
    im = im.convert("L")
    # resize to given size (if given)
    if (sz is not None):
        im = im.resize(sz, Image.ANTIALIAS)
    return np.asarray(im, dtype=np.uint8).flatten()

def plot_gallery(title, images, image_shape, n_col=4, n_row=4):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)



def plot_embedding(X, labels, title=None,xlabel=None, ylabel=None):
    EMOTIONS=["😱","😀","😠","😮","😓","😜"]
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], EMOTIONS[int(labels[i])],
                 color=plt.cm.Dark2((int(labels[i]) + 1) / 8.),
                 fontdict={'weight': 'bold', 'size': 20})

    #plt.xticks([]), plt.yticks([])
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=12)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=12)
    if title is not None:
        plt.title(title, fontsize=15)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

