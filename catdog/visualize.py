
from __future__ import division, print_function
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, classes, 
                          normalize=False,
                          cmap='viridis',
                          ax=None,
                          title=None,
                          xlabel='Predicted Label',
                          ylabel='True label',
                          clabel='Counts'):
    """Plot confusion matrix. Code adapted from:
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix output from sklearn.metrics.confusion_matrix
    classes : array-like
        Class names
    normalize : bool
        Normalize the rows of the confusion matrix
    ax : plt.axes.Axes
        Axes to plot the confusion matrix on
    
    Returns
    -------
    ax : plt.axes.Axes
        Axes containing the confusion matrix plot
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        clabel='Probability'

    if ax is None:
        ax = plt.gca()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, label=clabel)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='black' if cm[i, j] > thresh else 'white')
    
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    plt.tight_layout()

    return ax
