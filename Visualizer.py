# import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
import itertools
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix

if os.name == 'nt':
    fontp = FontProperties(family='Tahoma', size=10)
else:
    fontp = FontProperties(family='Tlwg Typo', size=10)


def plot_ConfuseMatrix(tp, fp, fn, tn, cm=None, labels=('positive', 'negative'), title='Confusion Matrix',
                       tight_layout=False, fig_size=(10, 10), save=None):
    if cm is None:
        cm = np.array([[1, 0], [0, 1]])

    fig = plt.figure(figsize=fig_size)
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('summer'))
    plt.title(title, fontproperties=FontProperties(None, size=14))
    plt.xticks(np.arange(2), labels, rotation=45)
    plt.yticks(np.arange(2), labels)
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.colorbar()

    s = [[tp, fp], [fn, tn]]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]), horizontalalignment="center",
                     fontproperties=FontProperties(None, size=14))

    if tight_layout:
        plt.tight_layout()

    if save is not None:
        plt.savefig(save)
        plt.close()
    else:
        return fig


def plot_confusion_matrix(cls_true, cls_pred, cls_names=None, normalize=False,
                          filter_true=None, filter_pred=None, title='Confusion matrix',
                          cmap='viridis', fig_size=(9, 9), save=None):
    if cls_names is None:
        cls_names = np.unique(cls_true)

    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred, labels=cls_names)

    real = cm.copy()
    if normalize:
        div = cm.sum(axis=1)[:, np.newaxis]
        div[div == 0] = 1
        cm = cm.astype('float') / div

    x_tick = np.arange(len(cls_names))
    x_tick_label = cls_names
    y_tick = np.arange(len(cls_names))
    y_tick_label = cls_names

    if filter_true is not None:
        idx = [cls_names.index(c) for c in filter_true]
        cm = cm[idx, :]
        real = real[idx, :]
        y_tick = np.arange(len(filter_true))
        y_tick_label = filter_true

    if filter_pred is not None:
        idx = [cls_names.index(c) for c in filter_pred]
        cm = cm[:, idx]
        real = real[:, idx]
        x_tick = np.arange(len(filter_pred))
        x_tick_label = filter_pred

    plt.figure(figsize=fig_size)
    ax = plt.gca()
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.xticks(x_tick, x_tick_label, rotation=45, fontproperties=fontp)
    plt.yticks(y_tick, y_tick_label, fontproperties=fontp)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(real.shape[0]), range(real.shape[1])):
        plt.text(j, i, str(real[i, j]), horizontalalignment="center",
                 fontproperties=FontProperties(None, size=8),
                 color="black" if cm[i, j] > thresh else "white")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.tight_layout()

    if save is not None:
        plt.savefig(save)
        plt.close()


def plot_bars(x, y, title='', ylim=None, save=None):
    plt.figure()
    bars = plt.bar(x, y)
    plt.ylim(ylim)
    plt.title(title)
    for b in bars:
        plt.annotate('{:.4f}'.format(b.get_height()),
                     xy=(b.get_x(), b.get_height()))

    if save is not None:
        plt.savefig(save)
        plt.close()


def plot_graphs(x_list, legends, title, ylabel, xlabel='epoch', xlim=None, ylim=None, save=None):
    plt.figure()
    for x in x_list:
        plt.plot(x)

    plt.legend(legends)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(xlim)
    plt.ylim(ylim)

    if save is not None:
        plt.savefig(save)
        plt.close()


def plot_x(x, title=None, fig_size=(12, 10)):
    fig = plt.figure(figsize=fig_size)
    x = np.squeeze(x)

    if len(x.shape) == 1:
        plt.plot(x)

    elif len(x.shape) == 2:
        plt.imshow(x, cmap='gray')
        plt.axis('off')

    elif len(x.shape) == 3:
        if x.shape[-1] == 3:
            plt.imshow(x)
            plt.axis('off')
        else:
            fig = plot_multiImage(x.transpose(2, 0, 1), fig_size=fig_size)

    elif len(x.shape) == 4:
        fig = plot_multiImage(x.transpose(3, 0, 1, 2), fig_size=fig_size)

    if title is not None:
        fig.suptitle(title)

    return fig


# images in shape (amount, h, w, c).
def plot_multiImage(images, labels=None, pred=None, title=None, fig_size=(12, 10), tight_layout=False, save=None):
    n = int(np.ceil(np.sqrt(len(images))))
    fig = plt.figure(figsize=fig_size)

    for i in range(len(images)):
        ax = fig.add_subplot(n, n, i + 1)
        ax.imshow(images[i])

        if labels is not None:
            ax.set_xlabel(str(labels[i]), color='g', fontproperties=fontp)
        if labels is not None and pred is not None:
            if labels[i] == pred[i]:
                clr = 'g'
            else:
                if len(labels[i]) == len(pred[i]):
                    clr = 'm'
                else:
                    clr = 'r'

            ax.set_xlabel('True: {}\nPred : {}'.format(u'' + labels[i], u'' + pred[i]),
                          color=clr, fontproperties=fontp)

    if title is not None:
        fig.suptitle(title)

    if tight_layout:  # This make process slow if too many images.
        fig.tight_layout()

    if save is not None:
        plt.savefig(save)
        plt.close()
    else:
        return fig


def get_fig_image(fig):  # figure to array of image.
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer._renderer)
    return img
