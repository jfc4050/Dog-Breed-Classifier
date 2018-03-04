import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def path_to_arr(path, target_dim):
    img = Image.open(path).resize((target_dim, target_dim))
    return np.asarray(img)


def plot_model_history(history):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    axes[0].plot(history.history['loss'])
    axes[0].plot(history.history['val_loss'])
    axes[0].legend(['loss', 'val_loss'], loc='upper right')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')

    axes[1].plot(history.history['acc'])
    axes[1].plot(history.history['val_acc'])
    axes[1].legend(['acc', 'val_acc'], loc='upper right')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')

    fig.show()


def get_errors(yhat, y):
    pred   = np.argmax(yhat, axis=1)
    actual = np.argmax(y,    axis=1)

    wrong = (pred != actual)
    wrong_indices = np.array(np.nonzero(wrong))

    errors = np.vstack((wrong_indices, np.squeeze(np.vstack((pred, actual))[:, wrong_indices])))

    return errors


def print_input_shapes(split_data):
    for name, matrix in split_data.items():
        print('{}.shape = {}'.format(name, matrix.shape))


def split_train_cv(x, y, cv_frac):
    n = x.shape[0]
    cv_start = int((1 - cv_frac) * n)

    permut = np.random.permutation(n)
    permut_tr = permut[:cv_start]
    permut_cv = permut[cv_start:]

    x_tr = x[permut_tr]
    y_tr = y[permut_tr]

    x_cv = x[permut_cv]
    y_cv = y[permut_cv]

    split_data = {'x': x,
                  'y': y,
                  'x_tr': x_tr,
                  'y_tr': y_tr,
                  'x_cv': x_cv,
                  'y_cv': y_cv}

    print_input_shapes(split_data)

    return split_data, permut_tr, permut_cv