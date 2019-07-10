#regression demo 代码
import numpy as np
import matplotlib.pyplot as plt


def gradient_decent(_x_data, _y_data, _b, _w, _iteration, _lr):
    """
    Gradient Decent
    :param _x_data:
    :param _y_data:
    :param _b:
    :param _w:
    :param _iteration:
    :param _lr:
    :return:
    """
    _b_history = [_b]
    _w_history = [_w]

    for _i in range(_iteration):
        b_grad = 0.0
        w_grad = 0.0

        for _n in range(len(x_data)):
            b_grad = b_grad - 2.0 * (y_data[_n] - _b - _w * _x_data[_n]) * 1.0
            w_grad = w_grad - 2.0 * (y_data[_n] - _b - _w * _x_data[_n]) * _x_data[_n]

        _b = _b - _lr * b_grad
        _w = _w - _lr * w_grad

        _b_history.append(_b)
        _w_history.append(_w)
    return _b_history, _w_history


def adagrad(_x_data, _y_data, _b, _w, _iteration, _lr):
    """
    Adagrad: auto adapt the learning rate.
    :param _x_data:
    :param _y_data:
    :param _b:
    :param _w:
    :param _iteration:
    :param _lr:
    :return:
    """
    _b_history = [_b]
    _w_history = [_w]
    lr_b = 0
    lr_w = 0

    for _i in range(_iteration):
        b_grad = 0.0
        w_grad = 0.0

        for _n in range(len(x_data)):
            b_grad = b_grad - 2.0 * (y_data[_n] - _b - _w * _x_data[_n]) * 1.0
            w_grad = w_grad - 2.0 * (y_data[_n] - _b - _w * _x_data[_n]) * _x_data[_n]

        lr_b = lr_b + b_grad ** 2
        lr_w = lr_w + w_grad ** 2
        _b = _b - _lr / np.sqrt(lr_b) * b_grad
        _w = _w - _lr / np.sqrt(lr_w) * w_grad

        _b_history.append(_b)
        _w_history.append(_w)
    return _b_history, _w_history


def plot_result(_b_history, _w_history):
    """
    Plot Result
    :param _b_history:
    :param _w_history:
    :return:
    """
    plt.contourf(x, y, z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
    plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
    plt.plot(_b_history, _w_history, 'o-', ms=3, lw=1.5, color='black')
    plt.xlim(-200, -100)
    plt.ylim(-5, 5)
    plt.xlabel(r'$b$', fontsize=16)
    plt.ylabel(r'$w$', fontsize=16)
    plt.show()


if __name__ == '__main__':
    x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
    y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]

    x = np.arange(-200, -100, 1)  # bias
    y = np.arange(-5, 5, 0.1)  # weight
    z = np.zeros((len(x), len(y)))
    X, Y = np.meshgrid(x, y)

    for i in range(len(x)):
        for j in range(len(y)):
            b = x[i]
            w = y[j]
            z[j][i] = 0

            for n in range(len(x_data)):
                z[j][i] = z[j][i] + (y_data[n] - b - w * x_data[n]) ** 2

            z[j][i] = z[j][i] / len(x_data)

    b = -120
    w = -4
    lr = 1
    iteration = 100000

    # b_history, w_history = gradient_decent(x_data, y_data, b, w, iteration, lr)
    b_history, w_history = adagrad(x_data, y_data, b, w, iteration, lr)

    plot_result(b_history, w_history)
