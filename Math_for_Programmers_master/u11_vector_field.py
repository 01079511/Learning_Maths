import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return -2 * y, x


def plot_vector_field(f, xmin, xmax, ymin, ymax, xstep=1, ystep=1):
    X,  Y = np.meshgrid(np.arange(xmin, xmax, xstep), np.arange(ymin, ymax, ystep))
    U = np.vectorize(lambda x, y: f(x, y)[0])(X, Y)
    V = np.vectorize(lambda x, y: f(x, y)[1])(X, Y)
    plt.quiver(X, Y, U, V, color='red')
    fig = plt.gcf()
    fig.set_size_inches(7, 7)

plot_vector_field(f,-5,5,-5,5)

