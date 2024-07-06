import matplotlib.pyplot as plt
import numpy as np


def plot_function(f, tmin, tmax, tlabel=None, xlabel=None, axes=False, **kwargs):
    ts = np.linspace(tmin, tmax, 1000)
    if tlabel:
        plt.xlabel(tlabel, fontsize=18)
    if xlabel:
        plt.ylabel(xlabel, fontsize=18)
    plt.plot(ts, [f(t) for t in ts], **kwargs)
    plt.show()
    plt.close()
    if axes:
        total_t = tmax-tmin
        plt.plot([tmin-total_t/10, tmax+total_t/10], [0, 0], c='k',linewidth=1)
        plt.xlim(tmin-total_t/10, tmax+total_t/10)
        xmin, xmax = plt.ylim()
        plt.plot([0, 0], [xmin, xmax], c='k', linewidth=1)
        plt.ylim(xmin, xmax)


def plot_volume(f, tmin, tmax, axes=False, **kwargs):
    plot_function(f, tmin, tmax, tlabel="time (hr)", xlabel="volume (bbl)", axes=axes, **kwargs)


def plot_flow_rate(f, tmin, tmax, axes=False, **kwargs):
    plot_function(f, tmin, tmax, tlabel="time (hr)", xlabel="flow rate (bbl/hr)", axes=axes, **kwargs)


def volume(t):
    return (t-4)**3 / 64 + 3.3


def flow_rate(t):
    return 3*(t-4)**2 / 64


plot_volume(volume, 0, 10)


def average_flow_rate(v, t1, t2):
    return (v(t2) - v(t1))/(t2 - t1)


print('volume(4)={0:.2f},\n'
      'volume(9)={1:.2f},\n'
      'average_flow_rate(volume,4,9)={2:.2f}'.format(volume(4),
                                                 volume(9),
                                                 average_flow_rate(volume, 4, 9)))

