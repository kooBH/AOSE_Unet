import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftshift

def cross_correlation_using_fft(x, y):
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)

# shift < 0 means that y starts 'shift' time steps before x # shift > 0 means that y starts 'shift' time steps after x
def compute_shift(x, y):
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift

def sync_wav(anchor,target):
    diff = compute_shift(anchor[0,:],target[0,:])
    if diff > 0 :
        synced_target = target[0,diff:]
    else :
        synced_target = target[0,-diff:]
    cutted_anchor = anchor[0,:len(synced_target)]
    return cutted_anchor,synced_target
