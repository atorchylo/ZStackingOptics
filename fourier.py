import numpy as np
from scipy.fftpack import fft, ifft, fft2, ifft2
from scipy.fftpack import fftshift, ifftshift, fftfreq

class Coordinates:
    """Defines a coordinate system for both position and momentum space"""
    def __init__(self, x_min, x_max, N):
        # set up position coordinates
        self.x = np.linspace(x_min, x_max, N)
        self.dx = 2 * x_max / (N - 1)
        
        # compute frequency window
        self.nu = fftshift(fftfreq(N, self.dx))
        self.dnu = 1 / (N * self.dx)
        nu_min, nu_max = self.nu[0], self.nu[-1]
        
        # for plotting purposes 
        self.pos_extent = [x_min, x_max, x_min, x_max]
        self.freq_extent = [nu_min, nu_max, nu_min, nu_max]
    
    def get_position_axis(self):
        x = self.x
        y = x[:,np.newaxis]
        return x, y
    
    def get_frequency_axis(self):
        nu_x = self.nu
        nu_y = nu_x[:,np.newaxis]
        return nu_x, nu_y

def FFT2(position_data, resolution):
    """
    Computes a 2d fast fourier transform of centered input data.
    
    fourrier_data - centered fourier transform of data.
    """
    fourrier_data = resolution**2 * fftshift(fft2(ifftshift(position_data)))
    return fourrier_data

def IFFT2(fourrier_data, resolution):
    """
    Computes a 2d inverse fast fourier transform of the centered input data.
    
    position_data - centered position data of the given fourier signa;
    """
    position_data = 1/resolution**2 * fftshift(ifft2(ifftshift(fourrier_data)))
    return position_data