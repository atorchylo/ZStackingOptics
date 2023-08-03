import numpy as np
from plotters import plot_kernels
from fourier import FFT2, IFFT2

class FreeSpace:
    def __init__(self, L, k_z, coordinates):
        self.L = L
        self.k_z = k_z
        self.coordinates = coordinates
        
    def get_kernel(self):
        nu_x, nu_y = self.coordinates.get_frequency_axis()
        return np.exp(1j*(2*np.pi**2)*(nu_x**2 + nu_y**2) * self.L / self.k_z)
    
class Lens:
    def __init__(self, f, k_z, coordinates):
        self.f = f
        self.k_z = k_z
        self.coordinates = coordinates
    
    def get_kernel(self):
        x, y = self.coordinates.get_position_axis()
        return np.exp(1j*self.k_z*(x**2 + y**2)/(2*self.f))
    
class CircularAperture:
    def __init__(self, D, coordinates):
        self.D = D
        self.coordinates = coordinates
    
    def get_kernel(self):
        x, y = self.coordinates.get_position_axis()
        if self.D is None:
            return (x**2 + y**2 > -1).astype(float)
        return (x**2 + y**2 <= self.D**2/4).astype(float)

class SpaceLensSpaceSystem:
    def __init__(self, a, b, f, D, k_z, coordinates):
        self.a = a # distance to the lens
        self.b = b # distance from the lens
        self.f = f # focal distance of the lens
        self.D = D # aperture size (None if no aperture)
        self.coordinates = coordinates 
        
        # get propagation kernels
        self.space1 = FreeSpace(a, k_z, coordinates).get_kernel()
        self.lens = Lens(f, k_z, coordinates).get_kernel()
        self.aperture = CircularAperture(D, coordinates).get_kernel()
        self.space2 = FreeSpace(b, k_z, coordinates).get_kernel()
        
    def propagate(self, u_field):
        # propagate distance a
        fourrier_u_field = FFT2(u_field, self.coordinates.dx)
        fourrier_u_field *= self.space1
        u_field = IFFT2(fourrier_u_field, self.coordinates.dx)
        
        # propagate through the lens
        u_field *= self.lens
        
        # propagate through the aperture
        u_field *= self.aperture
        
        # propagate distance b
        fourrier_u_field = FFT2(u_field, self.coordinates.dx)
        fourrier_u_field *= self.space2
        u_field = IFFT2(fourrier_u_field, self.coordinates.dx)
        return u_field
    
    def plot_kernels(self, zoom=None):
        plot_kernels(
            self.space1, 
            self.lens, 
            self.aperture, 
            self.space2, 
            self.coordinates, 
            zoom
        )