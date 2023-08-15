import numpy as np
from fourier import FFT2, IFFT2

class FreeSpace:
    def __init__(self, L, k_z, coordinates):
        self.L = L
        self.k_z = k_z
        self.coordinates = coordinates
        self.kernel = self.get_kernel()
        
    def get_kernel(self):
        nu_x, nu_y = self.coordinates.get_frequency_axis()
        return np.exp(1j*(2*np.pi**2)*(nu_x**2 + nu_y**2) * self.L / self.k_z)
    
    def propagate(self, field):
        fourrier_field = FFT2(field, self.coordinates.dx)
        fourrier_field *= self.kernel
        field = IFFT2(fourrier_field, self.coordinates.dx)
        return field
    
class Lens:
    def __init__(self, f, k_z, coordinates):
        self.f = f
        self.k_z = k_z
        self.coordinates = coordinates
        self.kernel = self.get_kernel()
    
    def get_kernel(self):
        x, y = self.coordinates.get_position_axis()
        return np.exp(1j*self.k_z*(x**2 + y**2)/(2*self.f))
    
    def propagate(self, field):
        return field * self.kernel
    
class CircularAperture:
    def __init__(self, D, coordinates):
        self.D = D
        self.coordinates = coordinates
        self.kernel = self.get_kernel()
    
    def get_kernel(self):
        x, y = self.coordinates.get_position_axis()
        if self.D is None:
            return (x**2 + y**2 > -1).astype(float)
        return (x**2 + y**2 <= self.D**2/4).astype(float)
    
    def propagate(self, field):
        return field * self.kernel
    
class GaussianAperture:
    def __init__(self, D, coordinates):
        self.D = D
        self.coordinates = coordinates
        self.kernel = self.get_kernel()
    
    def get_kernel(self):
        x, y = self.coordinates.get_position_axis()
        if self.D is None:
            return (x**2 + y**2 > -1).astype(float)
        return np.exp(-(x**2 + y**2)/(self.D**2/4))
    
    def propagate(self, field):
        return field * self.kernel

class SpaceLensSpaceSystem:
    def __init__(self, a, b, f, D, k_z, coordinates):
        self.a = a # distance to the lens
        self.b = b # distance from the lens
        self.f = f # focal distance of the lens
        self.D = D # aperture size (None if no aperture)
        self.coordinates = coordinates 
        
        # get propagation kernels
        self.space_a = FreeSpace(a, k_z, coordinates)
        self.lens = Lens(f, k_z, coordinates)
        self.aperture = CircularAperture(D, coordinates)
        self.space_b = FreeSpace(b, k_z, coordinates)
        
    def propagate(self, field):
        # propagate distance a
        field = self.space_a.propagate(field)
        
        # propagate through the lens
        field = self.lens.propagate(field)
        
        # propagate through the aperture
        field = self.aperture.propagate(field)
        
        # propagate distance b
        field = self.space_b.propagate(field)
        return field

class GaussianPSF:
    def __init__(self, delta, k, b, D, coordinates):
        self.delta = delta
        self.D = D
        self.k = k
        self.w0 = 4 * b / (D * k)
        self.zR = 8 * b**2 / (D**2 * k)
        self.w = self.w0 * np.sqrt(1 + (b**2 * delta / self.zR)**2)
        self.coordinates = coordinates
        self.kernel = self.get_kernel()
    
    def get_kernel(self):
        x, y = self.coordinates.get_position_axis()
        kernel = self.w0 / self.w * np.exp(-(x**2 + y**2) / self.w**2)
        return kernel
    
    def propagate(self, field):
        fourrier_field = FFT2(field, self.coordinates.dx)
        fourrier_field *= self.kernel
        field = IFFT2(fourrier_field, self.coordinates.dx)
        return field
    
    
class GaussianIntensityPSF:
    def __init__(self, delta, k, b, D, coordinates):
        self.delta = delta
        self.D = D
        self.k = k
        self.w0 = 4 * b / (D * k)
        self.zR = 8 * b**2 / (D**2 * k)
        self.w = self.w0 * np.sqrt(1 + (b**2 * delta / self.zR)**2)
        self.coordinates = coordinates
        self.kernel = self.get_kernel()
        
    def get_kernel(self):
        x, y = self.coordinates.get_position_axis()
        kernel = (self.w0 / self.w)**2 * np.exp(- 2 * (x**2 + y**2) / self.w**2)
        return kernel
    
    def propagate(self, img):
        fourrier_img = FFT2(img, self.coordinates.dx)
        fourrier_kernel = FFT2(self.kernel, self.coordinates.dx)
        fourrier_img *= fourrier_kernel
        img = IFFT2(fourrier_img, self.coordinates.dx)
        return np.abs(img)