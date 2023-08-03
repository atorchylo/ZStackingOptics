import matplotlib.pyplot as plt
import numpy as np

def plot_2d(field, coord, fourier_space=False, title=None, zoom=None):
    """Plots absolute value and phase  of the field"""
    fig, ax = plt.subplots(1, 4, figsize=(14, 7), gridspec_kw={'width_ratios': [12, 1, 12, 1]})
    extent = coord.freq_extent if fourier_space else coord.pos_extent
    abs_img = ax[0].imshow(np.abs(field), extent=extent)
    plt.colorbar(abs_img, ax[1])
    phase_img = ax[2].imshow(np.angle(field), extent=extent)
    plt.colorbar(abs_img, ax[3])
    if title is not None:
        ax[0].set_title('abs of ' + title)
        ax[2].set_title('phase of ' + title)
    if zoom is not None:
        ax[0].set_xlim(-zoom,zoom)
        ax[0].set_ylim(-zoom,zoom)
        ax[2].set_xlim(-zoom,zoom)
        ax[2].set_ylim(-zoom,zoom)
    # turn off ticks
    ax[2].set_yticks([])
    plt.show()
    
def plot_kernels(space1, lens, aperture, space2, coord, zoom):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(space1.real, extent=coord.freq_extent)
    ax[0].set_title(f'propagation by L=a kernel (zoom={zoom})')
    ax[1].imshow(lens.real, extent=coord.pos_extent)
    ax[1].set_title('lens kernel')
    ax[2].imshow(aperture, extent=coord.pos_extent)
    ax[2].set_title('aperture kernel')
    ax[3].imshow(space2.real, extent=coord.freq_extent)
    ax[3].set_title(f'propagation by L=b kernel (zoom={zoom})')
    if zoom is not None:
        ax[0].set_xlim(-zoom, zoom)
        ax[0].set_ylim(-zoom, zoom)
        ax[3].set_xlim(-zoom, zoom)
        ax[3].set_ylim(-zoom, zoom)
    plt.tight_layout()
    plt.show()