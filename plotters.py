import matplotlib.pyplot as plt
import numpy as np

def plot_field(field, coord, fourier_space=False, title=None, zoom=None):
    """Plots absolute value and phase  of the field"""
    fig, ax = plt.subplots(1, 4, figsize=(14, 7), gridspec_kw={'width_ratios': [12, 1, 12, 1]})
    extent = coord.freq_extent if fourier_space else coord.pos_extent
    abs_img = ax[0].imshow(np.abs(field), extent=extent)
    plt.colorbar(abs_img, ax[1])
    phase_img = ax[2].imshow(np.angle(field), extent=extent)
    plt.colorbar(phase_img, ax[3])
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
    
def plot_img(img, coord, zoom=None, FT=False, title=None):
    """Plot's a 2d scalar array along with 1d plot of middle row pixels"""
    extent = coord.freq_extent if FT else coord.pos_extent
    ticks = coord.nu if FT else coord.x
    
    # plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img, extent = extent, cmap='coolwarm')
    ax[1].plot(ticks, img[img.shape[0]//2, :])
    
    if zoom is not None:
        ax[0].set_xlim(-zoom, zoom)
        ax[0].set_ylim(-zoom, zoom)
        ax[1].set_xlim(-zoom, zoom)
        
    if title is not None:
        fig.suptitle(title)
    plt.show()