import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mpl_toolkits.axes_grid1 as mplax
import voronoimapspygad as vm
from astropy.io import fits
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from plotbin.display_bins_generators import display_bins_generators

labelsize=24      # Style parameters for the standard Voronoi plot
digitsize=24
textsize=26
titlesize=26
ncbarlabels=3
contourthickness=1

# qty: 'vel', 'sigma', 'h3', 'h4'
def plot_quantity(filename_bin, filename_non, plotnum, qty='vel', limits=[-20, 20]):
    
    hdu_v = fits.open(filename_bin)
    hdu = fits.open(filename_non)
    
    data_v = hdu_v[1].data
    data = hdu[2].data

    if qty == 'vel':
        v = data_v['VPXF'] - np.mean(data_v['VPXF'])
    elif qty == 'sigma':
        v = data_v['SPXF']
    elif qty == 'h3':
        v = data_v['H3PXF']
    else:
        v = data_v['H4PXF']

    plt.subplot(220 + plotnum)

    img = display_bins_generators(data_v['YS'], data_v['XS'], v, data['D'], data['A'], vmin=limits[0], vmax=limits[1], pixelsize=None)
    plt.tricontour(data_v['YS'], data_v['XS'], -2.5*np.log10(data_v['FLUX']/np.max(data_v['FLUX'])), levels=np.arange(20), colors='k', linewidths=contourthickness)

    if plotnum == 1:
        plt.title("$ \langle V \\rangle $ [km/s]", fontsize=titlesize, y=1.04)
        plt.ylabel("arcsec", fontsize=textsize)
    if plotnum == 2:
        plt.title("$\sigma$ [km/s]", fontsize=titlesize, y=1.04)
    if plotnum == 3:
        plt.title("$h_3$", fontsize=titlesize, y=1.04)
        plt.ylabel("arcsec", fontsize=textsize)
        plt.xlabel("arcsec", fontsize=textsize)
    if plotnum == 4:
        plt.title("$h_4$", fontsize=titlesize, y=1.04)
        plt.xlabel("arcsec", fontsize=textsize)

    ax = plt.gca()
    
    divider = mplax.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cticks = np.array([limits[0], np.sum(limits)/2, limits[1]])
    
    cb = plt.colorbar(img, cax = cax, ticks = cticks)

    ax.tick_params(direction = 'in', which='both', right=True, top=True, labelleft=False, labelsize=digitsize)
    ax.set_aspect('equal')
    
    cax.tick_params(direction = 'in', labelsize=digitsize*1.05)


targets = ['NGC3414_r6', 'NGC3522_r4', 'NGC4111_r1', 'NGC4472_r8']

target_num = 3


filename_v = 'PXF_bin_MS_'+targets[target_num]+'_idl.fits'
filename = 'MS_'+targets[target_num]+'_C2D.fits'

#fig, ax = plt.subplots(2, 2)
#axes = ax.flatten()

fig = plt.figure(figsize=(10, 10))

#matplotlib.rcParams.update({'font.size':14})

plot_quantity(filename_v, filename, 1, qty='vel', limits=[-50, 50])
plot_quantity(filename_v, filename, 2, qty='sigma', limits=[260, 330])
plot_quantity(filename_v, filename, 3, qty='h3', limits=[-0.1, 0.1])
plot_quantity(filename_v, filename, 4, qty='h4', limits=[-0.1, 0.1])

plt.subplots_adjust(wspace=0.3, hspace=0)



plt.savefig(targets[target_num]+'_voronoi.png', bbox_inches='tight')
