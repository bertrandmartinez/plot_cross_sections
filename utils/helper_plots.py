import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.ticker import Locator
from math import factorial
import matplotlib.pyplot as plt

def scale_template(alpha):
    keys = ['font.size', 'legend.fontsize', 'legend.title_fontsize',
            'xtick.major.size', 'xtick.minor.size', 'xtick.major.width', 'xtick.minor.width', 'xtick.labelsize',
            'ytick.major.size', 'ytick.minor.size', 'ytick.major.width', 'ytick.minor.width', 'ytick.labelsize',
            'axes.labelsize', 'axes.titlesize', 'axes.linewidth', 'lines.linewidth', 'lines.markersize', 'lines.markeredgewidth']

    for key in keys:
        plt.rcParams[key] *= alpha

    # Special treatment for the figsize because it is a list
    plt.rcParams['figure.figsize'][0] *= alpha
    plt.rcParams['figure.figsize'][1] *= alpha

    # We want to keep the same amount of pixels in total
    plt.rcParams['savefig.dpi'] *= 1./alpha

def cmap_transp_low(name, alpha=1, beta=1):

    colormap = cm.get_cmap(name, 256)
    newcolors = colormap(np.linspace(0, 1, 256))
    n1 = int((1.-alpha)*colormap.N)
    n2 = int(colormap.N-n1)
    newcolors[:,-1] = np.concatenate( ( beta * np.linspace(0,1,n1) , beta * np.ones(n2) ) )
    newcmp = ListedColormap(newcolors)

    return newcmp

def cmap_transp_cen(name, alpha=1, beta=1):

    colormap = cm.get_cmap(name, 512)
    newcolors = colormap(np.linspace(0, 1, 512))
    n1 = int( 512 * ( 1. - alpha ) / 2. )
    n2 = int( 512. * alpha )
    newcolors[:,-1] = np.concatenate( ( beta * np.ones(n1), beta * np.zeros(n2), beta * np.ones(n1) ) )
    newcmp = ListedColormap(newcolors)

    return newcmp

def ticks_sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the given number
    """
    if exponent is None:
        exponent = int(np.floor(np.log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r'${0:.{2}f}\times10^{{{1:d}}}$'.format(coeff, exponent, precision)

def ticks_pow_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the given number
    """

    if(num<0):
      sgn = r'$-$'
    else:
      sgn = r'$$'

    if num != 0 :

        if not exponent:
            exponent = int(np.floor(np.log10(abs(num))))
        coeff = round(num / float(10**exponent), decimal_digits)
        if not precision:
            precision = decimal_digits

        number_formatted = r'$10^{{{1:d}}}$'.format(coeff, exponent, precision)

    else :
        number_formatted = r'$0$'

    return number_formatted

def ticks_real(num, decimal_digits=1):
    """
    Returns a string representation of the given number
    """

    return r'${0:.{1}f}$'.format(num, decimal_digits)

def multicolor_line(x, y, f, fmin, fmax, cmap, lw):

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(fmin, fmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm)

    # Set the values used for colormapping
    lc.set_array(f)
    lc.set_linewidth(lw)

    return lc

class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))