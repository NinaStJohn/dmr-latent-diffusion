
import numpy as np
from porespy.tools import Results

def _parse_histogram(h, voxel_size=1, density=True):
    delta_x = h[1]
    P = h[0]
    bin_widths = delta_x[1:] - delta_x[:-1]
    temp = P * (bin_widths)
    C = np.cumsum(temp[-1::-1])[-1::-1]
    S = P * (bin_widths)
    if not density:
        P /= np.max(P)
        temp_sum = np.sum(P * bin_widths)
        C /= temp_sum
        S /= temp_sum

    bin_edges = delta_x * voxel_size
    bin_widths = (bin_widths) * voxel_size
    bin_centers = ((delta_x[1:] + delta_x[:-1]) / 2) * voxel_size
    hist = Results()
    hist.pdf = P
    hist.cdf = C
    hist.relfreq = S
    hist.bin_centers = bin_centers
    hist.bin_edges = bin_edges
    hist.bin_widths = bin_widths
    return hist

def pore_size_distribution(im, bins=10, log=True, voxel_size=1):
    r"""
    Calculate a pore-size distribution based on the image produced by the
    ``porosimetry`` or ``local_thickness`` functions.

    Parameters
    ----------
    im : ndarray
        The array of containing the sizes of the largest sphere that overlaps
        each voxel.  Obtained from either ``porosimetry`` or
        ``local_thickness``.
    bins : scalar or array_like
        Either an array of bin sizes to use, or the number of bins that should
        be automatically generated that span the data range.
    log : boolean
        If ``True`` (default) the size data is converted to log (base-10)
        values before processing.  This can help to plot wide size
        distributions or to better visualize the in the small size region.
        Note that you should not anti-log the radii values in the retunred
        ``tuple``, since the binning is performed on the logged radii values.
    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so the
        user can apply the scaling to the returned results after the fact.

    Returns
    -------
    result : Results object
        A custom object with the following data added as named attributes:

        *R* or *logR*
            Radius, equivalent to ``bin_centers``
        *pdf*
            Probability density function
        *cdf*
            Cumulative density function
        *satn*
            Phase saturation in differential form.  For the cumulative
            saturation, just use *cfd* which is already normalized to 1.
        *bin_centers*
            The center point of each bin
        *bin_edges*
            Locations of bin divisions, including 1 more value than
            the number of bins
        *bin_widths*
            Useful for passing to the ``width`` argument of
            ``matplotlib.pyplot.bar``

    Notes
    -----
    (1) To ensure the returned values represent actual sizes you can manually
    scale the input image by the voxel size first (``im *= voxel_size``)

    plt.bar(psd.R, psd.satn, width=psd.bin_widths, edgecolor='k')

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/pore_size_distribution.html>`_
    to view online example.

    """
    pore_sizes = pore_size_distribution_calc(im)

    # # Plotting the pore size distribution
    # plt.hist(pore_sizes, bins=range(1, max(pore_sizes)+2), edgecolor='black')
    # plt.xlabel('Pore Size (Number of Pixels)')
    # plt.ylabel('Frequency')
    # plt.title('Pore Size Distribution')
    # plt.grid(True)
    # plt.savefig("z/psd.png")

    # # im = im.flatten()
    # # vals = im[im > 0] * voxel_size
    # # if log:
    # #     vals = np.log10(vals)
    h = _parse_histogram(np.histogram(pore_sizes, bins=range(1, max(pore_sizes)+2), density=True))
    cld = Results()
    cld[f"{log*'Log' + 'R'}"] = h.bin_centers
    cld.pdf = h.pdf
    cld.cdf = h.cdf
    cld.satn = h.relfreq
    cld.bin_centers = h.bin_centers
    cld.bin_edges = h.bin_edges
    cld.bin_widths = h.bin_widths
    return cld


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

def pore_size_distribution_calc(binary_image):
    """
    Calculate the Pore Size Distribution (PSD) for the given phase in a binary image.
    
    Parameters:
    binary_image : numpy.ndarray
        2D array representing the binary image (with values 0 and 255).
    phase_value : int, optional
        The value representing the pore phase (default is 0).
    
    Returns:
    sizes : numpy.ndarray
        Array of sizes of identified pores.
    """
    
    # Label connected components in the binary image (pores)
    labeled_array, num_features = label(binary_image)
    
    # Calculate the size of each labeled region (pore)
    pore_sizes = np.bincount(labeled_array.ravel())[1:]  # Exclude the background (label 0)
    
    return pore_sizes // 100
