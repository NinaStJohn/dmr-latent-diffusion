
import numpy as np
from scipy.ndimage import label
import numpy as np
from scipy import fft as sp_ft
from porespy.tools import Results

def porosity(im):
    im = np.array(im, dtype=np.int64)
    Vp = np.sum(im == 1, dtype=np.int64)
    Vs = np.sum(im == 0, dtype=np.int64)
    e = Vp / (Vs + Vp)
    return e

def _radial_profile(autocorr, bins, pf=None, voxel_size=1):
    if len(autocorr.shape) == 2:
        adj = np.reshape(autocorr.shape, [2, 1, 1])
        inds = np.indices(autocorr.shape) - np.round(adj / 2)
        dt = np.sqrt(inds[0]**2 + inds[1]**2)
    elif len(autocorr.shape) == 3:
        adj = np.reshape(autocorr.shape, [3, 1, 1, 1])
        inds = np.indices(autocorr.shape) - np.round(adj / 2)
        dt = np.sqrt(inds[0]**2 + inds[1]**2 + inds[2]**2)
    else:
        raise Exception('Image dimensions must be 2 or 3')

    if np.max(bins) > np.max(dt):
        raise Exception('Bins specified distances exceeding maximum radial distance for image size.')

    bin_size = bins[1:] - bins[:-1]
    radial_sum = _get_radial_sum(dt, bins, bin_size, autocorr)
    
    # Normalize the radial sum to get probabilities
    norm_autoc_radial = radial_sum / np.max(autocorr)
    
    # Ensure values are within [0, 1] range
    norm_autoc_radial = np.clip(norm_autoc_radial, 0, 1)
    
    h = [norm_autoc_radial, bins]
    h = _parse_histogram(h, voxel_size=1)
    tpcf = Results()
    tpcf.distance = h.bin_centers * voxel_size
    tpcf.bin_centers = h.bin_centers * voxel_size
    tpcf.bin_edges = h.bin_edges * voxel_size
    tpcf.bin_widths = h.bin_widths * voxel_size
    tpcf.probability = norm_autoc_radial
    tpcf.probability_scaled = norm_autoc_radial * pf
    tpcf.pdf = h.pdf * pf
    tpcf.relfreq = h.relfreq
    return tpcf


def _get_radial_sum(dt, bins, bin_size, autocorr):
    radial_sum = np.zeros_like(bins[:-1], dtype=np.float64)  # Use float64 for large sums
    for i, r in enumerate(bins[:-1]):
        mask = (dt <= r) & (dt > (r - bin_size[i]))  # Logical AND for boolean mask
        if np.any(mask):  # Ensure there are non-zero elements in the mask
            radial_sum[i] = np.sum(autocorr[mask], dtype=np.float64) / np.sum(mask, dtype=np.float64)
    return radial_sum


def two_point_correlation(im, voxel_size=1, bins=100):
    cpus = 1
    pf = porosity(im)
    if isinstance(bins, int):
        r_max = (np.ceil(np.min(np.shape(im))) / 2).astype(int)
        bin_size = int(np.ceil(r_max / bins))
        bins = np.arange(0, r_max + bin_size, bin_size)

    with sp_ft.set_workers(cpus):
        F = sp_ft.ifftshift(sp_ft.rfftn(sp_ft.fftshift(im)))
        P = np.absolute(F**2)
        autoc = np.absolute(sp_ft.ifftshift(sp_ft.irfftn(sp_ft.fftshift(P))))

    tpcf = _radial_profile(autoc, bins, pf=pf, voxel_size=voxel_size)
    return tpcf


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



def lineal_path_distribution(im, bins=10, voxel_size=1, log=False):
    r"""
    Determines the probability that a point lies within a certain distance
    of the opposite phase *along a specified direction*

    This relates directly the radial density function defined by Torquato [1],
    but instead of reporting the probability of lying within a stated distance
    to the nearest solid in any direciton, it considers only linear distances
    along orthogonal directions.The benefit of this is that anisotropy can be
    detected in materials by performing the analysis in multiple orthogonal
    directions.

    Parameters
    ----------
    im : ndarray
        An image with each voxel containing the distance to the nearest solid
        along a linear path, as produced by ``distance_transform_lin``.
    bins : int or array_like
        The number of bins or a list of specific bins to use
    voxel_size : scalar
        The side length of a voxel.  This is used to scale the chord lengths
        into real units.  Note this is applied *after* the binning, so
        ``bins``, if supplied, should be in terms of voxels, not length units.
    log : boolean
        If ``True`` (default) the size data is converted to log (base-10)
        values before processing.  This can help to plot wide size
        distributions or to better visualize data in the small size region.
        Note that you should not anti-log the radii values in the retunred
        ``results``, since the binning is performed on the logged radii values.

    Returns
    -------
    result : Results object
        A custom object with the following data added as named attributes:

        *L* or *LogL*
            Length, equivalent to ``bin_centers``
        *pdf*
            Probability density function
        *cdf*
            Cumulative density function
        *relfreq*
            Relative frequency chords in each bin.  The sum of all bin
            heights is 1.0.  For the cumulative relativce, use *cdf* which is
            already normalized to 1.
        *bin_centers*
            The center point of each bin
        *bin_edges*
            Locations of bin divisions, including 1 more value than
            the number of bins
        *bin_widths*
            Useful for passing to the ``width`` argument of
            ``matplotlib.pyplot.bar``

    References
    ----------
    [1] Torquato, S. Random Heterogeneous Materials: Mircostructure and
    Macroscopic Properties. Springer, New York (2002)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/linearl_path_distribution.html>`_
    to view online example.

    """
        
    # Calculate the lineal path function for phase 255, length 2, horizontal direction
    lengths = range(1, 50)
    lpf_values = [lineal_path_function_vectorized(im, l=l, direction='horizontal') for l in lengths]

    # plt.close()
    # plt.plot(lengths, lpf_values, marker='o')
    # plt.xlabel('Segment Length (l)')
    # plt.ylabel('LPF')
    # plt.title('Lineal Path Function (Horizontal, Phase 255)')
    # plt.grid(True)
    # plt.savefig("z/lpf_horizontal.png")
    # plt.close()

    x = lpf_values
    if log:
        x = np.log10(x)
    h = list(np.histogram(x, bins=10, density=True))
    h = _parse_histogram(h=h, voxel_size=voxel_size)
    cld = Results()
    cld[f"{log*'Log' + 'L'}"] = h.bin_centers
    cld.pdf = h.pdf
    cld.cdf = h.cdf
    cld.relfreq = h.relfreq
    cld.bin_centers = h.bin_centers
    cld.bin_edges = h.bin_edges
    cld.bin_widths = h.bin_widths
    return cld


def lineal_path_function_vectorized(binary_image, l, direction='horizontal'):
    """
    Calculate the Lineal Path Function (LPF) for a given phase in a binary image.
    
    Parameters:
    binary_image : numpy.ndarray
        2D array representing the binary image (with values 0 and 255).
    l : int
        Length of the line segment for which to calculate the LPF.
    phase_value : int, optional
        The value representing the phase of interest (default is 255).
    direction : str, optional
        Direction in which to calculate the LPF ('horizontal' or 'vertical').

    Returns:
    float
        The Lineal Path Function (LPF) value for the given segment length l.
    """
    
    if direction == 'horizontal':
        im = binary_image
    elif direction == 'vertical':
        im = binary_image.T  # Transpose for vertical paths
    else:
        raise ValueError("Unsupported direction: Choose 'horizontal' or 'vertical'")
    
    # Create a sliding window view of the array where each window is of length l
    # This will create an array of shape (n_rows, n_cols-l+1, l)
    windowed_data = np.lib.stride_tricks.sliding_window_view(im, l, axis=1)
    
    # Check if all elements in each window are True (i.e., entirely within the phase)
    all_within_phase = np.all(windowed_data, axis=-1)
    
    # Calculate the LPF as the mean of this Boolean array
    lpf_value = np.mean(all_within_phase)
    
    return lpf_value


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

def chord_length_distribution(im, bins=10, log=False, voxel_size=1,
                              normalization='count'):
    r"""
    Determines the distribution of chord lengths in an image containing chords.

    Parameters
    ----------
    im : ndarray
        An image with chords drawn in the pore space, as produced by
        ``apply_chords`` or ``apply_chords_3d``.  ``im`` can be either boolean,
        in which case each chord will be identified using ``scipy.ndimage.label``,
        or numerical values in case it is assumed that chords have already been
        identifed and labeled. In both cases, the size of each chord will be
        computed as the number of voxels belonging to each labelled region.
    bins : scalar or array_like
        If a scalar is given it is interpreted as the number of bins to use,
        and if an array is given they are used as the bins directly.
    log : boolean
        If ``True`` (default) the size data is converted to log (base-10)
        values before processing.  This can help to plot wide size
        distributions or to better visualize the in the small size region.
        Note that you should not anti-log the radii values in the retunred
        ``tuple``, since the binning is performed on the logged radii values.
    normalization : string
        Indicates how to normalize the bin heights.  Options are:

        *'count' or 'number'*
            (default) This simply counts the number of chords in each bin in
            the normal sense of a histogram.  This is the rigorous definition
            according to Torquato [1].
        *'length'*
            This multiplies the number of chords in each bin by the
            chord length (i.e. bin size).  The normalization scheme accounts for
            the fact that long chords are less frequent than shorert chords,
            thus giving a more balanced distribution.

    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so the
        user can apply the scaling to the returned results after the fact.

    Returns
    -------
    result : Results object
        A custom object with the following data added as named attributes:

        *L* or *LogL*
            Chord length, equivalent to ``bin_centers``
        *pdf*
            Probability density function
        *cdf*
            Cumulative density function
        *relfreq*
            Relative frequency chords in each bin.  The sum of all bin
            heights is 1.0.  For the cumulative relativce, use *cdf* which is
            already normalized to 1.
        *bin_centers*
            The center point of each bin
        *bin_edges*
            Locations of bin divisions, including 1 more value than
            the number of bins
        *bin_widths*
            Useful for passing to the ``width`` argument of
            ``matplotlib.pyplot.bar``

    References
    ----------
    [1] Torquato, S. Random Heterogeneous Materials: Mircostructure and
    Macroscopic Properties. Springer, New York (2002) - See page 45 & 292

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/chord_length_distribution.html>`_
    to view online example.

    """
    x = calculate_chord_lengths(im)
    if bins is None:
        bins = np.array(range(0, x.max() + 2)) * voxel_size
    x = x * voxel_size
    if log:
        x = np.log10(x)
    if normalization == 'length':
        h = list(np.histogram(x, bins=bins, density=False))
        # Scale bin heigths by length
        h[0] = h[0] * (h[1][1:] + h[1][:-1]) / 2
        # Normalize h[0] manually
        h[0] = h[0] / h[0].sum(dtype=np.int64) / (h[1][1:] - h[1][:-1])
    elif normalization in ['number', 'count']:
        h = np.histogram(x, bins=bins, density=True)
    else:
        raise Exception('Unsupported normalization:', normalization)
    h = _parse_histogram(h)
    cld = Results()
    cld[f"{log*'Log' + 'L'}"] = h.bin_centers
    cld.pdf = h.pdf
    cld.cdf = h.cdf
    cld.relfreq = h.relfreq
    cld.bin_centers = h.bin_centers
    cld.bin_edges = h.bin_edges
    cld.bin_widths = h.bin_widths
    return cld

def calculate_chord_lengths(image, direction='horizontal'):
    if direction == 'horizontal':
        data = image
    elif direction == 'vertical':
        data = image.T
    else:
        raise ValueError("Unsupported direction: Choose 'horizontal' or 'vertical'")

    chord_lengths = []
    
    for row in data:
        padded = np.pad(row, (1, 1), mode='constant')
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == 255)[0]
        chord_lengths.extend(ends - starts)

    return chord_lengths
