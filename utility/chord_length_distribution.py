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
