


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
