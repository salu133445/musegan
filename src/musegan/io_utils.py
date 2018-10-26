"""Utilities for creating image grids from a batch of images.
"""
import numpy as np
import pypianoroll

def vector_to_image(array, inverted=True):
    """
    Convert a batched vector array to an image array.

    Arguments
    ---------
    array : `np.array`, ndim=2
        The vector array.

    Returns
    -------
    image : `np.array`, ndim=4
        The image array.
    """
    if array.ndim != 2:
        raise ValueError("Input array must have 2 dimensions.")

    # Scale the array to [-1, 1] (assume that the values range in [-2, 2])
    array = .25 * (array + 2.)

    # Invert the color
    if inverted:
        array = 1. - array

    # Minus a small value to avoid casting 256 to 0
    quantized = (array * 256 - 1e-5).astype(np.uint8)

    # Reshape to an image array
    image = np.reshape(quantized, (-1, quantized.shape[1], 1, 1))

    return image

def pianoroll_to_image(pianoroll, colormap=None, inverted=True,
                       boundary_width=1, boundary_color=0, frame=False,
                       gamma=1.):
    """
    Convert a batched pianoroll array to an image array.

    Arguments
    ---------
    pianoroll : `np.array`, ndim=5
        The pianoroll array. The shape is (n_pianorolls, n_bars, n_timestep,
        n_pitches, n_tracks).
    boundary_width : int
        Linewidth of the boundary lines. Default to 0.
    boundary_color : int
        Grayscale of the boundary lines. Valid values are 0 (black) to 255
        (white). Default to 0.
    frame : bool
        Whether to use a grid frame. Default to False.

    Returns
    -------
    image : `np.array`, ndim=4
        The image array.
    """
    if pianoroll.ndim != 5:
        raise ValueError("Input pianoroll array must have 5 dimensions.")

    # Flip the pitch axis
    pianoroll = np.flip(pianoroll, 3)

    # Apply the color
    if colormap is not None:
        pianoroll = np.matmul(1. - colormap, np.expand_dims(pianoroll, -1))
        pianoroll = pianoroll.squeeze(-1).clip(0., 1.)

    # Apply gamma correction
    if gamma != 1.:
        pianoroll = pianoroll ** gamma

    # Invert the color
    if inverted:
        pianoroll = 1. - pianoroll

    # Quantize the image (minus a small value to avoid casting 256 to 0)
    quantized = (pianoroll * 256 - 1e-5).astype(np.uint8)

    # Add the boundary lines
    if boundary_width:
        quantized = np.pad(
            quantized,
            ((0, 0), (0, 0), (boundary_width, 0), (boundary_width, 0), (0, 0)),
            'constant', constant_values=boundary_color)

    # Transpose and reshape to get the image array
    if colormap is None:
        transposed = np.transpose(quantized, (0, 4, 3, 1, 2))
        image = np.reshape(
            transposed, (-1, transposed.shape[1] * transposed.shape[2],
                         transposed.shape[3] * transposed.shape[4], 1))
    else:
        transposed = np.transpose(quantized, (0, 3, 1, 2, 4))
        image = np.reshape(transposed, (
            -1, transposed.shape[1], transposed.shape[2] * transposed.shape[3],
            transposed.shape[4]))

    # Deal with the frame
    if boundary_width:
        if frame:
            image = np.pad(
                image,
                ((0, 0), (0, boundary_width), (0, boundary_width), (0, 0)),
                'constant', constant_values=boundary_color)
        else:
            image = image[:, boundary_width:, boundary_width:]

    return image

def image_pair(image1, image2, mode='side-by-side', boundary_width=1,
               boundary_color=0, frame=False):
    """
    Pair two image arrays to one single image array.

    Arguments
    ---------
    image1 : `np.array`, ndim=4
        The image array at the left in 'side-by-side' mode or at the top in
        'top-bottom' mode.
    image2 : `np.array`, ndim=4
        The image array at the right in 'side-by-side' mode or at the bottom in
        'top-bottom' mode.
    mode : {'side-by-side', 'top-bottom'}
        Mode to pack the two images.
    boundary_width : int
        Linewidth of the boundary lines. Default to 0.
    boundary_color : int
        Grayscale of the boundary lines. Valid values are 0 (black) to 255
        (white). Default to 0.
    frame : bool
        Whether to use a grid frame. Default to False.

    Returns
    -------
    image : `np.array`
        The image array.
    """
    if image1.ndim != image2.ndim:
        raise ValueError("Input image arrays must have the same number of "
                         "dimensions.")
    if not (mode == 'side-by-side' or mode == 'top-down'):
        raise ValueError("Invalid mode received. Valid modes are "
                         "'side-by-side' and 'top-down'.")

    if mode == 'side-by-side':
        if boundary_width:
            # Add the boundary line
            image1 = np.pad(
                image1, ((0, 0), (0, 0), (0, boundary_width), (0, 0)),
                'constant', constant_values=boundary_color)
        image = np.concatenate((image1, image2), 2)

    elif mode == 'top-down':
        if boundary_width:
            # Add the boundary line
            image1 = np.pad(
                image1, ((0, 0), (0, boundary_width), (0, 0), (0, 0)),
                'constant', constant_values=boundary_color)
        image = np.concatenate((image1, image2), 1)

    if frame:
        image = np.pad(image, ((0, 0), (boundary_width, boundary_width),
                               (boundary_width, boundary_width), (0, 0)),
                       'constant', constant_values=boundary_color)

    return image

def image_grid(image, grid_shape, grid_width=3, grid_color=0, frame=True):
    """
    Convert a batched image array to one merged grid image array.

    Arguments
    ---------
    pianoroll : `np.array`, ndim=4
        The pianoroll array. The first axis is the batch axis. The second and
        third axes are the time and pitch axes, respectively, of the pianorolls.
        The last axis is the track axis.
    grid_shape : list or tuple of int
        Shape of the image grid (height, width).
    grid_width : int
        Linewidth of the grid. Default to 0.
    grid_color : int
        Grayscale of the grid. Valid values are 0 (black) to 255 (white).
        Default to 0.
    frame : bool
        Whether to use a grid frame. Default to False.

    Returns
    -------
    merged : `np.array`, ndim=3
        The merged image grid array.
    """
    if len(grid_shape) != 2:
        raise ValueError("`grid_shape` must be a list or tuple of two "
                         "integers.")
    if image.ndim != 4:
        raise ValueError("Input image array must have 4 dimensions.")

    # Slice the array to get the right number of images
    sliced = image[:(grid_shape[0] * grid_shape[1])]

    # Add the grid lines
    if grid_width:
        sliced = np.pad(
            sliced, ((0, 0), (grid_width, 0), (grid_width, 0), (0, 0)),
            'constant', constant_values=grid_color)

    # Reshape to split the first (batch) axis into two axes
    reshaped = np.reshape(sliced, ((grid_shape[0], grid_shape[1])
                                   + sliced.shape[1:]))

    # Transpose and reshape to get the image grid
    transposed = np.transpose(reshaped, (0, 2, 1, 3, 4))
    grid = np.reshape(
        transposed, (grid_shape[0] * transposed.shape[1],
                     grid_shape[1] * transposed.shape[3], image.shape[-1]))

    # Deal with the frame
    if grid_width:
        if frame:
            grid = np.pad(grid, ((0, grid_width), (0, grid_width), (0, 0)),
                          'constant', constant_values=grid_color)
        else:
            grid = grid[:, grid_width:, grid_width:]

    return grid

def save_pianoroll(filename, pianoroll, programs, is_drums, tempo,
                   beat_resolution, lowest_pitch):
    """Saves a batched pianoroll array to a npz file."""
    if not np.issubdtype(pianoroll.dtype, np.bool_):
        raise TypeError("Input pianoroll array must have a boolean dtype.")
    if pianoroll.ndim != 5:
        raise ValueError("Input pianoroll array must have 5 dimensions.")
    if pianoroll.shape[-1] != len(programs):
        raise ValueError("Length of `programs` does not match the number of "
                         "tracks for the input array.")
    if pianoroll.shape[-1] != len(is_drums):
        raise ValueError("Length of `is_drums` does not match the number of "
                         "tracks for the input array.")

    reshaped = pianoroll.reshape(
        -1, pianoroll.shape[1] * pianoroll.shape[2], pianoroll.shape[3],
        pianoroll.shape[4])

    # Pad to the correct pitch range and add silence between phrases
    to_pad_pitch_high = 128 - lowest_pitch - pianoroll.shape[3]
    padded = np.pad(
        reshaped, ((0, 0), (0, pianoroll.shape[2]),
                   (lowest_pitch, to_pad_pitch_high), (0, 0)), 'constant')

    # Reshape the batched pianoroll array to a single pianoroll array
    pianoroll_ = padded.reshape(-1, padded.shape[2], padded.shape[3])

    # Create the tracks
    tracks = []
    for idx in range(pianoroll_.shape[2]):
        tracks.append(pypianoroll.Track(
            pianoroll_[..., idx], programs[idx], is_drums[idx]))

    # Create and save the multitrack
    multitrack = pypianoroll.Multitrack(
        tracks=tracks, tempo=tempo, beat_resolution=beat_resolution)
    multitrack.save(filename)
