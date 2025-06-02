import numpy as np
from skimage.color import gray2rgb
from skimage.segmentation import slic


def grid_segmentation(instance, dims=(64, 64), grid_rows=8, grid_cols=8):
    h, w = dims

    assert (
        h % grid_rows == 0
    ), f"Image height {h} not divisible by grid_rows {grid_rows}"
    assert w % grid_cols == 0, f"Image width {w} not divisible by grid_cols {grid_cols}"

    cell_height = h // grid_rows
    cell_width = w // grid_cols
    seg_map = np.zeros((h, w), dtype=np.int32)

    label = 0
    for i in range(grid_rows):
        for j in range(grid_cols):
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width
            seg_map[y_start:y_end, x_start:x_end] = label
            label += 1

    return seg_map.flatten()


def slic_segmentation(instance, dims=(64, 64), n_segments=200, compactness=8, sigma=0):
    """
    Applies SLIC segmentation to the input image.

    Args:
        instance: The input image as a 3D numpy array (H, W, C).
        dims (tuple): The dimensions to resize/crop the image to (H, W).
        n_segments (int): Approximate number of superpixels.
        compactness (float): Balances color and spatial proximity.
        sigma (float): Gaussian smoothing applied before segmentation.

    Returns:
        Flattened segmentation map.
    """
    instance = instance.flatten()
    resized = gray2rgb(instance.reshape(*dims))
    seg_map = slic(
        resized,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
    )

    seg_map = seg_map.flatten()
    seg_map[instance == 0] = 0
    seg_map = seg_map.reshape(*dims)

    # Relabel segments for better visualization
    unique_segments = np.unique(seg_map)
    np.random.shuffle(unique_segments)
    seg_map_tmp = np.zeros_like(seg_map)
    for ii, seg in enumerate(unique_segments, start=1):
        seg_map_tmp[seg_map == seg] = ii
    seg_map = seg_map_tmp

    return seg_map.flatten()
