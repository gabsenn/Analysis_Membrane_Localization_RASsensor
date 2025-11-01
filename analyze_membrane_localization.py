import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from iaf.morph.watershed import (
    estimate_object_sizes,
    filter_labels_by_area,
    label_to_eroded_bw_mask,
    separate_neighboring_objects,
)
from iaf_apps.membranes import (
    assign_membranes_to_nuclei,
    discard_membranes_without_nucleus,
    discard_contacting_membranes,
    discard_incomplete_membranes,
)
from iaf.process import subtract_background
from scipy.ndimage import binary_fill_holes
from skimage.filters import gaussian, threshold_li, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk
from tifffile import imread, imwrite
from skimage.morphology import convex_hull_image
from multiprocessing import Pool
import os


def segment_nuclei(
    nuclei_image,
    sigma=1,
    background_radius=200,
    min_size=None,
    min_area=40,
    use_li=False,
    run_watershed=True
):
    nuclei_image = nuclei_image.astype(np.float32)
    nuclei_image = gaussian(nuclei_image, sigma=sigma)
    I3 = subtract_background(nuclei_image, radius=background_radius, down_size_factor=4)
    if use_li:
        threshold = threshold_li(I3)
    else:
        threshold = threshold_otsu(I3)
    bw = I3 > threshold
    labels, num = label(bw, background=0, return_num=True, connectivity=1)
    if run_watershed:
        if min_size is None:
            area, min_axis, max_axis, equiv_diam = estimate_object_sizes(bw)
            min_size = 0.5 * min_axis
        maxima_suppression_size = min_size
        labels_ws, num_ws, _ = separate_neighboring_objects(
            bw, labels, min_size=min_size, maxima_suppression_size=maxima_suppression_size
        )
        labels_ws = label_to_eroded_bw_mask(labels_ws, sel=disk(1))
    else:
        labels_ws = labels
    labels_ws_upd, _ = filter_labels_by_area(labels_ws, min_area=min_area)
    return labels_ws_upd


def segment_membrane(membrane_image, sigma=1, background_radius=200):
    membrane_image = membrane_image.astype(np.float32)
    membrane_image = gaussian(membrane_image, sigma=sigma)
    I = subtract_background(
        membrane_image, radius=background_radius, down_size_factor=4
    )
    threshold = threshold_li(I)
    bw = I > threshold
    return 255 * bw.astype(np.uint8)


def filter_incomplete_membranes(membrane_mask, min_area=50):

    # Make sure to have a mask?
    membrane_mask = (membrane_mask > 0).astype(np.uint8)

    # Perform closing operation
    kernel = np.ones((3, 3), np.uint8)
    closed_membrane_mask = cv2.morphologyEx(membrane_mask, cv2.MORPH_CLOSE, kernel)

    # Label connected components
    label_image = label(closed_membrane_mask)

    # Filter out objects based on their area
    filtered_mask = np.zeros_like(membrane_mask, dtype=np.uint8)
    discarded = 0
    for region in regionprops(label_image):
        if region.area >= min_area:
            coords = region.coords.T
            filtered_mask[coords[0], coords[1]] = 255
        else:
            discarded += 1
    print(f"Found {discarded} incomplete membranes to discard.")
    return filtered_mask


def prepare_signal(signal_image, background_radius=100):
    """Prepare the signal for quantification."""
    nuclei_image = signal_image.astype(np.float32)
    signal_image_sub = subtract_background(
        nuclei_image, radius=background_radius, down_size_factor=4
    )
    # Make sure to set negative values to 0
    signal_image_sub[signal_image_sub < 0] = 0
    return signal_image_sub


def remove_incomplete_membranes(membrane_labels):

    to_discard = []

    for prop in regionprops(membrane_labels):
        y0, x0, y, x = prop.bbox
        mask = membrane_labels[y0:y, x0:x] > 0
        filled_mask = binary_fill_holes(mask)
        if np.all(filled_mask == mask):
            # The binary fill holes operation failed. The membrane
            # is not closed. We enqueue if for discarding.
            to_discard.append(prop.label)

    print(f"Found {len(to_discard)} incomplete membranes to discard.")

    # Remove regions in the queue from the labeled membrane image
    for label in to_discard:
        membrane_labels[membrane_labels == label] = 0

    return membrane_labels


def quantify_signal(membrane_labels, signal, masks_folder=None):

    # Delete all preexisting QC figures, if needed
    if masks_folder is not None:
        for f in Path(masks_folder).glob("mask_*.tif"):
            f.unlink()

    # Keep track of the number of discarded cells
    discarded = 0

    cell_id = []
    membrane_signal = []
    total_signal = []
    ratio = []

    # Prepare output mask
    output_membrane_labels = np.zeros(membrane_labels.shape, dtype=membrane_labels.dtype)

    for i, prop in enumerate(regionprops(membrane_labels)):
        y0, x0, y, x = prop.bbox
        region_signal = signal[y0:y, x0:x]
        mask = membrane_labels[y0:y, x0:x] > 0
        filled_mask = binary_fill_holes(mask)
        nuclei_mask = np.logical_xor(mask, filled_mask)

        # Run some quality controls on the mask:
        #   * if the filled_mask has not content, it means that the membrane was not closed;
        #     we discard the cell
        #   * if the filled_mask has more than one connected component, it *may* indicate that the
        #     membrane contained more than one nucleus. This means that the membrane assignment did
        #     not work properly; we discard the cell.

        # Empty filled mask?
        if filled_mask.sum() == 0:
            discarded += 1
            continue

        # If the filled mask is the same as the mask, we also discard
        if np.all(mask == filled_mask):
            discarded += 1
            continue

        # Calculate the convex hull of the image
        mask_hull = convex_hull_image(mask)

        # If the ratio of the area of the filled image to the area of the convex hull
        # is lower than 0.75, we drop the mask (it probably has one or more large holes
        # inside that touch the border
        if np.sum(filled_mask) / np.sum(mask_hull) < 0.75:
            discarded += 1
            continue

        # More than one "nucleus"?
        min_accepted_area = 25
        cell_props = regionprops(label(nuclei_mask))
        n_valid = 0
        for cell_prop in cell_props:
            if cell_prop.area > min_accepted_area:
                n_valid += 1
        if n_valid != 1:
            discarded += 1
            continue

        # If the area of the membrane is larger than 0.25 times the
        # area of the whole cell, we discard it
        if 1 - (np.sum(nuclei_mask) / np.sum(filled_mask)) >= 0.75:
            discarded += 1
            continue

        # Save QC mask
        if masks_folder is not None:
            qc_mask = np.zeros((3, mask.shape[0], mask.shape[1]), dtype=np.uint8)
            qc_mask[0] = 255 * mask.astype(np.uint8)
            qc_mask[1] = 255 * filled_mask.astype(np.uint8)
            qc_mask[2] = 255 * nuclei_mask.astype(np.uint8)
            qc_mask_fname = f"mask_{i:04}.tif"
            # plt.imshow(np.swapaxes(qc_mask, 0, 2))
            # plt.show()
            imwrite(masks_folder / qc_mask_fname, qc_mask)

        # Calculate ratio and append
        cell_id.append(prop.label)
        m = np.sum(region_signal[mask])
        t = np.sum(region_signal[filled_mask])
        r = m / t
        membrane_signal.append(m)
        total_signal.append(t)
        ratio.append(r)

        # Add the membrane label to the output
        output_membrane_labels[y0:y, x0:x] = membrane_labels[y0:y, x0:x]

    # Inform
    print(f"Processed {len(regionprops(membrane_labels))} ({discarded} discarded).")

    # Create and return dataframe
    df = pd.DataFrame(columns=["cell_id", "membrane_signal", "total_signal", "ratio"])
    df["cell_id"] = cell_id
    df["membrane_signal"] = membrane_signal
    df["total_signal"] = total_signal
    df["ratio"] = ratio
    return df, output_membrane_labels


def create_membrane_nuclei_rgb_image(membrane_label, nuclei_label):
    """Create a new RGB image with all membranes and their corresponding nuclei.

    Parameters
    ----------

    membrane_label: np.ndarray
        Membrane label image.

    nuclei_label: np.ndarray
        Nuclei label image.

    Returns
    -------

    updated_membrane_mask: RGB image (np.ndarray, z-by-y-by-x)
        RGB images where membranes are shown in RED and their nuclei in green.
        All nuclei not contained in the membranes are dropped..
    """

    # Create an RGB image
    rgb_img = np.zeros((3, membrane_label.shape[0], membrane_label.shape[1]), dtype=np.uint8)

    for region in regionprops(membrane_label):
        # Compute the bounding box of the current region
        minr, minc, maxr, maxc = region.bbox

        # Check if any nucleus pixel is within the bounding box
        if np.any(nuclei_label[minr:maxr, minc:maxc]):
            # Add the region to the filtered mask
            rgb_img[0, region.coords.T[0], region.coords.T[1]] = 255
            n = 255 * (nuclei_label[minr:maxr, minc:maxc] > 0).astype(np.uint8)
            rgb_img[1, minr:maxr, minc:maxc] = n

    return rgb_img

def process_folder(folder_name):
    """This is the main processing function to be run in parallel."""

    # Extract the file filenames
    nuclei_image_name = None
    membrane_image_name = None
    signal_image_name = None
    for f in folder_name.glob("*.tif"):
        if f.name.endswith("_ch00.tif"):
            nuclei_image_name = f
        elif f.name.endswith("_ch03.tif"):
            membrane_image_name = f
        elif f.name.endswith("_ch01.tif"):
            signal_image_name = f
        else:
            continue

    # Check that all needed files are present
    if nuclei_image_name is None:
        print(f"{folder_name.name}: nuclei image file not found.")
        return False
    if membrane_image_name is None:
        print(f"{folder_name.name}: membrane image file not found.")
        return False
    if signal_image_name is None:
        print(f"{folder_name.name}: signal image file not found.")
        return False

    # Read all relevant images
    nuclei_img = imread(nuclei_image_name)
    membrane_img = imread(membrane_image_name)
    signal_img = imread(signal_image_name)

    # Check that the output folder exists
    out_folder = folder_name / "results"
    out_folder.mkdir(exist_ok=True)

    # Masks folder
    masks_folder = out_folder / "masks"
    masks_folder.mkdir(exist_ok=True)

    # QC folder
    qc_folder = out_folder / "qc"
    qc_folder.mkdir(exist_ok=True)

    # Segment the nuclei
    nuclei_label = segment_nuclei(nuclei_img)
    imwrite(str(qc_folder / "nuclei_label.tif"), nuclei_label.astype(np.int32))

    # Segment the membranes
    membrane_mask = segment_membrane(membrane_img)
    imwrite(str(qc_folder / "membrane_mask.tif"), membrane_mask)

    # Process the membranes one first time
    incomplete_filtered_mask = filter_incomplete_membranes(membrane_mask, min_area=50)
    filtered_membrane_mask = discard_membranes_without_nucleus(
        incomplete_filtered_mask, nuclei_label
    )
    labeled_membrane_mask = assign_membranes_to_nuclei(
        nuclei_label, filtered_membrane_mask
    )

    # Now remove all membranes that are significantly contacting other membranes.
    rcm_str = ""
    if REMOVE_CONTACTING_MEMBRANES:
        labeled_membrane_mask = discard_contacting_membranes(labeled_membrane_mask, threshold=rcm_threshold)
        rcm_str = f"_rcm_{rcm_threshold}_"

    # Clean broken membranes
    rim_str = ""
    if REMOVE_INCOMPLETE_MEMBRANES:
        labeled_membrane_mask = discard_incomplete_membranes(labeled_membrane_mask)
        rim_str = "_rim_"

    imwrite(str(qc_folder / f"labeled_membrane_mask{rcm_str}{rim_str}.tif"), labeled_membrane_mask)

    # Prepare the signal
    signal = prepare_signal(signal_img)

    # Analysis
    df, final_membrane_label = quantify_signal(labeled_membrane_mask, signal, masks_folder)

    # Save dataframe
    df.to_csv(str(out_folder / "results.csv"), index=False)

    # Create a final qc image
    rgb = create_membrane_nuclei_rgb_image(final_membrane_label, nuclei_label)

    # Save final membrane label image
    imwrite(str(out_folder / f"final_cell_selection.tif"), rgb)

    return True


if __name__ == "__main__":

    if os.name == "nt":
        ROOT_FOLDER = Path(
            "N:/04_Image_Analysis/Projects/external_data/2022/12/P_0022/E.GS3.66.1_Quantification of Membrane Localization/to_process/"
        )
    else:
        ROOT_FOLDER = Path(
            "/N/04_Image_Analysis/Projects/external_data/2022/12/P_0022/E.GS3.66.1_Quantification of Membrane Localization/to_process/"
        )

    # Get all subfolders to analyze
    folder_names = ROOT_FOLDER.glob("*/")

    # Remove files
    folder_names = [x for x in folder_names if x.is_dir()]

    # Toggle filters
    REMOVE_CONTACTING_MEMBRANES = True
    rcm_threshold = 50
    REMOVE_INCOMPLETE_MEMBRANES = True

    with Pool(os.cpu_count() - 1) as p:
        p.map(process_folder, folder_names)

    # inform
    print("All done!")
