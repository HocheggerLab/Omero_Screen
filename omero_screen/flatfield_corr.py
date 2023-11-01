#!/usr/bin/env python3
# omero_screen/flatfield_corr.py

"""Module generate flatfield masks for a given plate and upload them to the OMERO server.
The main function, fatfieldcorr, generates flatfield correction masks for each channel in the plate.
The masks are stored in the linked dataset in the Screens Project on the OMERO server.
Quality control examples are saved as pdf attachments to the dataset.
The flatfield corr function returns a dictionary with channel names and the corresponding flatfield correction masks.
"""


import os
import platform
import random
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import omero
from ezomero import get_image
from tqdm import tqdm

from omero_screen.aggregator import ImageAggregator
from omero_screen.general_functions import (
    scale_img,
    generate_image,
    omero_connect,
    add_map_annotation,
)
from omero_screen.metadata import MetaData, ProjectSetup

if platform.system() == "Darwin":
    matplotlib.use("MacOSX")  # avoid matplotlib warning about interactive backend


def flatfieldcorr(meta_data, project_data, conn) -> dict:
    """
    Fetches or generates flatfield correction masks for given metadata and connection.
    And saves them as an image in the dataset generated by the metadata class.
    Example images are added to the data set as pdf file attachments for quality control.

    Parameters:
    meta_data (MetaData): Metadata object that contains information about the plate, channels, image and dataset.
    conn (BlitzGateway): Connection object to an OMERO server.

    Returns:
    dict: Dictionary containing flatfield correction masks.
    """

    plate = conn.getObject("Plate", meta_data.plate_id)
    channels = meta_data.channels
    image_name = f"{meta_data.plate_id}_flatfield_masks"
    dataset_id = project_data.dataset_id
    dataset = conn.getObject("Dataset", dataset_id)  # Fetch the dataset
    image_dict = {}
    image_id = None
    # Loop over each image in the dataset to check if the required image is already present
    for image in dataset.listChildren():
        if image.getName() == image_name:
            image_id = image.getId()
            print(f"Image {image_name} already exists in the dataset.")
            image_dict = load_image_to_dict(conn, image_id)
            break  # stop the loop once the image is found
    # If the image is not already present, generate it
    if image_id is None:
        image_dict = generate_corr_dict(plate, channels, conn, dataset_id)
        upload_images(conn, dataset, image_name, image_dict)
        print(f"Uploaded {image_name} to dataset {dataset.getName()}")
    # If no flatfield correction masks found, raise an error
    if len(image_dict) == 0:
        raise ValueError("No flatfield correction masks found")
    else:
        print("Flatfield correction masks successfully loaded")
    return image_dict


def upload_images(conn, dataset, image_name, image_dict):
    """
    Uploads generated images to OMERO server.

    Parameters:
    conn (BlitzGateway): Connection object to an OMERO server.
    dataset (omero.gateway.DatasetWrapper): DatasetWrapper object representing the dataset to which the images are to be uploaded.
    image_name (str): Name to be given to the uploaded image.
    image_dict (dict): Dictionary containing channel names and corresponding image arrays.

    Returns:
    None. The image is saved to the OMERO server and linked to the specified dataset.
    """
    array_list = []
    channel_names = list(image_dict.keys())  # get channel names
    channel_number = len(channel_names)  # get number of channels
    for array in image_dict.values():
        # Wrap the array in a generato
        array_list.append(array)

    def plane_gen():
        """Generator that yields each plane in the array_list"""
        yield from array_list

    # Create the image in the dataset
    image = conn.createImageFromNumpySeq(
        plane_gen(), image_name, 1, channel_number, 1, dataset=dataset
    )
    print(f"Created image with ID: {image.getId()}")
    # Create a dictionary of channel names
    channel_dict = [[f"channel_{i}", name] for i, name in enumerate(channel_names)]
    add_map_annotation(image, channel_dict, conn=conn)


def load_image_to_dict(conn, image_id):
    """
    Loads an image from the OMERO server and converts it to a dictionary with channel names as keys and corresponding image arrays as values.

    Parameters:
    conn (BlitzGateway): Connection object to an OMERO server.
    image_id (int): ID of the image to be loaded from the OMERO server.

    Returns:
    dict: Dictionary containing channel names and corresponding image arrays.
    """
    # Fetch the image
    image = conn.getObject("Image", image_id)
    if not image:
        raise ValueError(f"No image found with ID: {image_id}")

    # Retrieve the pixels for the image
    pixels = image.getPrimaryPixels()

    # Initialize the dictionary
    image_dict = {}

    # Fetch annotations attached to the image
    annotations = image.listAnnotations()

    # Filter for MapAnnotations only
    map_anns = [
        ann
        for ann in annotations
        if isinstance(ann, omero.gateway.MapAnnotationWrapper)
    ]

    # Extract the channel names from the annotations
    for ann in map_anns:
        kv_pairs = ann.getValue()
        for kv in kv_pairs:
            key, value = kv
            if key.startswith("channel"):
                channel_num = int(
                    key.split("_")[-1]
                )  # Assumes channel keys are in the format 'Channel_X'
                # Retrieve the plane corresponding to the current channel
                plane = pixels.getPlane(0, channel_num, 0)  # Assumes Z=0, T=0
                # Add to the dictionary
                image_dict[value] = plane

    return image_dict


def generate_corr_dict(plate, channels, conn, dataset_id):
    """
    Generates a dictionary of flatfield correction masks for each channel in the plate.

    Parameters:
    plate (omero.gateway.PlateWrapper): PlateWrapper object representing the plate.
    channels (dict): Dictionary containing channel names and IDs.
    conn (BlitzGateway): Connection object to an OMERO server.
    dataset_id (int): ID of the dataset on the OMERO server to which the image should be attached.

    Returns:
    dict: Dictionary containing channel names and corresponding flatfield correction masks.
    """
    print(f"\nAssembling Flatfield Correction Masks for {len(channels)} Channels\n")
    corr_dict = {}
    img_list = random_imgs(plate)
    for channel in list(channels.items()):
        norm_mask = aggregate_imgs(img_list, channel, conn)
        example = gen_example(img_list, channel, norm_mask, conn)
        example_fig(conn, example, channel, dataset_id)
        corr_dict[channel[0]] = norm_mask  # associates channel name with flatfield mask
    return {
        k: corr_dict[k]
        for k, v in sorted(channels.items(), key=lambda item: item[1])
    }


def random_imgs(plate):
    """
    Selects 100 random images (or less) across all wells in a given plate.

    Parameters:
    plate (omero.gateway.PlateWrapper): PlateWrapper object representing the plate.

    Returns:
    list: List of random image IDs from each well in the plate.
    """
    # Get all the wells associated with the plate
    wells = plate.listChildren()
    img_list = []
    for well in wells:
        index = well.countWellSample()
        img_list.extend(well.getImage(index).getId() for index in range(index))
    return img_list if len(img_list) <= 100 else random.sample(img_list, 100)


def aggregate_imgs(img_list, channel, conn):
    """
    Aggregates images in a well for a specified channel and generates correction mask using the Aggregator Module.
    Zstack images are collapsed to a single image by taking the maximum intensity at each pixel location.
    For time lapse images we select 1 maximum of 10 randomly selected timepoints.
    Parameters:
    img_list (list): List of image IDs to aggregate.
    channel (dict): Dictionary containing channel information.
    conn (BlitzGateway): Connection object to an OMERO server.

    Returns:
    ndarray: Flatfield correction mask for the given channel.
    """
    agg = ImageAggregator(60)
    for img_id in tqdm(img_list):
        image, image_array = get_image(conn, img_id)
        mip_array = np.max(image_array[..., channel[1]], axis=1, keepdims=True)
        for t_img in random_timgs(mip_array):
            agg.add_image(t_img.reshape(1080, 1080))
    blurred_agg_img = agg.get_gaussian_image(30)
    return blurred_agg_img / blurred_agg_img.mean()

def random_timgs(image_array):
    individual_images = []
    num_images_to_select = min(10, image_array.shape[0])  # Ensure we don't select more than available
    selected_indices = random.sample(range(image_array.shape[0]), num_images_to_select)
    individual_images.extend(
        image_array[index : index + 1] for index in selected_indices
    )
    return individual_images


def gen_example(img_list, channel, mask, conn):
    """
    Generates an example image that applies the flatfield correction mask.

    Parameters:
    img_list (list): List of image IDs to use for generating the example.
    channel (dict): Dictionary containing channel information.
    mask (ndarray): Flatfield correction mask to apply to the image.
    conn (BlitzGateway): Connection object to an OMERO server.

    Returns:
    list: List of tuples, each containing an image or image data and a title for that image or data.
    """
    random_id = random.choice(img_list)
    image = conn.getObject("Image", random_id)
    example_img = generate_image(image, channel[1])
    scaled = scale_img(example_img)
    corr_img = example_img / mask
    bgcorr_img = corr_img - np.percentile(corr_img, 0.2) + 1
    corr_scaled = scale_img(bgcorr_img)
    # order all images for plotting
    return [
        (scaled, "original image"),
        (np.diagonal(example_img), "diag. intensities"),
        (corr_scaled, "corrected image"),
        (np.diagonal(corr_img), "diag. intensities"),
        (mask, "flatfield correction mask"),
    ]


def example_fig(conn, data_list, channel, dataset_id):
    """
    Creates a figure from the given data list and uploads it to the OMERO server.

    Parameters:
    conn (BlitzGateway): Connection object to an OMERO server.
    data_list (list): List of tuples, each containing an image or image data and a title for that image or data.
    channel (dict): Dictionary containing channel information.
    dataset_id (int): ID of the dataset on the OMERO server to which the figure should be attached.

    Returns:
    None. The figure is saved to the OMERO server and linked to the specified dataset.
    """
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    for i, data_tuple in enumerate(data_list):
        plt.sca(ax[i])
        if i in [0, 2, 4]:
            plt.imshow(data_tuple[0], cmap="gray")
        else:
            plt.plot(data_tuple[0])
            plt.ylim(0, 10 * data_tuple[0].min())
        plt.title(data_tuple[1])
    # save and close figure
    fig_id = f"{channel[0]}_flatfield_check.pdf"  # using channel name
    fig.tight_layout()

    # Save the figure to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    fig.savefig(temp_file.name, format="pdf")
    plt.close(fig)

    # Assuming 'conn' is a BlitzGateway connection
    dataset = conn.getObject("Dataset", dataset_id)

    # Upload file to server
    with open(temp_file.name, "rb") as f:
        size = os.path.getsize(temp_file.name)
        original_file = conn.createOriginalFileFromFileObj(
            f, fileSize=size, path="/", name=fig_id, mimetype="image/pdf"
        )

    # Attach OriginalFile to Dataset as FileAnnotation
    file_ann = omero.model.FileAnnotationI()
    file_ann.setFile(original_file._obj)
    file_ann = conn.getUpdateService().saveAndReturnObject(file_ann)

    # Link FileAnnotation to Dataset
    link = omero.model.DatasetAnnotationLinkI()
    link.parent = omero.model.DatasetI(dataset.getId(), False)
    link.child = file_ann
    conn.getUpdateService().saveAndReturnObject(link)

    # Delete the temporary file
    os.unlink(temp_file.name)


if __name__ == "__main__":

    @omero_connect
    def flatfield_test(conn=None):
        project_data = ProjectSetup(3, conn)
        meta_data = MetaData(conn, 3)
        return flatfieldcorr(meta_data, project_data, conn)

    flatfield_corr = flatfield_test()
    print(flatfield_corr.keys())
