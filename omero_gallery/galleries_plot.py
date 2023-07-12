#%%
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
from skimage import exposure
#%%


def resize_tf(image: tf.Tensor) -> tf.Tensor:
    return tf.image.resize(image, size=(41, 41,), method=tf.image.ResizeMethod.BILINEAR)


def fet_channel_indices() -> dict:
    return {'all':None, 'tubulins':1, 'dapi': 2}

def channel_img_list(cell_data_list: list[tf.Tensor], channel : str) -> list[np.ndarray]:
    channel_indices =fet_channel_indices()
    channel_idx=channel_indices[channel.lower()]
    if channel_idx is None:
        data_list=[resize_tf(i).numpy().astype('float32') for i in cell_data_list]
    else:
        data_list = [resize_tf(i[:, :, channel_idx]).numpy().astype('float32') for i in cell_data_list]
    return data_list

def plot_gallery(image_list:list[np.ndarray],check_phase,channels_option,nrows,):
    # convert the TensorFlow Tensor object to a NumPy array
    nor_list = channel_img_list(cell_data_list=image_list,channel=channels_option)
    images=plot_digits(nor_list, images_per_row=nrows,phase=check_phase)
    return images


def add_border(image, border_size_ratio, border_color):
    # Calculate the border size in pixels
    border_size = int(border_size_ratio * min(image.shape[0], image.shape[1]))

    # Add a border of the specified size and color around the image.
    return cv2.copyMakeBorder(
        image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=border_color
    )


def merge_images(images, images_per_row, border_size_ratio, border_color):
    # Assumes images is a list of (height, width, channels) images
    assert len(images) % images_per_row == 0
    images_per_col = len(images) // images_per_row

    # Add border to each image
    images_with_borders = [add_border(img, border_size_ratio, border_color) for img in images]

    # Reshape the images into a grid
    image_grid = [np.concatenate(images_with_borders[i:i + images_per_row], axis=1)  # concatenate images horizontally
                  for i in range(0, len(images_with_borders), images_per_row)]  # for every row of images

    # Concatenate the rows vertically
    merged_image = np.concatenate(image_grid, axis=0)  # concatenate rows vertically

    return merged_image


def plot_digits(sample, images_per_row, phase,):
    # Get the shape of the individual images
    img_shape = sample[0].shape
    height, width = img_shape[:2]
    n_images = len(sample)
    n_rows = (n_images - 1) // images_per_row + 1
    n_empty = n_rows * images_per_row - n_images

    # Process the images
    images = []
    for image in sample:
        percentiles = np.percentile(image, (1, 100))
        scaled = exposure.rescale_intensity(image, in_range=tuple(percentiles))
        images.append(scaled)

    # Add empty images to fill the last row
    images.extend([np.zeros((height, width, img_shape[2]))] * n_empty)
    images = merge_images(images, images_per_row, border_size_ratio=0.03, border_color=(255, 0, 255))
    return images



if __name__=='__main__':
    import pandas as pd

    df = pd.read_pickle(
        str('/Users/haoranyue/Desktop/OmeroScreen_test/cellcycle_summary/OmeroScreen_test_singlecell_cellcycle_detailed_imagedata'))
    df_M = df[df['inter_M'] == 'M']
    df_M_mislabled = df_M[df_M['cell_cycle'] != 'M']
    data_list = [resize_tf(i).numpy().astype('float32') for i in df_M_mislabled['cell_data'].tolist()]
    # %%
    plot_digits(data_list[:4],2)
