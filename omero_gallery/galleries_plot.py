#%%
import matplotlib.pyplot as plt
import random
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

def plot_gallery(image_list:list[np.ndarray],check_phase,channels_option,nrows,gallery_name,path):
    # convert the TensorFlow Tensor object to a NumPy array
    nor_list = channel_img_list(cell_data_list=image_list,channel=channels_option)
    plot_digits(nor_list, images_per_row=nrows,phase=check_phase,plot_name=gallery_name,save_path=path)


def plot_digits(sample, images_per_row, phase, plot_name, save_path):
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

    # Create the final grid of images
    n_cols = images_per_row
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))

    # Iterate through the axes and add images
    for idx, ax in enumerate(axs.flat):
        ax.imshow(images[idx])
        ax.set_xticks([])
        ax.set_yticks([])
        # if idx < n_images:
        #     ax.axis('off')
    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.03, wspace=0.03)
    # Save the final image to a PDF
    with PdfPages(f"{save_path}/{plot_name}_{phase}_Gallery.pdf") as pdf:
        plt.suptitle(f'{plot_name} {phase} Gallery',fontsize=20)
        pdf.savefig(fig)
        plt.close()


if __name__=='__main__':
    import pandas as pd

    df = pd.read_pickle(
        str('/Users/haoranyue/Desktop/OmeroScreen_test/cellcycle_summary/OmeroScreen_test_singlecell_cellcycle_detailed_imagedata'))
    df_M = df[df['inter_M'] == 'M']
    df_M_mislabled = df_M[df_M['cell_cycle'] != 'M']
    data_list = [resize_tf(i).numpy().astype('float32') for i in df_M_mislabled['cell_data'].tolist()]
    # %%
    plot_digits(data_list[:400],20, phase="mislabled_M", plot_name='check', save_path='/Users/haoranyue/Desktop/OmeroScreen_test/')
