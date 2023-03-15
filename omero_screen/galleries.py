#%%
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
from skimage import exposure
#%%


def resize_tf( image):
    return tf.image.resize(image, size=(41, 41,), method=tf.image.ResizeMethod.BILINEAR)

def gallery_data(df,cell_cycle_detaild,check_phase,gallery_name,total,images_per_row):

    tem_cell_list=df[df[cell_cycle_detaild]==check_phase]['cell_data'].tolist()
    # convert the TensorFlow Tensor object to a NumPy array
    nor_list = [resize_tf(i).numpy().astype('float32') for i in tem_cell_list]
    print(len(nor_list))
    # select the samples
    sample = random.sample(nor_list,total)
    print(f'{check_phase} Total number: {len(nor_list)}, Random select: {len(sample)}')
    plot_digits(sample, images_per_row=images_per_row,phase=check_phase,plot_name=gallery_name)

def plot_digits(sample, images_per_row,phase,plot_name):
    # Get the shape of the individual images
    img_shape = sample[0].shape
    height, width = img_shape[:2]
    # Calculate the number of rows needed to display all the images
    n_rows = (len(sample) - 1) // images_per_row + 1
    # Calculate the number of empty image slots needed to complete the last row
    n_empty = n_rows * images_per_row - len(sample)
    # Create a batch of processed images to display
    images = []
    for image in sample:
        percentiles = np.percentile(image, (1, 100))
        # print(type(percentiles), percentiles)
        scaled=exposure.rescale_intensity(image, in_range=tuple(percentiles))
        # scaled[:,:,0]=np.zeros((41, 41))
        images.append(scaled)
    # Add empty images to the batch to complete the last row
    images.append(np.zeros((height, width * n_empty,3)))
    # Concatenate the images in each row
    row_images = []
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    # Concatenate the rows to create the final image
    image = np.concatenate(row_images, axis=0)
    # # Display the final image
    # Create a PDF file and add images to it
    with PdfPages(f'{plot_name}_Gallery_{phase}.pdf') as pdf:
        fig, axs = plt.subplots(1, 1)
        axs.imshow(image, cmap='gray')
        axs.set_title(f'{plot_name} {phase} Gallery')
        fig.suptitle(f'{plot_name} {phase} sample')
        pdf.savefig(fig)
        # plt.close()
