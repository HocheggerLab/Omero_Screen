#%%
import matplotlib.pyplot as plt
import random
import numpy as np
from skimage import exposure
#%%

def gallery_data(df,pras,inter_phase,total,images_per_row):
    tem_df=df[pras]
    tem_cell_list=tem_df[tem_df[pras[1]]==inter_phase][pras[0]].tolist()
    nor_list=[np.array(i).astype('float32') for i in tem_cell_list if i.shape==(41,41,3)]
    print(len(nor_list))
    sample = random.sample(nor_list,total)
    print(len(sample))
    plot_digits_2(sample, images_per_row=images_per_row)

def plot_digits_2(sample, images_per_row=5, **options):
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
        print(image[:,:,0].max())
        percentiles = np.percentile(image, (1, 100))
        # print(type(percentiles), percentiles)
        scaled=exposure.rescale_intensity(image, in_range=tuple(percentiles))

        # scaled[:,:,0]=np.zeros((41, 41))
        images.append(scaled)
    # Add empty images to the batch to complete the last row
    images.append(np.zeros((height, width * n_empty)))
    # Concatenate the images in each row
    row_images = []
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    # Concatenate the rows to create the final image
    image = np.concatenate(row_images, axis=0)
    # # Display the final image
    plt.imshow(image, **options)
    plt.axis("off")
    plt.show()

