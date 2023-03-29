import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.general_functions import save_fig, generate_image, filter_segmentation, omero_connect, scale_img, \
    color_label
from skimage import measure, io

class Image_extract:
    """
    generates the corrected images Stores corrected images as dict,
    """

    def __init__(self, well, omero_image, meta_data, exp_paths, sample,flatfield_dict):
        self._well = well
        self.omero_image = omero_image
        self._meta_data = meta_data
        self._paths = exp_paths
        self._get_metadata()
        self.samples=sample
        self._flatfield_dict = flatfield_dict
        self.img_dict = self._get_img_dict()
        self.data=self._get_data(width=20)


    def _get_data(self, width=20):
        empty_channel = np.zeros((self.img_dict['DAPI'].shape[0], self.img_dict['DAPI'].shape[1]))
        comb_image = np.dstack([empty_channel, self.img_dict['Tub'], self.img_dict['DAPI']]).astype('float32')
        comb_image=(comb_image - comb_image.min()) / comb_image.max()-comb_image.min()
        data_list =[]
        for centroid in (self.samples):
            i = centroid[0]
            j = centroid[1]
            imin = int(round(max(0, i - width)))
            imax = int(round(min(self.img_dict['DAPI'].shape[0], i + width + 1)))
            jmin = int(round(max(0, j - width)))
            jmax = int(round(min(self.img_dict['DAPI'].shape[0], j + width + 1)))
            box = np.array(comb_image[imin:imax, jmin:jmax].copy())
            data_list.append(np.array(np.stack(tuple(box), axis=0)).astype('float32'))
            del box
        return data_list

    def _get_metadata(self):
        self.channels = self._meta_data.channels
        try:
            self.cell_line = self._meta_data.well_conditions(self._well.getId())['cell_line']
        except KeyError:
            self.cell_line = self._meta_data.well_conditions(self._well.getId())['Cell_Line']


        # self.condition = self._meta_data.well_conditions(self._well.getId())['condition']
        row_list = list('ABCDEFGHIJKL')
        self.well_pos = f"{row_list[self._well.row]}{self._well.column}"

    def _get_img_dict(self):
        """divide image_array with flatfield correction mask and return dictionary "channel_name": corrected image"""
        img_dict = {}
        for channel in list(self.channels.items()):  # produces a tuple of channel key value pair (ie ('DAPI':0)
            corr_img = generate_image(self.omero_image, channel[1]) / self._flatfield_dict[channel[0]]
            img_dict[channel[0]] = corr_img[30:1050, 30:1050]  # using channel key here to link each image with its channel
        return img_dict

    def segmentation_figure(self):
        """Generate matplotlib image for segmentation check and save to path (quality control)
        """
        dapi_img = scale_img(self.img_dict['DAPI'])
        tub_img = scale_img(self.img_dict['Tub'])
        dapi_color_labels = color_label(self.n_mask, dapi_img)
        tub_color_labels = color_label(self.cyto_mask, dapi_img)
        fig_list = [dapi_img, tub_img, dapi_color_labels, tub_color_labels]
        title_list = ["DAPI image", "Tubulin image", "DAPI segmentation", "Tubulin image"]
        fig, ax = plt.subplots(ncols=4, figsize=(16, 7))
        for i in range(4):
            ax[i].axis('off')
            ax[i].imshow(fig_list[i])
            ax[i].title.set_text(title_list[i])
        save_fig(self._paths.quality_ctr, f'{self.well_pos}_segmentation_check')
        plt.close(fig)

    def save_example_tiff(self):
        """Combines arrays from image_dict and saves images as tif files"""
        comb_image = np.dstack(list(self.img_dict.values()))
        io.imsave(str(self._paths.example_img / f'{self.well_pos}_segmentation_check.tif'), comb_image,
                  check_contrast=False)
