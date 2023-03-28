import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow_model.tf_model import tensorflow_model
from omero_screen.data_structure import Defaults, MetaData, ExpPaths
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.general_functions import save_fig, generate_image, filter_segmentation, omero_connect, scale_img, \
    color_label
import skimage
from cellpose import models
from skimage import measure, io
import cv2
import os

import tensorflow as tf
class Image:
    """
    generates the corrected images and segmentation masks.
    Stores corrected images as dict, and n_mask, c_mask and cyto_mask arrays.
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
        self.n_mask = self._n_segmentation()
        self.c_mask = self._c_segmentation()
        self.cyto_mask = self._get_cyto()
        self.data=self._get_data(width=20)


    def _get_data(self, width=20):
        empty_channel = np.zeros((self.img_dict['DAPI'].shape[0], self.img_dict['DAPI'].shape[1]))
        comb_image = np.dstack([empty_channel, self.img_dict['Tub'], self.img_dict['DAPI']]).astype('float32')
        comb_image=(comb_image - comb_image.min()) / comb_image.max()-comb_image.min()
        data_list =[]
        df_props = pd.DataFrame(measure.regionprops_table(self.c_mask, properties=('label', 'centroid',)))
        df_props_selected=df_props[df_props['label'].isin(self.samples)]
        for label in (df_props_selected['label'].tolist()):

            # centroid = region.centroid
            i = df_props.loc[df_props['label'] == label, 'centroid-0'].item()
            j = df_props.loc[df_props['label'] == label, 'centroid-1'].item()
            imin = int(round(max(0, i - width)))
            imax = int(round(min(self.c_mask.shape[0], i + width + 1)))
            jmin = int(round(max(0, j - width)))
            jmax = int(round(min(self.c_mask.shape[1], j + width + 1)))
            comb_image[:, :, 0] = self.c_mask
            red_channel= (comb_image[:, :, 0] == label) * np.ones(
                (comb_image[:, :, 0].shape[0], comb_image[:, :, 0].shape[1])).copy()*0.01
            green_channel=((red_channel!=0) * comb_image[:, :, 1]).copy()
            blue_channel=((red_channel!=0) * comb_image[:, :, 2]).copy()
            tem_comb_image = np.dstack([red_channel, green_channel, blue_channel]).astype('float32')
            box = np.array(tem_comb_image[imin:imax, jmin:jmax].copy())
            data_list.append(np.array(np.stack(tuple(box), axis=0)).astype('float32'))
            del box,red_channel,green_channel,blue_channel,tem_comb_image
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

    def _get_models(self):
        """
        Matches well with cell line and gets model_path for cell line from plate_layout
        :param number: int 0 or 1, 0 for nuclei model, 1 for cell model
        :return: path to model (str)
        """
        return Defaults.MODEL_DICT[self.cell_line.replace(" ", "").upper()]

    def _n_segmentation(self):
        """perform cellpose segmentation using nuclear mask """
        model = models.CellposeModel(gpu=True, model_type=os.path.dirname(os.getcwd())+'/data/CellPose_models/'+Defaults.MODEL_DICT['nuclei'])

        n_channels = [[0, 0]]
        n_mask_array, n_flows, n_styles = model.eval(self.img_dict['DAPI'], channels=n_channels)
        # return cleaned up mask using filter function
        return filter_segmentation(n_mask_array)

    def _c_segmentation(self):
        """perform cellpose segmentation using cell mask """
        model = models.CellposeModel(gpu=True, model_type=os.path.dirname(os.getcwd())+'/data/CellPose_models/'+self._get_models())
        c_channels = [[2, 1]]
        # combine the 2 channel numpy array for cell segmentation with the nuclei channel
        comb_image = np.dstack([self.img_dict['DAPI'], self.img_dict['Tub']])
        c_masks_array, c_flows, c_styles = model.eval(comb_image, channels=c_channels)
        # return cleaned up mask using filter function
        return filter_segmentation(c_masks_array)

    def _get_cyto(self):
        """substract nuclei mask from cell mask to get cytoplasm mask """
        overlap = (self.c_mask != 0) * (self.n_mask != 0)
        cyto_mask_binary = (self.c_mask != 0) * (overlap == 0)
        return self.c_mask * cyto_mask_binary

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
