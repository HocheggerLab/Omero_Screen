import matplotlib.pyplot as plt
import pandas as pd
from omero_screen.data_structure import Defaults, MetaData, ExpPaths
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.general_functions import  omero_connect
from omero_gallery.extract_cell_samples import Image_extract
from omero_screen.general_functions import save_fig, generate_image, filter_segmentation, omero_connect, scale_img, \
    color_label
import numpy as np




# def get_gallery_df(df,plate_id,well=None,cell_line=None,condition=None,image_id=None):
#     """
#     Return a filtered version of the input Dataframe based on the plate_id, welll, cell_line,condition,image_id
#     """
#     df_gallery = df[df['plate_id'] == plate_id]
#     if well is not None:
#        df_gallery = df_gallery[df_gallery['well_id']==well]
#     if cell_line is not None:
#        df_gallery = df_gallery[df_gallery['well_id']==cell_line]
#     if condition is not None:
#        df_gallery = df_gallery[df_gallery['condition']==condition]
#     if image_id is not None:
#        df_gallery= df_gallery[df_gallery['image_id'].isin(image_id)]
#     return df_gallery


def img_centroid_ids(random_df,image_id:int)->list:
    tem_df=random_df[random_df['image_id']==image_id]
    cell_id_list = tem_df['centroid'].tolist()
    return cell_id_list

def get_well_ids(df,well_id:int)->list:
    tem_df = df[df['well_id'] == well_id]
    img_ids_list = tem_df['image_id'].tolist()
    return img_ids_list

def well_images_dict(well_dict:dict,image_dict:dict)->dict:
    combined_dict = {}
    for well, id_list in well_dict.items():
        combined_dict[well] = {}
        for id_ in id_list:
            if id_ in image_dict:
                combined_dict[well][id_] = image_dict[id_]
    return combined_dict

def get_cell_phase_id(df:pd.DataFrame,cell_phase:str,selected_num:int)->dict:
    df=df[df['cell_cycle_detailed']==cell_phase]
    if df.empty:
        print(f"No rows with cell_phase '{cell_phase}' found.")
        return {}
    print(f'{cell_phase} Total number: {len(df)}, Random select: {selected_num}')
    # Randomly sample the rows from the DataFrame
    if selected_num> len(df):
        random_rows= df.sample(n=len(df))
    else:
        random_rows = df.sample(n=selected_num)
    random_rows['centroid'] = random_rows.apply(lambda row: (row['centroid-0'], row['centroid-1']), axis=1)
    image_id_list=random_rows['image_id'].unique()
    well_id_list = random_rows['well_id'].unique()
    image_dict = {image_id: img_centroid_ids(random_df=random_rows, image_id=image_id) for image_id in image_id_list}
    well_dict = {well: get_well_ids(df=random_rows, well_id=well) for well in
                      well_id_list}
    samples_ids=well_images_dict(well_dict,image_dict)
    return samples_ids

@omero_connect
def cell_data_extraction(plate_id,samples:dict,conn=None):
    meta_data = MetaData(plate_id, conn)
    exp_paths = ExpPaths(meta_data)
    samples_list_img = []
    masks_list=[]
    for well_id in samples.keys():
        well = conn.getObject("Well", well_id)
        flatfield_dict = flatfieldcorr(well, meta_data, exp_paths)
        omero_img_list=[well.getImage(number).getId() for number in range(len(list(well.listChildren())))]
        omero_img_index=[]
        for images_id in samples[well_id].keys():
            if int(images_id) in omero_img_list:
               omero_img_index.append(omero_img_list.index(images_id))

        for idx in omero_img_index:
            image = well.getImage(idx)
            imgs,masks=Image_extract(well, image, meta_data, exp_paths, samples[well_id][image.getId()], flatfield_dict)._get_data(width=20)

            samples_list_img.append(imgs)
            masks_list.append(masks)

            # samples_list.append(Image_extract(well,well.getImage(idx) , meta_data, exp_paths, samples[str(well)][well.getImage(idx).getId()], flatfield_dict).data)
    if samples_list_img:  # Check if samples_list is not empty
        if isinstance(samples_list_img[0],list):
           flat_list_img = [item for sublist in samples_list_img for item in sublist]
           flat_list_mask= [item for sublist in masks_list for item in sublist]
        else:
           flat_list_img=samples_list_img
           flat_list_mask=masks_list
    else:
        flat_list_img=[]
        flat_list_mask=[]
    return flat_list_img,flat_list_mask


@omero_connect
def load_well_image(plate_id,well_id,conn=None):
    meta_data = MetaData(plate_id, conn)
    exp_paths = ExpPaths(meta_data)
    well = conn.getObject("Well", well_id)
    flatfield_dict = flatfieldcorr(well, meta_data, exp_paths)
    omero_well_img_list = [well.getImage(number).getId() for number in range(len(list(well.listChildren())))]
    img_list={}
    for idx in omero_well_img_list:
        image = conn.getObject("Image", idx)
        img_dict=get_img_dict(meta_data,flatfield_dict,image)
        assert img_dict['Tub'].shape == img_dict['DAPI'].shape == img_dict['EdU'].shape == img_dict['H3P'].shape, "All images must have the same shape"

        # Stack the images along a new axis to create a 4-channel image
        image_4channel = np.stack((img_dict['DAPI'], img_dict['Tub'], img_dict['EdU']), axis=-1)
        comb_image = (image_4channel - image_4channel.min()) / (image_4channel.max() - image_4channel.min())
        img_list[idx]=comb_image

    return img_list

def get_img_dict(meta_data,flatfield_dict,image):
        """divide image_array with flatfield correction mask and return dictionary "channel_name": corrected image"""
        img_dict = {}
        for channel in list(meta_data.channels.items()):  # produces a tuple of channel key value pair (ie ('DAPI':0)
            corr_img = generate_image(image, channel[1]) / flatfield_dict[channel[0]]
            img_dict[channel[0]] = corr_img[30:1050,30:1050]
        return img_dict


if __name__=='__main__':
    img_list=load_well_image(1237,15401)

    plt.imshow(img_list[452327])
    plt.show()
