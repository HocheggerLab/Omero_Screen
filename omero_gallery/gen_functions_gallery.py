import matplotlib.pyplot as plt
import pandas as pd
from omero_screen.data_structure import Defaults, MetaData, ExpPaths
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.general_functions import  omero_connect
from omero_gallery.extract_cell_samples import Image_extract






def get_gallery_df(df,plate_id,well=None,cell_line=None,condition=None,image_id=None):
    """
    Return a filtered version of the input Dataframe based on the plate_id, welll, cell_line,condition,image_id
    """
    df_gallery = df[df['plate_id'] == plate_id]
    if well is not None:
       df_gallery = df_gallery[df_gallery['well_id']==well]
    if cell_line is not None:
       df_gallery = df_gallery[df_gallery['cell_line']==cell_line]
    if condition is not None:
       df_gallery = df_gallery[df_gallery['condition']==condition]
    if image_id is not None:
       df_gallery= df_gallery[df_gallery['image_id'].isin(image_id)]
    return df_gallery


def img_centroid_ids(random_df,image_id:int)->list:
    tem_df=random_df[random_df['image_id']==image_id]
    cell_id_list = tem_df['centroid'].tolist()

    return cell_id_list

def get_cell_phase_id(df:pd.DataFrame,cell_phase:str,selected_num:int)->dict:
    df=df[df['cell_cycle_detailed']==cell_phase]
    print(f'{cell_phase} Total number: {len(df)}, Random select: {selected_num}')
    # Randomly sample the rows from the DataFrame
    if selected_num> len(df):
        random_rows= df.sample(n=len(df))
    else:
        random_rows = df.sample(n=selected_num)
    random_rows['centroid'] = random_rows.apply(lambda row: (row['centroid-0'], row['centroid-1']), axis=1)
    image_id_list=random_rows['image_id'].unique()
    sample_ids = {image_id: img_centroid_ids(random_df=random_rows, image_id=image_id) for image_id in image_id_list}
    return sample_ids

@omero_connect
def cell_data_extraction(plate_id,well_id,samples:dict,conn=None):
    meta_data = MetaData(plate_id, conn)
    exp_paths = ExpPaths(meta_data)
    well = conn.getObject("Well", well_id)
    flatfield_dict = flatfieldcorr(well, meta_data, exp_paths)
    image_number = len(list(well.listChildren()))
    omero_img_list=[well.getImage(number).getId() for number in range(image_number)]
    omero_img_index=[]
    samples_list=[]
    for images_id in samples.keys():
        if int(images_id) in omero_img_list:
            omero_img_index.append(omero_img_list.index(images_id))
        else:
            print(f'Cannot find the images id {images_id}')
    for idx in omero_img_index:
        samples_list.append(Image_extract(well,well.getImage(idx) , meta_data, exp_paths, samples[well.getImage(idx).getId()], flatfield_dict).data)
    flat_list = [item for sublist in samples_list for item in sublist]
    return flat_list
