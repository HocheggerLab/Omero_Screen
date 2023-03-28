import matplotlib.pyplot as plt
import pandas as pd
from omero_gallery.galleries_plot import plot_gallery
import os
import random
from omero_screen.data_structure import Defaults, MetaData, ExpPaths
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.general_functions import save_fig, generate_image, filter_segmentation, omero_connect, scale_img, \
    color_label
from omero_gallery.extract_cell_samples import Image

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

def fet_cell_phase_id(df:pd.DataFrame,cell_phase:str,selected_num:int)->list:
    df=df[df['cell_cycle_detailed']==cell_phase]
    image_id_list=df['image_id'].unique()
    image_tem_id=selected_num//len(image_id_list)
    sample_ids=dict()
    def each_imeage(image_id,image_tem_id:int)->list:
        tem_df=df[df['image_id']==image_id]
        phase_id_list = tem_df['Cyto_ID'].tolist()
        if image_tem_id > len(phase_id_list):
            sample_ids = random.sample(phase_id_list, len(phase_id_list))
        else:
            sample_ids = random.sample(phase_id_list, image_tem_id)
        return sample_ids

    for i in image_id_list:
        sample_ids[i]=each_imeage(image_id=i,image_tem_id=image_tem_id)
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
        samples_list.append(Image(well,well.getImage(idx) , meta_data, exp_paths, samples[well.getImage(idx).getId()], flatfield_dict).data)
    flat_list = [item for sublist in samples_list for item in sublist]
    return flat_list

def main():
    # Ask user for path to csv file
    file_path = input("Enter path to file: ")
    df=pd.read_csv(str(file_path))
    save_dir=os.path.dirname(file_path)
    # Ask user for number of rows and columns
    num_rows = int(input("Enter the number of rows: "))
    num_cols = int(input("Enter the number of columns: "))
    # Ask user for plate_id
    plate_id = int(input("Enter the plate id: "))

    # Initialize variables for user input
    well = cell_line = condition = image_ids = None
    # Ask user if they want to select a specific well or all wells
    if input("Do you want to select a specific well? (Yes/No): ").lower() == 'yes':
        well = int(input('Enter the well_id: '))
        if input("Do you want to select a cell_line? (Yes/No): ").lower()=='yes':
            cell_line = input('Enter the cell_line: ')
            if input("Do you want to select a condition? (Yes/No): ").lower() == 'yes':
                condition = input('Enter the condition: ')
                if input("Do you want to select specific image_id? (Yes/No): ").lower() == 'yes':
                    image_id_input = input('Enter the image_Id (comma-separated for multiple ids): ')
                    image_ids = [int(id.strip()) for id in image_id_input.split(',')]

    df_gallery = get_gallery_df(df, plate_id, well=well, cell_line=cell_line, condition=condition, image_id=image_ids)
    well_ids=df_gallery['well_id'].unique()

    # Ask user a specific cell cycle phase
    phase_option = input("Please select a specific cell cycle phase? (All/Sub-G1/Polyploid/G1/Early S/Late S/Polyploid(replicating)/G2/M) ")
    # Ask user for a specific channel
    channels_option = input('Please select the specific channel? (All/Tubulin/Dapi) ')
    # Ask user for a specific channel
    name_option = str(input('Please input the specific name for saving? (for example: Screen_test) '))

    if phase_option.lower()=='all':
        cc_phases=['All',"Sub-G1",'Polyploid', 'G1', 'Early S', 'Late S', 'Polyploid(replicating)', 'G2', 'M']

    elif phase_option in ["Sub-G1",'Polyploid', 'G1', 'Early S', 'Late S', 'Polyploid(replicating)', 'G2', 'M']:
        cc_phases = [phase_option.capitalize()]

    else:
        raise ValueError('Invalid inuput. Please enter a correct cell phase')

    for cc_phase in cc_phases:
        sample_ids = fet_cell_phase_id(df=df_gallery, cell_phase=phase_option, selected_num=num_cols * num_rows)
        if len(well_ids)>1:
            for wel in well_ids:
                tem_filtered_df = cell_data_extraction(plate_id, wel, sample_ids)
            filtered_images = [item for sublist in tem_filtered_df for item in sublist]
        else:
            filtered_images = cell_data_extraction(plate_id, well_ids[0], sample_ids)

        plot_gallery(filtered_images, check_phase=cc_phase, channels_option=channels_option, gallery_name=name_option,
                     nrows=num_rows, ncols=num_cols,
                     path=save_dir)

if __name__=="__main__":
    main()
    # feature_extraction_test(plate_id=1237,well_id=15400,samples=[1,2,3])
