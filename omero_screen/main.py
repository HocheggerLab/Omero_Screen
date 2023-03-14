from omero_screen import Defaults, SEPARATOR
from omero_screen.general_functions import omero_connect
from omero_screen.data_structure import MetaData, ExpPaths
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.omero_loop import well_loop
from cellcycle_analysis import cellcycle_analysis
from stardist.models import StarDist2D
import pandas as pd
from CNN_pytorch.galleries import gallery_data
from CNN_pytorch.training_gui import TrainingScreen
import numpy as np
from tensorflow_model.tf_model import tensorflow_model
import tensorflow as tf

#%%

#%%










@omero_connect
def main(plate_id, conn=None):
    stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
    meta_data = MetaData(plate_id, conn)
    exp_paths = ExpPaths(meta_data)
    df_final = pd.DataFrame()
    df_quality_control = pd.DataFrame()
    for count, well in enumerate(list(meta_data.plate_obj.listChildren())):
        # print(well)
        if count+1 in [96,95,94,93,92,91,90,89,87,88,86,85,84,83,82,81,80,79,78,77]:
            continue
        ann = well.getAnnotation(Defaults.NS)
        try:
            cell_line = dict(ann.getValue())['cell_line']
        except KeyError:
            cell_line = dict(ann.getValue())['Cell_Line']
        if cell_line != 'Empty':

            print(f"\n{SEPARATOR} \nAnalysing well row:{well.row}/col:{well.column} - {count + 1} of {meta_data.plate_length}.\n{SEPARATOR}")
            flatfield_dict = flatfieldcorr(well, meta_data, exp_paths)
            well_data, well_quality = well_loop(well, meta_data, exp_paths, flatfield_dict, stardist_model)
            df_final = pd.concat([df_final, well_data])

            df_quality_control = pd.concat([df_quality_control, well_quality])

    df_final = pd.concat([df_final.loc[:, 'experiment':], df_final.loc[:, :'experiment']], axis=1).iloc[:, :-1]

    # df_final.to_csv(exp_paths.final_data / f"{meta_data.plate}_final_data.csv")


    # nor_list = [np.array(i).astype('float32') for i in df_check_M[df_check_M['inter_M']=='inter']['cell_data'].tolist()]
    # nor_list = [np.array(i).astype('float32') for i in df_check_M['cell_data'].tolist() if i.shape == (41, 41, 3)]
    # gallery_data(df_check_M, ['cell_data', 'inter_M'], 'inter',600, images_per_row=30)
    # screen = TrainingScreen(nor_list)
    # gallery_data(df_check_inter, ['cell_data', 'inter_M'], 'M',180, images_per_row=20)
    # gallery_data(df_check_M, ['cell_data', 'inter_M'], 'inter',120, images_per_row=12)
    if 'H3P' in meta_data.channels.keys():
        cc_data=cellcycle_analysis(df_final, exp_paths.path, meta_data.plate, H3=True)
    else:
        cc_data=cellcycle_analysis(df_final, exp_paths.path, meta_data.plate, H3=False)

    gallery_data(cc_data,check_phase='M', total=25,images_per_row=5)

if __name__ == '__main__':
    # main(928)
    main(1054)
    # main(1056)

