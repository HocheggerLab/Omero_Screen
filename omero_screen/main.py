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



#%%

#%%
def merge_data(df1,df2,merge_clue_columns='well_id',merge_key_columns=['well','plate_id','well_id','image_id','cell_line','condition','Cyto_ID']):
    """
    :param df1: Dataframe, original analysis data
    :param df2: Dataframe, cell cycle data
    :param merge_clue_columns: the columns using to split df2 to multiple dataframes , default 'well_id'
    :param merge_key_columns: key columns using to merge two dataframe,
                             default ['experiment','plate_id','well_id','cell_line','condition','Cyto_ID','intensity_mean_EdU_cyto','intensity_mean_H3P_cyto','area_cell','area_nucleus',]
    :return: merged Dataframe
    """
    all_merged_df=pd.DataFrame()
    for i in df1[merge_clue_columns].unique().tolist():
        # merge two data based on the same well id, hwo=inner:use intersection of keys from both frames, drop NAN columns before merge
        merged_df = pd.merge(df1[df1[merge_clue_columns]==i], df2[df2[merge_clue_columns]==i],on=merge_key_columns).dropna()
        all_merged_df=pd.concat([all_merged_df,merged_df])
    return all_merged_df








@omero_connect
def main(plate_id, conn=None):
    stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
    meta_data = MetaData(plate_id, conn)
    exp_paths = ExpPaths(meta_data)
    df_final = pd.DataFrame()
    df_quality_control = pd.DataFrame()
    for count, well in enumerate(list(meta_data.plate_obj.listChildren())):
        print(well)
        if count+1 in [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,]:
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
            # df_final.to_csv('/Users/haoranyue/Desktop/data/data.csv')
            # print("df_final")
            df_quality_control = pd.concat([df_quality_control, well_quality])
            # for index, i in enumerate(df_final['cell_data'].tolist()):
            #     print(i[:,:,0].max(),index)

    df_final = pd.concat([df_final.loc[:, 'experiment':], df_final.loc[:, :'experiment']], axis=1).iloc[:, :-1]
    df_final.to_csv(exp_paths.final_data / f"{meta_data.plate}_final_data.csv")

    if 'H3P' in meta_data.channels.keys():
        cellcycle_analysis(df_final, exp_paths.path, meta_data.plate, H3=True)
    else:
        cellcycle_analysis(df_final, exp_paths.path, meta_data.plate, H3=False)


if __name__ == '__main__':
    main(928)

