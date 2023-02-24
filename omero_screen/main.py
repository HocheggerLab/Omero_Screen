from omero_screen import Defaults, SEPARATOR
from omero_screen.general_functions import omero_connect
from omero_screen.data_structure import MetaData, ExpPaths
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.omero_loop import well_loop
from cellcycle_analysis import cellcycle_analysis
from stardist.models import StarDist2D
import pandas as pd
from CNN_pytorch.galleries import gallery_data





# #%%
# def merge_data(df1,df2,merge_clue_columns='well_id',merge_key_columns=['experiment','plate_id','well_id','cell_line','condition','Cyto_ID','intensity_mean_EdU_cyto','intensity_mean_H3P_cyto','area_cell','area_nucleus',]):
#     """
#     :param df1: Dataframe, original analysis data
#     :param df2: Dataframe, cell cycle data
#     :param merge_clue_columns: the columns using to split df2 to multiple dataframes , default 'well_id'
#     :param merge_key_columns: key columns using to merge two dataframe,
#                              default ['experiment','plate_id','well_id','cell_line','condition','Cyto_ID','intensity_mean_EdU_cyto','intensity_mean_H3P_cyto','area_cell','area_nucleus',]
#     :return: merged Dataframe
#     """
#     all_merged_df=pd.DataFrame()
#     for i in df1[merge_clue_columns].unique().tolist():
#         # merge two data based on the same well id, hwo=inner:use intersection of keys from both frames, drop NAN columns before merge
#         merged_df = pd.merge(df1[df1[merge_clue_columns]==i], df2[df2[merge_clue_columns]==i],how='right',on=merge_key_columns).dropna()
#         all_merged_df=pd.concat([all_merged_df,merged_df])
#     return all_merged_df
# #%%











@omero_connect
def main(plate_id, conn=None):
    stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
    meta_data = MetaData(plate_id, conn)
    exp_paths = ExpPaths(meta_data)
    df_final = pd.DataFrame()
    df_quality_control = pd.DataFrame()
    for count, well in enumerate(list(meta_data.plate_obj.listChildren())):
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
    df_final.to_csv(exp_paths.final_data / f"{meta_data.plate}_final_data.csv")
    # df_original = pd.read_csv(
    #     '/Users/haoranyue/Downloads/221011_Cellcycleprofile_Exp3_siRNAs_MM231_RPE1_U2OS_singlecell_cellcycle.csv')
    # df_merge = merge_data(df_final, df_original)
    # df_check = df_merge[df_merge['cell_cycle'] == "G2/M"]
    # df_check_M = df_check[df_check['cell_cycle_detailed'] == "M"]
    # gallery_data(df_check_M, ['cell_data', 'inter_M'], 'inter',400, images_per_row=20)
    # gallery_data(df_check_M, ['cell_data', 'inter_M'], 'M', 390, images_per_row=20)
    if 'H3P' in meta_data.channels.keys():
        cellcycle_analysis(df_final, exp_paths.path, meta_data.plate, H3=True)
    else:
        cellcycle_analysis(df_final, exp_paths.path, meta_data.plate, H3=False)


if __name__ == '__main__':
    main(928)

