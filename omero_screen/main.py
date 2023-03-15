from omero_screen import Defaults, SEPARATOR
from omero_screen.general_functions import omero_connect
from omero_screen.data_structure import MetaData, ExpPaths
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.omero_loop import well_loop
from cellcycle_analysis import cellcycle_analysis
from stardist.models import StarDist2D
import pandas as pd
from omero_screen.galleries import gallery_data




@omero_connect
def main(plate_id, conn=None):
    stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
    meta_data = MetaData(plate_id, conn)
    exp_paths = ExpPaths(meta_data)
    df_final = pd.DataFrame()
    df_quality_control = pd.DataFrame()
    for count, well in enumerate(list(meta_data.plate_obj.listChildren())):
        # print(well)
        if count+1 in [96,95,94,93,92,91,90,89,87,88,86,
                       85,84,83,82,81,80,79,78,77,76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61,60,59,
                       # 58,57,56,55,54,53,52,51,50,49,48,47,46,45,43,42,41,40,39,38,37,36,35,34,32,31,30,29,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,
                       ]:
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
    df_final.to_csv(exp_paths.final_data / f"{meta_data.plate}_final_data.csv")
    if 'H3P' in meta_data.channels.keys():
        cc_data=cellcycle_analysis(df_final, exp_paths.path, meta_data.plate, H3=True)
    else:
        cc_data=cellcycle_analysis(df_final, exp_paths.path, meta_data.plate, H3=False)
    # %% generate the gallery to check
    gallery_data(cc_data,check_phase='M', total=4000,images_per_row=80)

if __name__ == '__main__':
    # main(928)
    main(1054)
    # main(1056)

