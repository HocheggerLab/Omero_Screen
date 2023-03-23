from omero_screen import Defaults, SEPARATOR
from omero_screen.general_functions import omero_connect
from omero_screen.data_structure import MetaData, ExpPaths
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.omero_loop import well_loop
from cellcycle_analysis import cellcycle_analysis
from stardist.models import StarDist2D
import pandas as pd





@omero_connect
def main(plate_id, conn=None):
    """
    Analyze a plate of well data and output the results.

    :param plate_id: The ID of the plate to analyze
    :param conn: Omero connection to login in.

    :return:
       None
    """
    stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
    meta_data = MetaData(plate_id, conn)
    exp_paths = ExpPaths(meta_data)
    df_final = pd.DataFrame()
    df_quality_control = pd.DataFrame()
    for count, well in enumerate(list(meta_data.plate_obj.listChildren())):
        if count in range( 8,96):
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
        # Ask for user input
        user_input = input("Do you want to perform H3P analysis? (yes/no): ")
        # Check user input and perform the appropriate analysis
        if user_input.lower() == 'no':
        # Call the CNN_classification function
            cellcycle_analysis(df_final, exp_paths.path, meta_data.plate, H3=False)
        elif user_input.lower() == 'yes':
        # Call the H3P_analysis function
            cellcycle_analysis(df_final, exp_paths.path, meta_data.plate, H3=True)
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
    else:
        cellcycle_analysis(df_final, exp_paths.path, meta_data.plate, H3=False)

    # if cc_data is not None:
    #     gallery_data(cc_data,cell_cycle_detaild='cell_cycle_detailed',check_phase='M', gallery_name='CNN_determined',total=25,images_per_row=5)





if __name__ == '__main__':
    # main(928)
    main(1237)
    # main(1054)
    # main(1125)
    # main(1056)
    # main(1273)
