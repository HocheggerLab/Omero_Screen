import pandas as pd
from omero_gallery.galleries_plot import plot_gallery
import os
import random



def get_gallery_df(df,plate_id,well=None,cell_line=None,condition='siCtr'):
    """
    Return a filtered version of the input Dataframe based on the plate_id, welll, cell_line,condition
    """
    df_gallery = df[df['plate_id'] == plate_id]
    if well is not None:
       df_gallery = df_gallery[df_gallery['well_id'] == well]
    if cell_line is not None:
       df_gallery = df_gallery[df_gallery['cell_line'] == cell_line]
    if condition is not None:
       df_gallery = df_gallery[df_gallery['condition'] == condition]

    return df_gallery

def main():
    # Ask user for plate id
    plate_id=int(input("Enter the plate id: "))

    # Ask user for path to csv file
    file_path = input("Enter path to file: ")
    df=pd.read_pickle(str(file_path))
    save_dir=os.path.dirname(file_path)

    # Ask user for number of rows and columns
    num_rows = int(input("Enter the number of rows: "))
    num_cols = int(input("Enter the number of columns: "))

    # Ask user if they want to select a specific well or all wells
    well_option=input("Do you want to select a specific well? (Yes/No): ")
    well= None
    cell_line = None
    condition=None
    if well_option.lower()=='yes':
        well=int(input('Enter the well_id: '))
        cell_line_option = input("Do you want to select a cell_line? (Yes/No): ")
        # cell_line=None
        df_gallery = get_gallery_df(df, plate_id, well=well, cell_line=cell_line, condition=condition)
        if cell_line_option.lower()=='yes':
            cell_line=str(input('Enter the cell_line '))
            condition_option=input('Do you want to select a condition? (Yes/No):')
            condition=None
            df_gallery = get_gallery_df(df, plate_id, well=well, cell_line=cell_line, condition=condition)
            if condition_option.lower()=='yes':
                condition=input('Enter the condition: ')
                df_gallery = get_gallery_df(df, plate_id, well=well, cell_line=cell_line, condition=condition)

    else:
        df_gallery=get_gallery_df(df,plate_id,well=well,cell_line=cell_line,condition=condition)

    print(df_gallery)
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
        plot_gallery(df_gallery, cell_cycle_detailed='cell_cycle_detailed', check_phase=cc_phase,
                     channels_option=channels_option, gallery_name=name_option, nrows=num_rows, ncols=num_cols,
                     path=save_dir)



if __name__=="__main__":
    main()






