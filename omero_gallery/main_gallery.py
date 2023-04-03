import matplotlib.pyplot as plt
import pandas as pd
from omero_gallery.galleries_plot import plot_gallery
import os
import random
from omero_gallery.gen_functions_gallery import get_gallery_df,get_cell_phase_id,cell_data_extraction



def main():
    # Ask user for path to csv file
    file_path = input("Enter path to cellcycle_detailed file: ")
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
    if df_gallery.empty:
        print("No data found based on the provided filters. Please check your input parameters.")
        return
    # Ask user a specific cell cycle phase
    phase_option = input("Please select adf specific cell cycle phase? (All/Sub-G1/Polyploid/G1/Early S/Late S/Polyploid(replicating)/G2/M) ")
    # Ask user for a specific channel
    channels_option = input('Please select the specific channel? (All/Tubulin/Dapi) ')
    # Ask user for a specific channel
    name_option = str(input('Please input the specific name for saving? (for example: Screen_test) '))

    if phase_option.lower()=='all':
        cc_phases=["Sub-G1",'Polyploid', 'G1', 'Early S', 'Late S', 'Polyploid(replicating)', 'G2', 'M']

    elif phase_option in ["Sub-G1",'Polyploid', 'G1', 'Early S', 'Late S', 'Polyploid(replicating)', 'G2', 'M']:
        cc_phases = [phase_option.capitalize()]

    else:
        raise ValueError('Invalid inuput. Please enter a correct cell phase')

    for cc_phase in cc_phases:
        sample_ids = get_cell_phase_id(df=df_gallery, cell_phase=cc_phase, selected_num=num_cols * num_rows)
        filtered_images = cell_data_extraction(plate_id, sample_ids)


        if filtered_images:  # Add this line to check if the list is not empty
            plot_gallery(filtered_images, check_phase=cc_phase, channels_option=channels_option,
                         gallery_name=name_option,
                         nrows=num_rows,
                         path=save_dir)
        else:
            print(f"No images found for phase {cc_phase}. Skipping this phase.")

if __name__=="__main__":
    main()

