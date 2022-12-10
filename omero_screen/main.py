from omero_screen.general_functions import omero_connect
from omero_screen.data_structure import MetaData, ExpPaths
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen import EXCEL_PATH
from omero_loop import *
from plotnine import *
import pandas as pd
from data_phase_summary import assign_cell_cycle_phase,save_folder
from data_plots import plot_scatter_Edu_G2,plot_distribution_H3_P
from omero.gateway import BlitzGateway
from heatmap_cell_cycle import figure_heatmap, _ggsave
@omero_connect
def main(excel_path=EXCEL_PATH, conn=None):
    meta_data = MetaData(excel_path)
    exp_paths = ExpPaths(conn, meta_data)
    df_final = pd.DataFrame()
    df_quality_control = pd.DataFrame()
    for count, ID in enumerate(meta_data.plate_layout["Well_ID"]):
        print(f"Analysing well {meta_data.well_pos(ID)} - {count + 1} of {meta_data.plate_length}.\n{SEPARATOR}")
        well = conn.getObject("Well", ID)
        flatfield_dict = flatfieldcorr(well, meta_data, exp_paths)
        well_data, well_quality = well_loop(well, meta_data, exp_paths, flatfield_dict)
        df_final = pd.concat([df_final, well_data])
        df_quality_control = pd.concat([df_quality_control, well_quality])
    df_final = pd.concat([df_final.iloc[:,- 7:], df_final.iloc[:,:-7]], axis=1)
    df_final.to_csv(exp_paths.final_data / f"{exp_paths.plate_name}_final_data.csv")
    df_quality_control.to_csv(exp_paths.quality_ctr / f"{exp_paths.plate_name}_quality_data.csv")
    plot_scatter_Edu_G2(data_dir=exp_paths.final_data, path_export=exp_paths.path, conn=conn, )
    plot_distribution_H3_P(data_dir=exp_paths.final_data, path_export=exp_paths.path, conn=conn)




if __name__ == '__main__':
    main()



