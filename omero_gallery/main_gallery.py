import napari
from omero_gallery.omero_gui_utils import MyWidget

def get_gallery_df(df,plate_id,well=None,cell_line=None,condition='siCtr'):
    """
    Return a filtered version of the input Dataframe based on the plate_id, welll, cell_line,condition
    """
    df_gallery = df.loc[df['plate_id'] == int(plate_id)]
    if well is not None:
       df_gallery = df_gallery[df_gallery['well_id'] == well]
    if cell_line is not None:
       df_gallery = df_gallery[df_gallery['cell_line'] == cell_line]
    if condition is not None:
       df_gallery = df_gallery[df_gallery['condition'] == condition]

    return df_gallery


def main():
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(MyWidget(viewer), area='right')
    napari.run()

if __name__ == '__main__':
    main()
