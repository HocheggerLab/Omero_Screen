import matplotlib.pyplot as plt
import skimage.io
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel
import pandas as pd
from omero_gallery.galleries_plot import plot_gallery
from omero_gallery.gen_functions_gallery import get_cell_phase_id,cell_data_extraction,load_well_image


def get_gallery_df(df,plate_id,well_id=None):
    """
    Return a filtered version of the input Dataframe based on the plate_id, welll, cell_line,condition
    """
    df_gallery = df.loc[df['plate_id'] == int(plate_id)]
    if well_id is not None:
       df_gallery = df_gallery.loc[df_gallery['well_id'] == well_id]

    return df_gallery

def processing_image(plate_id, file_path, num_rows, num_cols,well_id,cell_phase,channel):

    df = pd.read_csv(str(file_path))
    df_gallery=get_gallery_df(df,plate_id,well_id=well_id)
    if cell_phase in ["Sub-G1",'Polyploid', 'G1', 'Early S', 'Late S', 'Polyploid(replicating)', 'G2', 'M']:
        cc_phases = [cell_phase.capitalize()]
    else:
        raise ValueError('Invalid inuput. Please enter a correct cell phase')
    for cc_phase in cc_phases:
        sample_ids = get_cell_phase_id(df=df_gallery, cell_phase=cc_phase, selected_num=num_cols * num_rows)
        filtered_images = cell_data_extraction(plate_id, sample_ids)

    if filtered_images:  # Add this line to check if the list is not empty
        image_gallery=plot_gallery(filtered_images, check_phase=cc_phase, channels_option=channel,
                     nrows=num_rows)
    else:
        print(f"No images found for phase {cc_phase}. Skipping this phase.")

    return image_gallery

class MyWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer
        self.setLayout(QVBoxLayout())



        self.plate_id_edit = QLineEdit()
        self.file_path_edit = QLineEdit()
        self.num_rows_edit = QLineEdit()
        self.num_cols_edit = QLineEdit()
        self.Well_id_edit = QLineEdit()
        self.channel_edit = QLineEdit()
        self.cell_phase_edit = QLineEdit()

        self.layout().addWidget(QLabel('Plate ID:'))
        self.layout().addWidget(self.plate_id_edit)
        self.layout().addWidget(QLabel('File path:'))
        self.layout().addWidget(self.file_path_edit)
        self.layout().addWidget(QLabel('Well_id:'))
        self.layout().addWidget(self.Well_id_edit)
        self.layout().addWidget(QLabel('Num rows:'))
        self.layout().addWidget(self.num_rows_edit)
        self.layout().addWidget(QLabel('Num cols:'))
        self.layout().addWidget(self.num_cols_edit)
        self.layout().addWidget(QLabel('Channel:'))
        self.layout().addWidget(self.channel_edit)
        self.layout().addWidget(QLabel('Cell phase:'))
        self.layout().addWidget(self.cell_phase_edit)

        self.btn = QPushButton('Start', self)
        self.btn.clicked.connect(self.load_images)
        self.layout().addWidget(self.btn)

        self.save_btn = QPushButton('Save', self)
        self.save_btn.clicked.connect(self.save_process)
        self.layout().addWidget(self.save_btn)
    def load_images(self):
        plate_id = self.plate_id_edit.text()
        file_path = self.file_path_edit.text()
        num_rows = int(self.num_rows_edit.text()) if self.num_rows_edit.text() else None
        num_cols = int(self.num_cols_edit.text()) if self.num_cols_edit.text() else None
        well_id = int(self.Well_id_edit.text())
        channel = self.channel_edit.text() if self.channel_edit.text() else None
        cell_phase = self.cell_phase_edit.text() if self.cell_phase_edit.text() else None

        self.log_parameters(plate_id,file_path,num_rows,num_cols,well_id,channel,cell_phase)

        # Check if the necessary parameters are provided
        if plate_id and file_path and well_id is not None:
            self.image_list = load_well_image(plate_id, well_id)
            for i in self.image_list.keys():
                self.viewer.add_image(self.image_list[i], name=str(i), contrast_limits=[0, 1], rgb=True)

        if plate_id and file_path and well_id and num_cols and channel and cell_phase and num_rows is not None:
            self.image_gallery = processing_image(plate_id, file_path, num_rows, num_cols, well_id, cell_phase, channel)
            self.viewer.add_image(self.image_gallery, contrast_limits=[0, 1], rgb=True)

    def log_parameters(self, plate_id, file_path, num_rows, num_cols, well_id, channel, cell_phase):
        print(f"Plate ID: {plate_id}")
        print(f"File path: {file_path}")
        print(f"Num rows: {num_rows}")
        print(f"Num cols: {num_cols}")
        print(f"Well ID: {well_id}")
        print(f"channel: {channel}")
        print(f"Cell phase: {cell_phase}")

    def save_process(self):
        print('Save the current image')


if __name__=="__main__":
    processing_image(plate_id=1237,file_path='/Users/haoranyue/Desktop/OmeroScreen_test/cellcycle_summary/OmeroScreen_test_singlecell_cellcycle_detailed.csv',
                     num_rows=2, num_cols=2,well_id=15401,cell_phase='G1',channel="all")
