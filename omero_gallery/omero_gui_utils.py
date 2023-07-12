import matplotlib.pyplot as plt
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel
import pandas as pd
from omero_gallery.galleries_plot import plot_gallery
from omero_gallery.gen_functions_gallery import get_cell_phase_id,cell_data_extraction


def processing_image(plate_id, file_path, num_rows, num_cols,well,condition, cell_line,cell_phase,channel):

    df = pd.read_csv(str(file_path))
    # if well is not None:
    #     df_gallery = get_gallery_df(df, plate_id, well=well, cell_line=cell_line, condition=condition)
    #     if cell_line is not None:
    #         df_gallery = get_gallery_df(df, plate_id, well=well, cell_line=cell_line, condition=condition)
    #         if condition is not None:
    #             df_gallery = get_gallery_df(df, plate_id, well=well, cell_line=cell_line, condition=condition)
    # else:
    #     # df_gallery=get_gallery_df(df,plate_id,well=well,cell_line=cell_line,condition=condition)
    df_gallery=df

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
        self.Well_ID_edit = QLineEdit()
        self.condition_edit = QLineEdit()
        self.cell_line_edit = QLineEdit()
        self.channel_edit = QLineEdit()
        self.cell_phase_edit = QLineEdit()

        self.layout().addWidget(QLabel('Plate ID:'))
        self.layout().addWidget(self.plate_id_edit)
        self.layout().addWidget(QLabel('File path:'))
        self.layout().addWidget(self.file_path_edit)
        self.layout().addWidget(QLabel('Num rows:'))
        self.layout().addWidget(self.num_rows_edit)
        self.layout().addWidget(QLabel('Num cols:'))
        self.layout().addWidget(self.num_cols_edit)
        self.layout().addWidget(QLabel('Well_ID:'))
        self.layout().addWidget(self.Well_ID_edit)
        self.layout().addWidget(QLabel('Condition:'))
        self.layout().addWidget(self.condition_edit)
        self.layout().addWidget(QLabel('Cell line:'))
        self.layout().addWidget(self.cell_line_edit)
        self.layout().addWidget(QLabel('Channel:'))
        self.layout().addWidget(self.channel_edit)
        self.layout().addWidget(QLabel('Cell phase:'))
        self.layout().addWidget(self.cell_phase_edit)

        self.btn = QPushButton('Start', self)
        self.btn.clicked.connect(self.load_images)
        self.layout().addWidget(self.btn)

    def load_images(self):
        plate_id = self.plate_id_edit.text()
        file_path = self.file_path_edit.text()
        num_rows = int(self.num_rows_edit.text())
        num_cols = int(self.num_cols_edit.text())
        well_id = self.Well_ID_edit.text()
        condition = self.condition_edit.text()
        cell_line = self.cell_line_edit.text()
        channel = self.channel_edit.text()
        cell_phase = self.cell_phase_edit.text()

        # For this version, we're just printing the values
        print(f"Plate ID: {plate_id}")
        print(f"File path: {file_path}")
        print(f"Num rows: {num_rows}")
        print(f"Num cols: {num_cols}")
        print(f"Well ID: {well_id}")
        print(f"condition: {condition}")
        print(f"cell_line: {cell_line}")
        print(f"channel: {channel}")
        print(f"Cell phase: {cell_phase}")
        # Here you should put your code to load and visualize images using the given parameters.
        self.image_gallery=processing_image(plate_id, file_path, num_rows, num_cols,well_id,condition, cell_line,cell_phase,channel)
        self.viewer.add_image(self.image_gallery,contrast_limits=[0, 1], rgb=True)

