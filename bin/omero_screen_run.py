import omero_screen
from omero_screen import main, EXCEL_PATH

EXEL_PATH = omero_screen.EXCEL_PATH
if __name__ == '__main__':
    if not EXCEL_PATH:
        EXCEL_PATH = input("Provide path to metadata excel file: ")
    omero_screen.main.main(excel_path=EXCEL_PATH)
