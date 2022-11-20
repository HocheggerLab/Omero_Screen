from omero_screen.aggregator import ImageAggregator
from omero_screen.general_functions import save_fig, scale_img, generate_image, generate_random_image, \
    omero_connect
from omero_screen.data_structure import MetaData, ExpPaths
from omero_screen import EXCEL_PATH, SEPARATOR
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib
import pathlib
import glob

matplotlib.use('MacOSX')  # avoid matplotlib warning about interactive backend


def flatfieldcorr(well, meta_data, exp_paths) -> dict:
    """

    :return:
    """

    channels = meta_data.channels
    well_pos = meta_data.well_pos(well.getId())
    template_path = exp_paths.flatfield_templates
    rep_fig_path = exp_paths.flatfield_rep_figs
    template_subfolder_path = template_path / f"{well_pos}"
    rep_fig_subfolder_path = rep_fig_path / f"{well_pos}"
    if pathlib.Path.exists(template_subfolder_path):
        return load_corr_dict(template_subfolder_path, channels)
    return generate_corr_dict(well, well_pos, channels, template_subfolder_path, rep_fig_subfolder_path)


def load_corr_dict(path, channels):
    print(f"Loading Flatfield Correction Masks from File\n{SEPARATOR}")
    corr_img_list = glob.glob(f'{str(path)}/*.tif')
    if len(corr_img_list) == len(list(channels.keys())):
        array_list = list(map(io.imread, corr_img_list))
        channel_list = list(channels.keys())
        return dict(zip(channel_list, array_list))


def generate_corr_dict(well, well_pos, channels, template_path, rep_image_path):
    """
    Saves each flat field mask file with well position and channel name
    :return: a dictionary with channel_name : flatfield correction masks
    """
    print(f"\nAssembling Flatfield Correction Masks for each Channel\n{SEPARATOR}")
    template_path.mkdir(exist_ok=True)
    rep_image_path.mkdir(exist_ok=True)
    corr_dict = {}
    # iteration extracts tuple (channel_name, channel_number) as channel
    for channel in tqdm(list(channels.items())):
        corr_img_id = f"{well_pos}_{channel[0]}"
        norm_mask = aggregate_imgs(well, channel)
        io.imsave(template_path / f"{corr_img_id}_flatfield_masks.tif", norm_mask)
        example = gen_example(well, channel, norm_mask)
        example_fig(example, well_pos, channel, rep_image_path)
        corr_dict[channel[0]] = norm_mask  # associates channel name with flatfield mask
    return corr_dict


def aggregate_imgs(well, channel):
    """
    Aggregate images in well for specified channel and generate correction mask using the Aggregator Module
    :param channel: dictionary from self.exp_data.channels
    :return: flatfield correction mask for given channel
    """
    agg = ImageAggregator(60)
    for i, img in enumerate(well.listChildren()):
        image = well.getImage(i)
        image_array = generate_image(image, channel[1])
        agg.add_image(image_array)
    blurred_agg_img = agg.get_gaussian_image(30)
    return blurred_agg_img / blurred_agg_img.mean()


def gen_example(well, channel, mask):
    example_img = generate_random_image(well, channel)
    scaled = scale_img(example_img)
    corr_img = example_img / mask
    corr_scaled = scale_img(corr_img)
    # order all images for plotting
    return [(scaled, 'original image'), (np.diagonal(example_img), 'diag. intensities'),
            (corr_scaled, 'corrected image'), (np.diagonal(corr_img), 'diag. intensities'),
            (mask, 'flatfield correction mask')]


def example_fig(data_list, well_pos, channel, path):
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    for i, data_tuple in enumerate(data_list):
        plt.sca(ax[i])
        if i in [0, 2, 4]:
            plt.imshow(data_tuple[0], cmap='gray')
        else:
            plt.plot(data_tuple[0])
            plt.ylim(data_tuple[0].min(), 5 * data_tuple[0].min())
        plt.title(data_tuple[1])
    # save and close figure
    fig_id = f"{well_pos}_{channel[0]}_flatfield_check"  # using channel name
    save_fig(path, fig_id)
    plt.close(fig)


# test


if __name__ == "__main__":
    @omero_connect
    def flatfield_test(excel_path, conn=None):
        meta_data = MetaData(excel_path)
        exp_paths = ExpPaths(conn, meta_data)
        well = conn.getObject("Well", 10707)
        return flatfieldcorr(well, meta_data, exp_paths)


    flatfield_corr = flatfield_test(EXCEL_PATH)
    print(flatfield_corr['DAPI'].shape)
