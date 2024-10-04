""" The convert click command """

import click

from layer_recognition.convert import single_image_conversion
from layer_recognition.io import list_images
from layer_recognition.utilities import get_config


@click.command()
@click.option("--config-file-path", required=False, help="Configuration file path")
def cmd(config_file_path):
    """
    Convert QuPath output files to pandas dataframes
    Args:
        config-file-path (str): The configuration file path
    """
    (
        input_detection_directory,
        cell_position_suffix,
        input_annotation_directory,
        annotations_geojson_suffix,
        exclude_flag,
        pixel_size,
        output_path,
    ) = get_config(config_file_path)
    images_dictionary = list_images(
        input_detection_directory,
        cell_position_suffix,
        input_annotation_directory,
        annotations_geojson_suffix,
    )

    for image_prefix in images_dictionary.keys():
        print(f"INFO: Process single image {image_prefix}")
        single_image_conversion(
            output_path,
            image_prefix,
            input_detection_directory,
            input_annotation_directory,
            pixel_size,
            exclude=exclude_flag,
        )
