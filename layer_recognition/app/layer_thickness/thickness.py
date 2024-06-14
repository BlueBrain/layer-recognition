""" The layer thickness click command """

from collections import defaultdict
import configparser

import glob
import pathlib
import os

import click
import numpy as np
import pandas as pd

from layer_recognition.utilities import get_s1hl_corners

from layer_recognition.utilities import (
    get_animal_by_image_id,
    get_image_to_exlude_list,
    get_image_id,
)

from layer_recognition.geometry import get_layers_thickness


@click.command()
@click.option(
    "--feature-file-path",
    type=pathlib.Path,
    required=True,
    help="Path to the directory that contains the cells'features including the RF_predictioh",
)
@click.option(
    "--points-annotations-path",
    type=pathlib.Path,
    required=True,
    help="Path to the directory that contains thepoints annotations",
)
@click.option(
    "--metadata-path",
    type=pathlib.Path,
    required=True,
    help="Path to the directory that contains the cells'features including the RF_predictioh",
)
@click.option(
    "--output-filename",
    type=pathlib.Path,
    required=True,
    help="Path to file that will contain the dataframe with layer thickness results",
)
@click.option(
    "--image-to-exclude-path",
    type=pathlib.Path,
    required=False,
    help="If provided, exclude the image listed in this file",
)
def cmd(
    feature_file_path,
    points_annotations_path,
    metadata_path,
    output_filename,
    image_to_exclude_path,
):
    """
    Compute layers thickness and saved result to a dataframe2
    """
    regex = feature_file_path / "*.csv"
    features_filelist = glob.glob(regex.as_posix())
    total = len(features_filelist)
    index = 0

    if image_to_exclude_path:
        df_image_to_exclude = pd.read_excel(
            image_to_exclude_path, index_col=0, skiprows=[0, 1, 2, 3, 4, 5, 6, 7]
        )
        db_image_to_exclude_list = get_image_to_exlude_list(df_image_to_exclude)
    else:
        db_image_to_exclude_list = []

    animal_by_image = get_animal_by_image_id(metadata_path)

    """
    rectangle_widths_by_animal = defaultdict(lambda: defaultdict(list))
    for features_path in features_filelist:
        image_id = get_image_id(features_path)
        animal = animal_by_image[image_id]

        if image_id in db_image_to_exclude_list:
            print(f"INFO Exclude {image_id}")
            continue
        df_feat = pd.read_csv(features_path, index_col=0)
        rectangle_widths_by_animal[animal] = get_image_layers_thickness(
            df_feat, rectangle_widths_by_animal[animal]
        )
        print(f"INFO Done {index}/{total}\r", end="")
        index += 1
        if index >= 20:
            break
    """
    animals_layers_thickness = defaultdict(lambda: defaultdict(list))

    for features_path in features_filelist:
        image_id = get_image_id(features_path)
        animal = animal_by_image[image_id]

        if image_id in db_image_to_exclude_list:
            print(f"INFO Exclude {image_id}")
            continue
        df_feat = pd.read_csv(features_path, index_col=0)

        points_annotation_path = (
            str(points_annotations_path) + "/" + image_id + "_points_annotations.csv"
        )
        points_annotation_df = pd.read_csv(points_annotation_path, index_col=0)
        top_left, top_right, bottom_right, bottom_left = get_s1hl_corners(
            points_annotation_df
        )

        animals_layers_thickness[animal] = get_layers_thickness(
            df_feat, top_left, top_right, bottom_left, bottom_right
        )
        print(f"INFO Done {index}/{total}\r", end="")
        index += 1
    print(f"DEBUG animals_layers_thickness {animals_layers_thickness}")

    thickness_mean_by_animal = defaultdict(list)

    for animal, layers_thickness in animals_layers_thickness.items():
        for layer, thickness in layers_thickness.items():
            thickness_mean_by_animal[layer].append(np.mean(thickness))

    thickness_mean_animal = []
    thickness_layer_animal = []
    for layer, values in thickness_mean_by_animal.items():
        thickness_mean_animal.append(values)
        thickness_layer_animal.append(layer)

    print(f"thickness_mean_animal {thickness_mean_animal}")
    print(f"thickness_layer_animal {thickness_layer_animal}")

    d = {
        "layers": thickness_layer_animal,
        "thickness_mean": thickness_mean_animal,
    }

    df = pd.DataFrame(data=d)
    df.to_csv(output_filename)
    print(f"INFO DONE: layers thickness information saved to {output_filename} ")
