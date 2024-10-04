""" The convert click command """

from collections import defaultdict
import configparser
import glob
import os

import click


import numpy as np
import pandas as pd
from pathlib import Path


from layer_recognition.density import (
    single_image_process_per_depth,
    single_image_process_per_layer,
    compute_animal_densities,
)

from layer_recognition.io import write_dataframe_to_file
from layer_recognition.utilities import get_image_to_exlude_list


@click.command()
@click.option("--config-file-path", required=False, help="Configuration file path")
@click.option("--visualisation-flag", is_flag=True)
@click.option("--save-plot-flag", is_flag=True)
def cmd_animal(
    config_file_path,
    visualisation_flag,
    save_plot_flag,
):
    """
    Compute cell densities as function of brain depth
    """
    config = configparser.ConfigParser()
    config.sections()
    config.read(config_file_path)

    cell_feature_path = Path(config["BATCH"]["cell_feature_path"])
    s1hl_path = Path(config["BATCH"]["s1hl_path"])
    metadata_path = Path(config["BATCH"]["metadata_path"])
    output_path = Path(config["BATCH"]["output_path"])
    layer_thickness = float(config["BATCH"]["thickness_cut"])

    try:
        cell_position_file_prefix = config["BATCH"]["cell_position_file_prefix"]
    except KeyError:
        cell_position_file_prefix = "Features_"

    try:
        S1HL_file_sufix = config["BATCH"]["s1lh_file_prefix"]
    except KeyError:
        S1HL_file_sufix = "_S1HL_annotations"

    try:
        image_to_exlude_path = config["BATCH"]["image_to_exlude_path"]
        df_image_to_exclude = pd.read_excel(
            image_to_exlude_path, index_col=0, skiprows=[0, 1, 2, 3, 4, 5, 6, 7]
        )
        db_image_to_exlude_list = get_image_to_exlude_list(df_image_to_exclude)
    except KeyError:
        db_image_to_exlude_list = []

    densities = compute_animal_densities(
        cell_feature_path,
        s1hl_path,
        metadata_path,
        layer_thickness,
        cell_position_file_prefix,
        S1HL_file_sufix,
        db_image_to_exlude_list,
    )

    densities_mean = []
    densities_std = []

    for animal, values in densities.items():
        mean = np.mean(values)
        densities_mean.append(mean)
        std = np.std(values)
        densities_std.append(std)
        print(
            f"INFO {animal} mean cells density {mean:.0f} cells/mm3,  standard deviation {std:.2f}"
        )

    # dictionary of lists
    dict = {
        "animal": densities.keys(),
        "densities_mean": densities_mean,
        "densities_std": densities_std,
    }

    if not os.path.exists(output_path):
        # if the directory is not present then create it.
        os.makedirs(output_path)
    df = pd.DataFrame.from_dict(dict)
    output_filepath = output_path / "cell_density_by_animal.csv"
    df.to_csv(output_filepath)

    print(
        f"INFO: S1HL mean cells density {np.mean(densities_mean):.0f} cells/mm3,  standard deviation {np.std(densities_mean):.2f} "
    )
    print(f"INFO: Done Dataframe saved to {output_filepath}")


@click.command()
@click.option("--config-file-path", required=False, help="Configuration file path")
@click.option("--visualisation-flag", is_flag=True)
@click.option("--save-plot-flag", is_flag=True)
@click.option(
    "--image-to-exlude-path",
    help="exel files that contains the list of image to exclude (xlsx).",
    required=False,
)
def cmd_depth(
    config_file_path,
    visualisation_flag,
    save_plot_flag,
    image_to_exlude_path,
):
    """
    Compute cell densities as function of brain depth
    """
    config = configparser.ConfigParser()
    config.sections()
    config.read(config_file_path)

    output_path = config["BATCH"]["output_path"]
    cell_position_path = config["BATCH"]["cell_position_path"]
    try:
        cell_position_file_prefix = config["BATCH"]["cell_position_file_prefix"]
    except KeyError:
        cell_position_file_prefix = "Features_"

    points_annotations_path = config["BATCH"]["points_annotations_path"]

    try:
        points_annotations_file_prefix = config["BATCH"][
            "points_annotations_file_prefix"
        ]
    except KeyError:
        points_annotations_file_prefix = ""

    s1hl_path = config["BATCH"]["s1hl_path"]

    try:
        s1hl_file_prefix = config["BATCH"]["s1hl_file_prefix"]
    except KeyError:
        s1hl_file_prefix = ""

    thickness_cut = int(config["BATCH"]["thickness_cut"])
    nb_row = int(config["BATCH"]["nb_row"])
    nb_col = int(config["BATCH"]["nb_col"])

    multiple_image_process_per_depth(
        cell_position_path,
        cell_position_file_prefix,
        output_path,
        image_to_exlude_path,
        visualisation_flag,
        save_plot_flag,
        points_annotations_path=points_annotations_path,
        points_annotations_file_prefix=points_annotations_file_prefix,
        s1hl_path=s1hl_path,
        s1hl_file_prefix=s1hl_file_prefix,
        thickness_cut=thickness_cut,
        nb_col=nb_col,
        nb_row=nb_row,
    )


@click.command()
@click.option("--config-file-path", required=False, help="Configuration file path")
@click.option("--visualisation-flag", is_flag=True)
@click.option("--save-plot-flag", is_flag=True)
@click.option(
    "--image-to-exlude-path",
    help="exel files that contains the list of image to exclude (xlsx).",
    required=False,
)
def cmd_layer(
    config_file_path,
    visualisation_flag,
    save_plot_flag,
    image_to_exlude_path,
):
    """
    Compute cell densities per brain layer
    """
    config = configparser.ConfigParser()
    config.sections()
    config.read(config_file_path)

    output_path = config["BATCH"]["output_path"]
    cell_position_path = config["BATCH"]["cell_position_path"]

    try:
        cell_position_file_prefix = config["BATCH"]["cell_position_file_prefix"]
    except KeyError:
        cell_position_file_prefix = "Features_"

    try:
        alpha = int(config["BATCH"]["alpha"])
    except KeyError:
        alpha = 0.05

    try:
        meta_df_path = config["BATCH"]["meta_df_path"]
    except KeyError:
        meta_df_path = None

    try:
        animal_id = config["BATCH"]["animal_id"]
    except KeyError:
        animal_id = None

    multiple_image_process_per_layer(
        cell_position_path,
        cell_position_file_prefix,
        output_path,
        image_to_exlude_path,
        visualisation_flag,
        save_plot_flag,
        meta_df_path=meta_df_path,
        animal_id=animal_id,
        alpha=alpha,
    )


def multiple_image_process_per_layer(
    cell_position_path,
    cell_position_file_prefix,
    output_path,
    image_to_exlude_path,
    visualisation_flag,
    save_plot_flag,
    meta_df_path=None,
    animal_id=None,
    alpha=0,
):
    """
    loop over image and execute single_image_process for each image
    Args:
        cell_position_path: (str)
        cell_position_file_prefix: (str)
        output_path: (str)
        image_to_exlude_path: (str)
        visualisation_flag (bool) If True display plots
        save_plot_flag (bool) If True, save plots
        alpha: (float) alphashape alpha value. Only used if compute_per_layer is True
    """
    # List images to compute
    image_path_list = glob.glob(cell_position_path + "/*.csv")
    animal_frames = []
    if meta_df_path and animal_id:
        meta_df = pd.read_csv(meta_df_path, index_col=0)
        animal_df = meta_df[meta_df["Project_ID"] == animal_id]
        animal_images = animal_df.Image_Name.to_list()

    image_list = []
    feature_str_length = len(cell_position_file_prefix)
    for path in image_path_list:
        prefix_pos = path.rfind(cell_position_file_prefix)
        if prefix_pos > -1:
            feature_pos = path.rfind(cell_position_file_prefix) + feature_str_length
            image_name = path[feature_pos : path.find(".csv")]
            find_animal = True
            if animal_id:
                find_animal = False
                for image in animal_images:
                    if image_name.find(image) > -1:
                        find_animal = True
        if find_animal:
            image_list.append(image_name)

    if len(image_list) == 0:
        print("WARNING: No input files to process.")
        return

    print(f'INFO" {len(image_list)} to process')

    if not os.path.exists(output_path):
        # if the directory is not present then create it.
        os.makedirs(output_path)
        print(f"INFO: Create output_path {output_path}")

    # Verify that the image is not in the exlude images list
    df_image_to_exclude = None
    if image_to_exlude_path:
        df_image_to_exclude = pd.read_excel(
            image_to_exlude_path, index_col=0, skiprows=[0, 1, 2, 3, 4, 5, 6, 7]
        )

        # Some image may be keep if we only compute the density per layer
        df_image_to_exclude = df_image_to_exclude[
            df_image_to_exclude["Exclusion reason (Cell density calculation)"].str.find(
                "DistanceToMidline_3.05-3.25"
            )
            == -1
        ]
    for image_name in image_list:
        print("INFO: Process single image ", image_name)
        cell_position_file_path = (
            cell_position_path + "/" + cell_position_file_prefix + image_name + ".csv"
        )

        per_layer_dataframe, layer_boundaries_dataframe = single_image_process_per_layer(
            image_name,
            cell_position_file_path,
            output_path,
            df_image_to_exclude=df_image_to_exclude,
            thickness_cut=50,
            visualisation_flag=visualisation_flag,
            save_plot_flag=save_plot_flag,
            alpha=alpha,
        )

        if layer_boundaries_dataframe is not None:
            layer_boundary_dataframe_full_path = (
                output_path + "/" + image_name + "__layer_boundaries.csv"
            )
            write_dataframe_to_file(
                layer_boundaries_dataframe, layer_boundary_dataframe_full_path
            )
            print(
                f"INFO: Write layers boundary dataframe to \
             {layer_boundary_dataframe_full_path}"
            )



        animal_frames.append(per_layer_dataframe)

        if per_layer_dataframe is None:
            print(
                "ERROR: The computed density per layer is not valid to compute the per \
             layer density"
            )
        else:
            densities_per_layer_dataframe_full_path = (
                output_path + "/" + image_name + "_per_layer.csv"
            )
            write_dataframe_to_file(
                per_layer_dataframe, densities_per_layer_dataframe_full_path
            )
            print(
                f"INFO: Write density per layer dataframe to \
             {densities_per_layer_dataframe_full_path}"
            )

    if meta_df_path and animal_id:
        animal_df = pd.concat(animal_frames, ignore_index=True, axis=0).mean(axis=0)
        animal_per_layer_dataframe_full_path = (
            output_path + animal_id + "_per_layer.csv"
        )
        write_dataframe_to_file(animal_df, animal_per_layer_dataframe_full_path)
        print(
            f"INFO: Write density per layer for animal {animal_id} dataframe to \
         {animal_per_layer_dataframe_full_path}"
        )


def multiple_image_process_per_depth(
    cell_position_path,
    cell_position_file_prefix,
    output_path,
    image_to_exlude_path,
    visualisation_flag,
    save_plot_flag,
    points_annotations_path=None,
    points_annotations_file_prefix=None,
    s1hl_path=None,
    s1hl_file_prefix=None,
    nb_col=20,
    nb_row=20,
    thickness_cut=50,
):
    """
    loop over image and execute single_image_process for each image
    Args:
        cell_position_path: (str)
        cell_position_file_prefix: (str)
        output_path: (str)
        image_to_exlude_path: (str)
        points_annotations_path: (str)
        points_annotations_file_prefix: (str)
        s1hl_path: (str)
        s1hl_file_prefix: (str)
        thickness_cut: (float) : The thickness of the slices
        nb_col: (int): Nb of grifd columns. Only used if compute_per_depth is True
        nb_row: (int) : Nb of grid rows. Only used if compute_per_depth is True
        visualisation_flag (bool) If True display plots
        save_plot_flag (bool) If True, save plots
        alpha: (float) alphashape alpha value. Only used if compute_per_layer is True
    """
    # List images to compute
    image_path_list = glob.glob(cell_position_path + "/*.csv")

    image_list = []
    feature_str_length = len(cell_position_file_prefix)
    for path in image_path_list:
        prefix_pos = path.rfind(cell_position_file_prefix)
        if prefix_pos > -1:
            feature_pos = path.rfind(cell_position_file_prefix) + feature_str_length
            image_list.append(path[feature_pos : path.find(".csv")])

    if len(image_list) == 0:
        print("WARNING: No input files to process.")
        return

    print(f'INFO" {len(image_list)} to process {image_list}')

    if not os.path.exists(output_path):
        # if the directory is not present then create it.
        os.makedirs(output_path)
        print(f"INFO: Create output_path {output_path}")

    # Verify that the image is not in the exlude images list
    df_image_to_exclude = None
    if image_to_exlude_path:
        df_image_to_exclude = pd.read_excel(
            image_to_exlude_path, index_col=0, skiprows=[0, 1, 2, 3, 4, 5, 6, 7]
        )

    for image_name in image_list:
        print("INFO: Process single image ", image_name)
        cell_position_file_path = (
            cell_position_path + "/" + cell_position_file_prefix + image_name + ".csv"
        )

        points_annotations_file_path = (
            points_annotations_path
            + "/"
            + points_annotations_file_prefix
            + image_name
            + "_points_annotations.csv"
        )
        s1hl_file_path = (
            s1hl_path + "/" + s1hl_file_prefix + image_name + "_S1HL_annotations.csv"
        )

        densities_dataframe, layer_boundaries_dataframe = single_image_process_per_depth(
            image_name,
            cell_position_file_path,
            points_annotations_file_path,
            s1hl_file_path,
            output_path,
            df_image_to_exclude=df_image_to_exclude,
            thickness_cut=thickness_cut,
            nb_col=nb_col,
            nb_row=nb_row,
            visualisation_flag=visualisation_flag,
            save_plot_flag=save_plot_flag,
        )

        if densities_dataframe is None:
            print(
                f"ERROR: {image_name} The computed density is not valid to compute\
the per depth density"
            )
        else:
            densities_dataframe_full_path = output_path + "/" + image_name + "_density_depth.csv"

            write_dataframe_to_file(densities_dataframe, densities_dataframe_full_path)
            print(f"INFO: Write density dataframe =to {densities_dataframe_full_path}")

        if layer_boundaries_dataframe is not None:
            layer_boundary_dataframe_full_path = (
                output_path + "/" + image_name + "_layer_boundaries.csv"
            )
            write_dataframe_to_file(
                layer_boundaries_dataframe, layer_boundary_dataframe_full_path
            )
            print(
                f"INFO: Write layers boundary dataframe to \
                {layer_boundary_dataframe_full_path}"
            )
