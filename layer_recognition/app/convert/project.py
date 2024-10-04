""" The quPath project to metadata convert click command """

import os

import click
import pandas as pd

from layer_recognition.io import get_qpproject_images_metadata


@click.command()
@click.option(
    "--qupath-project-path",
    required=True,
    help="qupath project path that contains images metadata",
)
@click.option(
    "--output-path",
    required=True,
    help="Directory path that will contain the converted QuPath project's \
        metadata into Dataframes file",
)
def cmd(
    output_path,
    qupath_project_path,
):
    """
    Convert QuPath project metadata top pandas dataframe
    """

    os.makedirs(output_path, exist_ok=True)
    print(f"INFO: Get QuPath Project metadata {qupath_project_path}")
    images_metadata = get_qpproject_images_metadata(qupath_project_path)
    project_meta_data = {}

    for meta in images_metadata:
        image_name = meta["imageName"].replace(" ", "")
        project_meta_data[image_name] = {}
        project_meta_data[image_name]["Image Name"] = image_name
        try:
            value = meta["metadata"]["Exclude"]
            if value.find("Exclude") > -1:
                project_meta_data[image_name]["Exclude"] = True
            else:
                project_meta_data[image_name]["Exclude"] = False
        except KeyError:
            project_meta_data[image_name]["Exclude"] = False

        try:
            value = meta["metadata"]["Analyze"]
            if value == "True":
                project_meta_data[image_name]["Analyze"] = True
            else:
                project_meta_data[image_name]["Analyze"] = False

        except KeyError:
            project_meta_data[image_name]["Analyze"] = False

        try:
            project_meta_data[image_name]["Distance to midline"] = meta["metadata"][
                "Distance to midline"
            ]
        except KeyError:
            project_meta_data[image_name]["Distance to midline"] = 0

        try:
            project_meta_data[image_name]["Comment"] = meta["metadata"]["Comment"]
        except KeyError:
            project_meta_data[image_name]["Comment"] = "0"

    meta_df = pd.DataFrame.from_dict(project_meta_data).T

    slash_pos = qupath_project_path.rfind("/") + 1
    project_name = qupath_project_path[slash_pos:]
    meta_path = output_path + "/" + project_name + "_" "Metadata_information.csv"
    meta_path = meta_path.replace(" ", "")
    print(f"INFO: Export Project metadata to {meta_path}")
    meta_df = meta_df.reset_index(drop=True)
    meta_df.to_csv(meta_path)
