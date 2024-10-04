""" USe a mahcine learning model to predict brain layers """

import os
import pickle
import pathlib
import click


from layer_recognition.ml.train_and_predict import predict
from layer_recognition.ml.utils import get_classes_and_features


@click.command()
@click.option(
    "--model-file",
    type=pathlib.Path,
    help=("Path to the file where to load the RF model."),
    required=True,
)
@click.option(
    "--pred-dir",
    type=pathlib.Path,
    help=(
        "Directory containing images to be predicted + glob pattern to match the"
        " corresponding files. No prediction if not specified."
    ),
    required=True,
)
@click.option(
    "--pred-save",
    type=pathlib.Path,
    help="Where to save the CSVs containing predictions.",
    required=True,
)
@click.option(
    "--pred-glob",
    type=str,
    default="*.csv",
    help="Glob pattern to match the prediction files.",
)
@click.option(
    "--pred-extension",
    type=click.Choice(["txt", "csv"]),
    default="csv",
    help="Extension of the files used for prediction.",
)
@click.option(
    "--gt-column",
    type=str,
    default="Expert_layer",
    help="Name of the ground truth column in the CSVs.",
)
@click.option(
    "--distinguishable-second-layer",
    "-d",
    is_flag=True,
    default=False,
    help="Treats layer 2 and 3 as separate layers.",
)
@click.option(
    "--extension",
    type=click.Choice(["txt", "csv"]),
    default="csv",
    help="extension of the files containing the data.",
)
@click.option(
    "--clean",
    "-c",
    type=bool,
    default=True,
    help=(
        "Use post-processing to attempt removing wrongly classified cluster. Can"
        " potentially harm the prediction's quality."
    ),
)
@click.option(
    "--eps",
    type=float,
    default=0.05,
    help="Radius of the circle used in the DBSCAN algorithm for post-processing.",
)
@click.option(
    "--min-samples",
    type=int,
    default=3,
    help=(
        "Minimum number of neighbors required within eps to be treated as a central"
        " point."
    ),
)
@click.option(
    "--neighbors",
    type=int,
    help="Number of neighbors for KNN.",
    default=32,
)
def cmd(
    model_file,
    gt_column,
    distinguishable_second_layer,
    extension,
    clean,
    eps,
    min_samples,
    pred_dir,
    pred_save,
    pred_glob,
    pred_extension,
    neighbors,
):
    """
    predict brain layers on images
    """

    # Features kept for classification.
    classes, features = get_classes_and_features(distinguishable_second_layer)

    if model_file and not model_file.suffix == ".pkl":
        raise ValueError(
            "The model_file argument should end with a filename with the '.pkl'"
            " extension."
        )
    assert os.path.exists(model_file)
    rf = pickle.load(open(model_file, "rb"))

    if pred_dir:
        predict(
            pred_dir=pred_dir,
            model=rf,
            features=features,
            classes=classes,
            pred_save=pred_save,
            pred_glob=pred_glob,
            distinguishable_second_layer=distinguishable_second_layer,
            pred_extension=pred_extension,
            gt_column=gt_column,
            clean=clean,
            eps=eps,
            min_samples=min_samples,
        )
