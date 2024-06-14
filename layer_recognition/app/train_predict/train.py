""" Train a machine learning model that predicts brain layers """

import pathlib
import click
from sklearn.model_selection import train_test_split
from layer_recognition.ml.utils import get_image_files, get_classes_and_features
from layer_recognition.ml.train_and_predict import train_and_evaluate_model


@click.command()
@click.option(
    "--train-dir",
    type=pathlib.Path,
    help="Directory containing training data files.",
    required=True,
)
@click.option(
    "--save-dir",
    type=pathlib.Path,
    help="Directory where to save the model.",
    required=True,
)
@click.option(
    "--train-glob",
    type=str,
    default="*.csv",
    help="Glob pattern to match the training files.",
)
@click.option(
    "--extension",
    type=click.Choice(["txt", "csv"]),
    default="csv",
    help="extension of the files containing the data.",
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
    default=True,
    help="Treats layer 2 and 3 as separate layers.",
)
@click.option(
    "--random-split",
    is_flag=True,
    default=True,
    help=(
        "Use test images randomly extracted from the train set. If false, defaults"
        " to the predefine hardcoded images."
    ),
)
@click.option(
    "--split-ratio",
    "-s",
    type=float,
    default=0.1,
    help=(
        "Fraction of the dataset sent to the test set. Only if using" " --random-split."
    ),
)
@click.option(
    "--estimators",
    "-e",
    type=int,
    default=100,
    help="Number of estimators for the random forest model.",
)
@click.option(
    "--train-knn",
    is_flag=True,
    default=False,
    help="Train a K Nearest Neighbor model",
)
def cmd(
    train_dir,
    train_glob,
    extension,
    save_dir,
    gt_column,
    distinguishable_second_layer,
    random_split,
    split_ratio,
    estimators,
    train_knn,
):
    """
    Train a model based on cells feature that predicts brain layers
    """
    # Features kept for classification.
    classes, features = get_classes_and_features(distinguishable_second_layer)

    # Get the image names and split them in train/test.
    filenames = get_image_files(train_dir, train_glob)

    if split_ratio > 0 and random_split:
        train_image_names, test_image_names = train_test_split(
            filenames, test_size=split_ratio, random_state=42, shuffle=True
        )
    elif split_ratio == 0 and random_split:
        train_image_names = filenames
        test_image_names = None
    else:
        test_image_names = [
            "Features_SLD_0000565.vsi-20x_01.csv",
            "Features_SLD_0000540.vsi-20x_01.csv",
            "Features_SLD_0000536.vsi-20x_02.csv",
            "Features_SLD_0000560.vsi-20x_05.csv",
            "Features_SLD_0000533.vsi-20x_03.csv",
            "Features_SLD_0000563.vsi-20x_01.csv",
        ]
        train_image_names = [
            image for image in filenames if image not in test_image_names
        ]
    print(
        f"The training set contains {len(train_image_names)} images. {train_image_names}"
    )
    print(f"The test set contains {len(test_image_names)} images. {test_image_names}")

    # Train the model and optionally evaluate it.
    train_and_evaluate_model(
        train_dir=train_dir,
        save_dir=save_dir,
        features=features,
        train_image_names=train_image_names,
        gt_column=gt_column,
        extension=extension,
        distinguishable_second_layer=distinguishable_second_layer,
        estimators=estimators,
        clean_predictions=False,
        test_image_names=test_image_names,
        split_ratio=split_ratio,
        classes=classes,
        train_knn=train_knn,
    )
