"""Main script to train, evaluate and predict with cell prediction models."""

import argparse
import logging
import os
import pathlib
import pickle
import warnings

import tqdm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from layer_recognition.ml.utils import (
    clean_predictions,
    get_image_files,
    image_to_df,
    plot_eval_metrics,
    plot_results,
)

logger = logging.getLogger(__name__)


def train_and_evaluate_model(
    train_dir: pathlib.Path,
    save_dir: pathlib.Path,
    features: list[str],
    train_image_names: list[str],
    gt_column: str = "Expert_layer",
    extension: str = "csv",
    distinguishable_second_layer: bool = True,
    estimators: int = 100,
    clean_predictions: bool = False,
    eps: float = 0.05,
    min_samples: int = 3,
    test_image_names: list[str] | None = None,
    split_ratio: float = 0.1,
    classes: list[str] | None = None,
    show: bool = False,
    train_knn: bool = False,
    neighbors: int = 32,
) -> RandomForestClassifier:
    """Train the random forest classifier, and optionally compute its accuracy. Optionally train a KNN classifier as well for comparison only.

    Parameters
    ----------
    train_dir
        Directory containing the training files
    save_dir
        Path to the directory where to save metrics and images
    features
        List of features to use for training
    train_image_names
        List of filenames used for training
    gt_column
        Name of the column containing the ground truth in the csv
    extension
        File extension of the training files
    distinguishable_second_layer
        Use Layer 2/3 if False, Layer 2 and Layer 3 if True
    estimators
        Number of estimators of the RF model
    clean_predictions
        Apply post-processing algorithm if True
    eps
        Radius of the circle for DBSCAN algorithm
    min_samples
        Number of neighbors that must be contained within eps circle in DBSCAN
    test_image_names
        List of filenames used for testing
    split_ratio
        Ratio of train/test images
    show
        Display figures after testing if True
    train_knn
        Flag to train a KNN or not.
    neighbors
        KNN neighbors parameter.

    Returns
    -------
    The fitted random forest classifier.
    """
    logger.info("Reading training images")
    x_train, y_train, _ = image_to_df(
        image_names=train_image_names,
        data_path=train_dir,
        classes=classes,
        features=features,
        filter=True,
        distinguish_second_layer=distinguishable_second_layer,
        extension=extension,
        gt_column=gt_column,
    )
    logger.info(f"Training dataset shape : {x_train.shape}")
    rf = RandomForestClassifier(n_estimators=estimators, random_state=42)
    logger.info("Training the random forest...")
    rf.fit(x_train, y_train)

    pickle.dump(rf, open(save_dir / "trained_rf.pkl", "wb"))

    if split_ratio > 0 and test_image_names and classes:
        models = [(rf, "Random Forest Classifier")]

        if train_knn:
            logger.info("Training the knn...")
            knn = KNeighborsClassifier(n_neighbors=neighbors)
            knn.fit(x_train, y_train)
            models.append((knn, "K Nearest Neighbor"))

        test_images = []
        predictions = []
        for image_name in tqdm.tqdm(
            test_image_names,
            desc="Evaluating on test images",
            total=len(test_image_names),
        ):
            ground_truth, prediction = plot_results(
                image_name=image_name,
                data_path=train_dir,
                save_path=save_dir,
                classes=classes,
                models=models,
                clean=clean_predictions,
                eps=eps,
                min_samples=min_samples,
                features=features,
                distinguish_second_layer=distinguishable_second_layer,
                extension=extension,
                gt_column=gt_column,
                show=show,
            )
            test_images.append(ground_truth)
            predictions.append(prediction)
            for pred, (_, model_name) in zip(prediction, models):
                class_scores = metrics.precision_recall_fscore_support(
                    ground_truth, pred
                )
                confusion_matrix = metrics.confusion_matrix(ground_truth, pred)
                save_path = (
                    os.path.join(
                        save_dir,
                        pathlib.Path(image_name).stem
                        + f"_{'_'.join(model_name.split(' '))}",
                    )
                    + "_metrics"
                )
                per_class_accuracy = confusion_matrix.diagonal() / confusion_matrix.sum(
                    axis=1
                )
                plot_eval_metrics(
                    class_scores,
                    per_class_accuracy,
                    confusion_matrix,
                    classes,
                    model_name,
                    save_path=save_path,
                    show=show,
                )
        test_images = [label for sub_list in test_images for label in sub_list]
        rf_predictions = [pred[0] for pred in predictions]
        flattened_rf_predictions = [
            label for image in rf_predictions for label in image
        ]
        print(
            "RF Accuracy:",
            metrics.accuracy_score(test_images, flattened_rf_predictions),
        )

        if train_knn:
            knn_predictions = [pred[1] for pred in predictions]
            flattened_knn_predictions = [
                label for image in knn_predictions for label in image
            ]
            print(
                "KNN Accuracy:",
                metrics.accuracy_score(test_images, flattened_knn_predictions),
            )

    return rf


def predict(
    pred_dir: pathlib.Path,
    model: RandomForestClassifier,
    features: list[str],
    classes: list[str],
    pred_save: pathlib.Path | None = None,
    pred_glob: str = "*.csv",
    distinguishable_second_layer: bool = True,
    pred_extension: str = "csv",
    gt_column: str = "Expert_layer",
    clean: bool = False,
    eps: float = 0.05,
    min_samples: int = 3,
) -> None:
    """Predict cell classes on non annotated images.

    Parameters
    ----------
    pred_dir
        Directory where the data to predict is located.
    model
        Trained random forest classifier.
    features
        List of data feature kept.
    classes
        List of classes the model should output.
    pred_save
        Path to the directory where to save the predictions.
    pred_glob
        Glob pattern to select only certain files in pred_dir
    distinguishable_second_layer
        Whether to keep layer 2 and 3 separated.
    pred_extension
        Extension of the pred files.
    gt_column
        Name of the column carrying the ground truth in training data that was used for the model
    clean_predictions
        Whether to apply post-processing.
    eps
        Radius of the circle used in DBSCAN for post-processing.
    min_samples
        Minimum number of samples to find to define a central point in DBSCAN.
    """
    print('DEBUG enter predict function')
    import pandas as pd
    logger.info("Predicting on un-annotated images.")
    image_names = get_image_files(pred_dir, pred_glob)
    for image in tqdm.tqdm(image_names, total=len(image_names)):
        print(f'DEBUG  predict function image {image}')
        df = pd.read_csv(str(pred_dir) + '/' +  image)
        print(f'DEBUG  len(df) {len(df)}')
        x, _, detection_df = image_to_df( 
            [image],
            pred_dir,
            classes=classes,
            features=features,
            filter=True,
            distinguish_second_layer=distinguishable_second_layer,
            extension=pred_extension,
            gt_column=gt_column,
        )
        print(f'DEBUG  len(x) {len(x)}')
        try:
            if clean:
                predictions = clean_predictions(
                    detection_df.copy(),
                    model.predict(x),
                    eps=eps,
                    min_samples=min_samples,
                    gt_column=gt_column,
                )
            else:
                print('DEBUG START prediction')
                predictions = model.predict(x)
                print('DEBUG END prediction')
        except ValueError as e:
            print(f"ERROR: {e}")
            raise (SystemExit(-1))
        to_remove = [col for col in detection_df.columns if "Unnamed" in col]
        detection_df["RF_prediction"] = predictions
        detection_df.drop(
            columns=to_remove,
            inplace=True,
        )
        if not pred_save:
            warnings.warn("--pred-save not set. Not saving CSVs")
        else:
            if not os.path.exists(pred_save):
                os.makedirs(pred_save, exist_ok=True)
            print(f"Saved prediction to {os.path.join(pred_save, image)} ")
            detection_df.to_csv(
                os.path.join(pred_save, image),
                index=False,
            )
            print(f"INFO: prediction saved to {os.path.join(pred_save, image)}")
    print(f"predict Done ")
