#!/bin/bash

QUPATH_EXPERT_LAYERS_ANNOTATED_PROJECT_PATH="../ProjectQuPath_for_test/Ground_truth.qpproj"
QUPATH_PROJECT_PATH="../ProjectQuPath_for_test/ProjectQuPath_for_test.qpproj"
RESULT_PATH="/arch_results"
FIGURE_PATH="/arch_results/Figures"

FOR_PREDICTION_PATH="/arch_results/prediction"
FOR_TRAINING_PATH="/arch_results/training"
MODEL_PATH="/arch_results/training/model"

printf "\nPIPELINE INFO: QuPath: Detect cells, export annotations and cells features\n"
#qupath script ./qupath_scripts/full_QuPath_script.groovy -p $QUPATH_PROJECT_PATH

printf "\nPIPELINE INFO: QuPath: Detect cells, export annotations and cells features for the Ground Truth images\n"
#qupath script ./qupath_scripts/full_QuPath_script.groovy -p $QUPATH_EXPERT_LAYERS_ANNOTATED_PROJECT_PATH

printf "\nPIPELINE INFO: Convert cells features and annotation to pandas dataframes\n"
pylayer_recognition convert --config-file-path ./Config/batch_convert.ini

printf "\nPIPELINE INFO: Convert QuPath project metadata to a pandas dataframes\n"
pylayer_recognition convert-qupath-project --qupath-project-path $QUPATH_PROJECT_PATH --output-path $RESULT_PATH

printf "\nPIPELINE INFO: Compute cell densities as function of brain depth\n"
pylayer_recognition density-per-depth --config-file-path ./Config/batch_density_depth.ini

printf "\nPIPELINE INFO: Train a Machine Learning model\n"
pylayer_recognition -v train-model --train-dir $FOR_TRAINING_PATH --train-glob "Feat*" --extension csv  --save-dir $MODEL_PATH --distinguishable-second-layer

printf "\nPIPELINE INFO: Predict the cells layer\n"
pylayer_recognition -v  layers-predict --model-file $MODEL_PATH/trained_rf.pkl  --pred-dir $FOR_PREDICTION_PATH --pred-save  $RESULT_PATH  --pred-glob "Feat*" --distinguishable-second-layer

printf "\nPIPELINE INFO: Compute cell densities per Layer with L2 and L3 merged\n"
pylayer_recognition density-per-layer --config-file-path ./Config/batch_density_layer_merged.ini

printf "\nPIPELINE INFO: Compute cell densities per Layer with L2 and L3 distinguished\n"
pylayer_recognition density-per-layer --config-file-path ./Config/batch_density_layer_distinguish.ini

printf "\nPIPELINE INFO: Concatenate Dataframe for Cell area\n"
pylayer_recognition cell-size --config-file-path Config/batch_size.ini

printf "\nPIPELINE INFO: Generate dataframe for the layer thinkness\n"
pylayer_recognition layer-thickness --feature-file-path /arch_results/converted/predicted/distinguish --output-filename $RESULT_PATH/layer_thickness.csv


printf "\nPIPELINE INFO: Produces cell densities and cell sizes figures\n"
python figures_script/cells_density.py $RESULT_PATH/output_path_batch/PerDepth $RESULT_PATH/output_path_batch/merged/ $RESULT_PATH/output_path_batch/distinguish/ $FIGURE_PATH ./data/metadata.csv
python figures_script/cells_size.py /arch_results/output_path_batch/cell_area/cells_area.csv /tmp/cell_size.png per_layer
python figures_script/layer_thickness.py $RESULT_PATH/layer_thickness.csv $FIGURE_PATH/layers_thickness.svg
printf "\n----- Done ------"
