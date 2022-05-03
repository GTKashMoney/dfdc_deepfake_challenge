#!/bin/bash

# Prediction
ROOT_DIR=$1

EPOCH="2"
RUN_NAME="end3"
python predict_folder.py --weights-dir $ROOT_DIR/weights/$RUN_NAME/ \
 --network DeepFakeClassifierWithSimpleEnD --models ${RUN_NAME}_DeepFakeClassifierWithSimpleEnD_tf_efficientnet_b7_ns_0_${EPOCH} \
 --test-dir $ROOT_DIR/test/ --output $ROOT_DIR/test_pred_${RUN_NAME}_${EPOCH}.csv

EPOCH="1"
RUN_NAME="end2"
python predict_folder.py --weights-dir $ROOT_DIR/weights/$RUN_NAME/ \
 --network DeepFakeClassifierWithSimpleEnD --models ${RUN_NAME}_DeepFakeClassifierWithSimpleEnD_tf_efficientnet_b7_ns_0_${EPOCH} \
 --test-dir $ROOT_DIR/test/ --output $ROOT_DIR/test_pred_${RUN_NAME}_${EPOCH}.csv