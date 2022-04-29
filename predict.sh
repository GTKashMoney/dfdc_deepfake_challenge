#!/bin/bash

# Prediction
ROOT_DIR=$1
EPOCH=$2

RUN_NAME="end1"
python predict_folder.py --weights-dir $ROOT_DIR/weights/$RUN_NAME/ \
 --network DeepFakeClassifierWithSimpleEnD --models b7_999_DeepFakeClassifierWithSimpleEnD_tf_efficientnet_b7_ns_0_$EPOCH \
 --test-dir $ROOT_DIR/test/ --output $ROOT_DIR/test_pred_${RUN_NAME}.csv