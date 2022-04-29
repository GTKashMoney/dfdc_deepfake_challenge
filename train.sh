#!/bin/bash

ROOT_DIR=$1
NUM_GPUS=$2
# python -u -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 9901 training/pipelines/train_classifier.py \
#  --distributed --config configs/b7.json --freeze-epochs 0 --test_every 1 --opt-level O1 --label-smoothing 0.01 --folds-csv folds.csv   --fold 0 --seed 111 --data-dir $ROOT_DIR --prefix b7_111_ > logs/b7_111

# python -u -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 9901 training/pipelines/train_classifier.py \
#  --distributed --config configs/b7.json --freeze-epochs 0 --test_every 1 --opt-level O1 --label-smoothing 0.01 --folds-csv folds.csv  --fold 0 --seed 555 --data-dir $ROOT_DIR --prefix b7_555_ > logs/b7_555

# python -u -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 9901 training/pipelines/train_classifier.py \
#  --distributed --config configs/b7.json --freeze-epochs 0 --test_every 1 --opt-level O1 --label-smoothing 0.01 --folds-csv folds.csv  --fold 0 --seed 777 --data-dir $ROOT_DIR --prefix b7_777_ > logs/b7_777

# python -u -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 9901 training/pipelines/train_classifier.py \
#  --distributed --config configs/b7.json --freeze-epochs 0 --test_every 1 --opt-level O1 --label-smoothing 0.01 --folds-csv folds.csv  --fold 0 --seed 888 --data-dir $ROOT_DIR --prefix b7_888_ > logs/b7_888

RUN_NAME="end1"
python -u -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
 --master_port 9901 training/pipelines/train_classifier.py --distributed \
 --config configs/simple_end_b7.json --freeze-epochs 0 --test_every 1 \
 --opt-level O1 --label-smoothing 0.01 --resume $ROOT_DIR/weights/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23 \
 --folds-csv $ROOT_DIR/folds.csv  --fold 0 --seed 999 --from-zero \
 --output-dir $ROOT_DIR/weights/$RUN_NAME/ --data-dir $ROOT_DIR \
 --logdir $ROOT_DIR/logs/$RUN_NAME/ --prefix ${RUN_NAME}_ > logs/$RUN_NAME