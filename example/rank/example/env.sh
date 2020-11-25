#! /bin/bash
#
# Copyright 2019 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

export BUILD_DIR_ABS=$(cd ../../.. && make build_dir_abs)
export TRAINER=$BUILD_DIR_ABS/example/rank/trainer
export PREDICTOR=$BUILD_DIR_ABS/example/rank/predictor
export DIST_TRAINER=$BUILD_DIR_ABS/example/rank/dist_trainer
export EVAL_AUC=$BUILD_DIR_ABS/eval_auc
export MERGE_MODEL_SHARD=$BUILD_DIR_ABS/merge_model_shard
