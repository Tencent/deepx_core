#! /bin/bash
#
# Copyright 2020 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

cd $(dirname $0)
source env.sh

$TRAINER \
    --model=dtn \
    --model_config="user_group_config=libsvm_user_group_config.txt;item_group_config=libsvm_item_group_config.txt;user_deep_dims=64;item_deep_dims=64;deep_dims=64,32" \
    --epoch=1 \
    --thread=1 \
    --in=libsvm.txt \
    --out_model=model
md5sum model/*

$PREDICTOR \
    --thread=1 \
    --in=libsvm.txt \
    --in_model=model \
    --out_predict=model.predict
md5sum model.predict/*

$EVAL_AUC --in=model.predict
