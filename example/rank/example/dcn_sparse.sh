#! /bin/bash
#
# Copyright 2020 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

cd $(dirname $0)
source env.sh

$TRAINER \
    --model=dcn \
    --model_config="config=libsvm_group_config.txt;sparse=1;deep_dims=64,32;cross=3" \
    --model_shard=4 \
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

$MERGE_MODEL_SHARD PREDICTOR \
    --in_model=model \
    --out_model=model.merge
md5sum model.merge

$EVAL_AUC --in=model.predict
