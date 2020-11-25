#! /bin/bash
#
# Copyright 2019 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

cd $(dirname $0)
source env.sh

$TRAINER \
    --model=deep_fm \
    --model_config="config=libsvm_group_config.txt;deep_dims=64,32" \
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
