#! /bin/bash
#
# Copyright 2019 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

cd $(dirname $0)
source env.sh

$TRAINER \
    --instance_reader=libsvm_ex \
    --instance_reader_config="x_size=2" \
    --model=pairwise_deep_fm \
    --model_config="config=pairwise_deep_fm_libsvm_ex_group_config.txt;deep_dims=64,32" \
    --epoch=1 \
    --thread=1 \
    --in=pairwise_deep_fm_libsvm_ex.txt \
    --out_model=model
md5sum model/*

$PREDICTOR \
    --instance_reader=libsvm \
    --thread=1 \
    --in=pairwise_deep_fm_libsvm.txt \
    --in_model=model \
    --out_predict=model.predict
md5sum model.predict/*
