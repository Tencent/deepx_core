#! /bin/bash
#
# Copyright 2020 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

cd $(dirname $0)
source env.sh

$TRAINER \
    --model=dcn \
    --model_config="config=libsvm_group_config.txt;deep_dims=64,32;cross=3" \
    --epoch=1 \
    --thread=1 \
    --in=libsvm.txt \
    --out_model=model.warmup
md5sum model.warmup/*

$TRAINER \
    --model=auto_int \
    --model_config="config=libsvm_group_config.txt;att_t=8;att_h=2;att_s=2" \
    --epoch=1 \
    --thread=1 \
    --in=libsvm.txt \
    --warmup_model=model.warmup \
    --out_model=model
md5sum model/*
