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
    --out_model=model.warmup
md5sum model.warmup/*

$TRAINER \
    --model=auto_int \
    --model_config="config=libsvm_group_config.txt;sparse=1;att_t=8;att_h=2;att_s=2" \
    --model_shard=4 \
    --epoch=1 \
    --thread=1 \
    --in=libsvm.txt \
    --warmup_model=model.warmup \
    --out_model=model
md5sum model/*
