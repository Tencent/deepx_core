#! /bin/bash
#
# Copyright 2020 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

cd $(dirname $0)
source env.sh

TRAINER_PARAM="--model=dcn \
    --model_config=config=libsvm_group_config.txt;sparse=1;deep_dims=64,32;cross=3 \
    --epoch=1 \
    --in=libsvm.txt \
    --out_model=model.warmup"
$DIST_TRAINER \
    --sub_command=train --role=ps --ps_id=0 \
    --cs_addr="127.0.0.1:61000" \
    --ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
    $TRAINER_PARAM > ps0.log 2>&1 &
$DIST_TRAINER \
    --sub_command=train --role=ps --ps_id=1 \
    --cs_addr="127.0.0.1:61000" \
    --ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
    $TRAINER_PARAM > ps1.log 2>&1 &
$DIST_TRAINER \
    --sub_command=train --role=ps --ps_id=2 \
    --cs_addr="127.0.0.1:61000" \
    --ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
    $TRAINER_PARAM > ps2.log 2>&1 &
$DIST_TRAINER \
    --sub_command=train --role=ps --ps_id=3 \
    --cs_addr="127.0.0.1:61000" \
    --ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
    $TRAINER_PARAM > ps3.log 2>&1 &
$DIST_TRAINER \
    --sub_command=train --role=wk \
    --cs_addr="127.0.0.1:61000" \
    --ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
    $TRAINER_PARAM > wk0.log 2>&1 &
wait
md5sum model.warmup/*

TRAINER_PARAM="--model=auto_int \
    --model_config=config=libsvm_group_config.txt;sparse=1;att_t=8;att_h=2;att_s=2 \
    --epoch=1 \
    --in=libsvm.txt \
    --warmup_model=model.warmup \
    --out_model=model"
$DIST_TRAINER \
    --sub_command=train --role=ps --ps_id=0 \
    --cs_addr="127.0.0.1:61000" \
    --ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
    $TRAINER_PARAM > ps0.log 2>&1 &
$DIST_TRAINER \
    --sub_command=train --role=ps --ps_id=1 \
    --cs_addr="127.0.0.1:61000" \
    --ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
    $TRAINER_PARAM > ps1.log 2>&1 &
$DIST_TRAINER \
    --sub_command=train --role=ps --ps_id=2 \
    --cs_addr="127.0.0.1:61000" \
    --ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
    $TRAINER_PARAM > ps2.log 2>&1 &
$DIST_TRAINER \
    --sub_command=train --role=ps --ps_id=3 \
    --cs_addr="127.0.0.1:61000" \
    --ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
    $TRAINER_PARAM > ps3.log 2>&1 &
$DIST_TRAINER \
    --sub_command=train --role=wk \
    --cs_addr="127.0.0.1:61000" \
    --ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
    $TRAINER_PARAM > wk0.log 2>&1 &
wait
md5sum model/*
