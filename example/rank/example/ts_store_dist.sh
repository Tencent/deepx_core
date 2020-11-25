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
    --out_model=model \
    --ts_enable=1 --ts_now=10000 --ts_expire_threshold=10"
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

PREDICTOR_PARAM="--in=libsvm.txt \
    --in_model=model \
    --out_predict=model.predict"
$DIST_TRAINER \
    --sub_command=predict --role=ps --ps_id=0 \
    --cs_addr="127.0.0.1:61000" \
    --ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
    $PREDICTOR_PARAM > ps0.log 2>&1 &
$DIST_TRAINER \
    --sub_command=predict --role=ps --ps_id=1 \
    --cs_addr="127.0.0.1:61000" \
    --ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
    $PREDICTOR_PARAM > ps1.log 2>&1 &
$DIST_TRAINER \
    --sub_command=predict --role=ps --ps_id=2 \
    --cs_addr="127.0.0.1:61000" \
    --ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
    $PREDICTOR_PARAM > ps2.log 2>&1 &
$DIST_TRAINER \
    --sub_command=predict --role=ps --ps_id=3 \
    --cs_addr="127.0.0.1:61000" \
    --ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
    $PREDICTOR_PARAM > ps3.log 2>&1 &
$DIST_TRAINER \
    --sub_command=predict --role=wk \
    --cs_addr="127.0.0.1:61000" \
    --ps_addrs="127.0.0.1:60000;127.0.0.1:60001;127.0.0.1:60002;127.0.0.1:60003" \
    $PREDICTOR_PARAM > wk0.log 2>&1 &
wait
md5sum model.predict/*

$MERGE_MODEL_SHARD PREDICTOR \
    --in_model=model \
    --out_model=model.merge
md5sum model.merge

$EVAL_AUC --in=model.predict
