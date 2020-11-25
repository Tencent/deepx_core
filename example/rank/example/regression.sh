#! /bin/bash
#
# Copyright 2020 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

for sh in $(find -name "*.sh" -and -not -name "env.sh" -and -not -name "regression.sh" | sort); do
    echo $sh
    bash -e $sh 2>/dev/null
    rm -rf model* *.log
done
