#! /bin/bash
#
# Copyright 2019 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

if test "x$1" == x; then
    exit 1
fi
if test "x$2" == x; then
    exit 1
fi
PREFIX=$1
BUILD_DIR_ABS=$2

set -e
cd $(dirname $0)
pwd=$(pwd)

SOURCE_ROOT=$pwd/..
install=$pwd/install

cd $SOURCE_ROOT/include
echo Installing headers to $PREFIX/include
for f in $(find . -type f -name "*.h"); do
    $install -m 0644 $f $PREFIX/include/$f
done

cd $BUILD_DIR_ABS
echo Installing libraries to $PREFIX/lib
for f in $(find libsage2.a); do
    $install -m 0644 $f $PREFIX/lib/$f
done
