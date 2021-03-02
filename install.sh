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

cd $(dirname $0)
SOURCE_ROOT=$(pwd)
install=$SOURCE_ROOT/install

cd $SOURCE_ROOT/include
echo Installing headers to $PREFIX/include
for f in $(find . -type f -name "*.h"); do
    $install -m 0644 $f $PREFIX/include/$f
done

cd $SOURCE_ROOT/thirdparty
echo Installing thirdparty headers to $PREFIX/include
for f in $(find asio asio.hpp -type f); do
    $install -m 0644 $f $PREFIX/include/$f
done

for f in $(find feature -type f -name "*.h"); do
    $install -m 0644 $f $PREFIX/include/$f
done

for f in $(find gflags -type f -name "*.h"); do
    $install -m 0644 $f $PREFIX/include/$f
done

for f in $(find gtest -type f -name "*.h"); do
    $install -m 0644 $f $PREFIX/include/$f
done

for f in $(find hdfs_c.h); do
    $install -m 0644 $f $PREFIX/include/$f
done

for f in $(find lz4.h lz4frame.h lz4frame_static.h lz4hc.h xxhash.h); do
    $install -m 0644 $f $PREFIX/include/$f
done

for f in $(find zconf.h zlib.h); do
    $install -m 0644 $f $PREFIX/include/$f
done

cd $SOURCE_ROOT/script
echo Installing scripts to $PREFIX/script
for f in $(find compile_flags.sh); do
    $install -m 0755 $f $PREFIX/script/$f
done

cd $BUILD_DIR_ABS
echo Installing libraries to $PREFIX/lib
for f in $(find . -name "libdeepx_*.a"); do
    $install -m 0644 $f $PREFIX/lib/$f
done
