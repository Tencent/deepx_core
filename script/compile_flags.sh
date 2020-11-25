#! /bin/bash
#
# Copyright 2019 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

git=$(which git 2>/dev/null || true)

function get_git_hash() {
    cd $1
    if test "x$git" == x; then
        local git_hash=unknown
    else
        local git_hash=$($git rev-parse --short HEAD 2>/dev/null || true)
        if test "x$git_hash" == x; then
            local git_hash=unknown
        fi
    fi
    echo $git_hash
    cd - 1>/dev/null
}

function get_git_dirty() {
    cd $1
    if test "x$git" == x; then
        local git_dirty=unknown
    else
        local git_diff=$($git diff --name-only --diff-filter=ACM 2>/dev/null || true)
        if test "x$git_diff" == x; then
            local git_dirty=0
        else
            local git_dirty=1
        fi
    fi
    echo $git_dirty
    cd - 1>/dev/null
}

echo "\"git hash: $(get_git_hash .)\\n\""
echo "\"git dirty: $(get_git_dirty .)\\n\""
if test -f .gitmodules; then
    submodules=$(awk '/path = /{print $3}' .gitmodules)
    for submodule in $submodules; do
        echo "\"$submodule git hash: $(get_git_hash $submodule)\\n\""
        echo "\"$submodule git dirty: $(get_git_dirty $submodule)\\n\""
    done
fi

until [ $# -lt 2 ]; do
    echo "\"$1: $2\\n\""
    shift 2
done
