# Copyright 2020 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#


def _default_copts():
  return [
      '-std=c++11',
      '-Wall',
      '-Wextra',
      # '-Werror',
      '-pedantic',
      '-Wno-float-equal',
      '-DOS_LINUX=1',
      '-DOS_POSIX=1',
      '-DNDEBUG',
      '-DHAVE_SAGE2=1',
      '-DHAVE_SAGE2_SGEMM=1',
      # '-DHAVE_SAGE2_SGEMM_JIT=1',
      '-DHAVE_WXG_LOG=1',
  ]


def deepx_core_library(name):
  copts = _default_copts() + [
      '-Immsearchgateway/deepx_core/thirdparty',
      '-Wno-array-bounds',
  ]

  deps = [
      '//mm3rd/lz4-1.8.2:lz4',
      '//mm3rd/zlib-1.2.3:z',
      '//mmsearchgateway/sage2:sage2',
      '//comm2/core:core',
  ]

  native.cc_library(
      name=name,
      srcs=native.glob(
          [
              'src/**/*.cc',
          ],
          exclude=[
              'src/**/*_test.cc',
              'src/**/*_main.cc',
          ],
      ),
      hdrs=native.glob([
          'include/**/*.h',
      ]),
      includes=[
          'include',
      ],
      copts=copts,
      deps=deps,
      alwayslink=True,
  )


def deepx_core_user_library(name,
                            srcs=[],
                            hdrs=[],
                            includes=[],
                            copts=[],
                            deps=[],
                            alwayslink=False):
  for copt in _default_copts():
    if copt not in copts:
      copts = copts + [copt]

  deps = deps + [':deepx_core']

  native.cc_library(
      name=name,
      srcs=srcs,
      hdrs=hdrs,
      includes=includes,
      copts=copts,
      deps=deps,
      alwayslink=alwayslink,
  )


def deepx_core_user_binary(name, srcs=[], copts=[], deps=[]):
  for copt in _default_copts():
    if copt not in copts:
      copts = copts + [copt]

  deps = deps + [':deepx_core']

  native.cc_binary(
      name=name,
      srcs=srcs,
      copts=copts,
      deps=deps,
  )
