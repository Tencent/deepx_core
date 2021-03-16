# Copyright 2020 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#
# only for mmsearchgateway
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


def deepx_core_library():
  copts = _default_copts() + [
      '-Immsearchgateway/deepx_core/thirdparty',
  ]

  deps = [
      '//mmsearchgateway/sage2:sage2',
      '//mm3rd/lz4:lz4',
      '//mm3rd/zlib:z',
      '//comm2/core:core',
  ]

  native.cc_library(
      name='deepx_core',
      includes=['include'],
      hdrs=native.glob(['include/**/*.h']),
      srcs=native.glob(
          ['src/**/*.cc'],
          exclude=[
              'src/**/*_test.cc',
              'src/**/*_main.cc',
          ],
      ),
      copts=copts,
      linkopts=['-ldl'],
      deps=deps,
      visibility=['//visibility:public'],
      alwayslink=True,
  )


def deepx_core_user_library(name,
                            includes=[],
                            hdrs=[],
                            srcs=[],
                            copts=[],
                            linkopts=[],
                            deps=[],
                            visibility=['//visibility:public'],
                            alwayslink=False):
  for copt in _default_copts():
    if copt not in copts:
      copts = copts + [copt]

  deps = deps + ['//mmsearchgateway/deepx_core:deepx_core']

  native.cc_library(
      name=name,
      includes=includes,
      hdrs=hdrs,
      srcs=srcs,
      copts=copts,
      linkopts=linkopts,
      deps=deps,
      visibility=visibility,
      alwayslink=alwayslink,
  )


def deepx_core_user_binary(name, srcs=[], copts=[], linkopts=[], deps=[]):
  for copt in _default_copts():
    if copt not in copts:
      copts = copts + [copt]

  deps = deps + ['//mmsearchgateway/deepx_core:deepx_core']

  native.cc_binary(
      name=name,
      srcs=srcs,
      copts=copts,
      linkopts=linkopts,
      deps=deps,
  )
