# Copyright 2020 the deepx authors.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

package(default_visibility=['//visibility:public'])

load('/mmsearchgateway/deepx_core/build_config', 'deepx_core_library',
     'deepx_core_user_library', 'deepx_core_user_binary')

deepx_core_library(name='deepx_core')

deepx_core_user_library(
    name='example_rank_model_server',
    srcs=['example/rank/model_server.cc'],
    hdrs=['example/rank/model_server.h'],
    alwayslink=True,
)

deepx_core_user_binary(
    name='feature_kv_demo',
    srcs=['src/tools/feature_kv_demo_main.cc'],
    copts=['-DHAVE_WXG_FEATURE_KV_CLIENT=1'],
    deps=[
        '//mm3rd/gflags:gflags',
        '//platform/kvsvr_feature/new/kvsvr:featurekvclient',
    ],
)
