# hdfs

deepx支持操作/读写hdfs文件系统.

## 获取libhdfs

推荐从[libhdfs](https://git.code.oa.com/mmrecommend/libhdfs)获取"libhdfs.so".

也可以从hdfs安装包获取"libhdfs.so".

## 使用libhdfs

deepx程序动态加载"libhdfs.so", deepx程序要和"libhdfs.so"一起发布并放在同一目录.

deepx程序运行前参考"[env.sh](https://git.code.oa.com/mmrecommend/libhdfs/blob/master/env.sh)"导出环境变量.

### 设置hdfs ugi

```shell
export DEEPX_HDFS_UGI=user,group
# 等价于hadoop fs -Dhadoop.job.ugi=user,group
```

### 设置hdfs用户名

```shell
export DEEPX_HDFS_USER=user
```

或者.

```shell
export HADOOP_USER_NAME=user
```

### 失败

如果加载失败, 会看到如下日志.

```
libhdfs is unavailable.
```

此时, 可以使用["load\_hdfs\_so.py"](https://git.code.oa.com/mmrecommend/libhdfs/blob/master/tools/load_hdfs_so.py)来定位库的加载问题.

### 成功

如果加载成功, 会看到如下日志.

```
Loaded libhdfs functions from ./libhdfs.so.
```

此后, ["stream.h"](../include/deepx_core/common/stream.h)中的AutoFileSystem, AutoInputFileStream, AutoOutputFileStream可以操作以"hdfs://"开头的路径.

deepx中, 绝大多数的文件操作使用这3个类, 这些工具便具备了操作/读写hdfs的能力.
