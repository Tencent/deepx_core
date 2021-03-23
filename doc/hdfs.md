# hdfs

deepx使用libhdfs操作/读写hdfs文件系统.

## 获取libhdfs

推荐从[libhdfs(腾讯内部)](https://git.code.oa.com/mmrecommend/libhdfs)获取"libhdfs.so".

也可以从hadoop安装包获取"libhdfs.so".

## 使用libhdfs

获取"libhdfs.so"并和deepx程序一起发布.

参考以下例子设置环境变量.

```shell
CDH=$HOME/cdh-5.5.0
export JAVA_HOME=$CDH/jdk1.7.0_51
export JRE_HOME=$JAVA_HOME/jre
export HADOOP_HOME=$CDH/hadoop-2.6.0-cdh5.5.0
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
CLASSPATH=
HADOOP_CLASSPATH=
JARS=$(find $HADOOP_HOME/share -name "*.jar" -type f)
for j in $JARS
do
    CLASSPATH=$CLASSPATH:$j
    HADOOP_CLASSPATH=$HADOOP_CLASSPATH:$j
done
export CLASSPATH=$HADOOP_CONF_DIR:$CLASSPATH
export HADOOP_CLASSPATH=$HADOOP_CONF_DIR:$HADOOP_CLASSPATH
export PATH=$JAVA_HOME/bin:$HADOOP_HOME/bin:$PATH
export LIBRARY_PATH=$JRE_HOME/lib/amd64:$JRE_HOME/lib/amd64/server:$LIBRARY_PATH
export LD_LIBRARY_PATH=$JRE_HOME/lib/amd64:$JRE_HOME/lib/amd64/server:$LD_LIBRARY_PATH

# 设置hdfs用户名.
# 等价于"export HADOOP_USER_NAME=user"
export DEEPX_HDFS_USER=user

# 设置hdfs ugi.
# 等价于"hadoop fs -Dhadoop.job.ugi=user,group"
#
# 较低版本的"libhdfs.so"没有hdfsBuilder系列函数, 设置DEEPX_HDFS_UGI将导致运行异常.
# terminate called after throwing an instance of 'std::runtime_error'
#   what():  Get: hdfs builder functions were not loaded.
# 如果必须设置ugi, 请升级hadoop客户端和"libhdfs.so".
export DEEPX_HDFS_UGI=user,group
```

deepx程序启动时, 将尝试动态加载"libhdfs.so".

如果加载失败, 将看到以下日志.

```
libhdfs is unavailable.
```

请检查.

1. "libhdfs.so"是否可以被加载到.
2. 环境变量是否正确.

如果加载成功, 将看到以下日志.

```
Loaded libhdfs functions from ./libhdfs.so.
```
