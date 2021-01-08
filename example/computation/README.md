# 张量计算

## [example1](example1_main.cc)

计算数学表达式"Z = X * W + B".

Z, X, W, B是标量.

输入X, W, B, 输出Z.

- 初始化计算图.

  ```c++
  // 构造计算图
  InstanceNode X("X", Shape(1), TENSOR_TYPE_TSR);
  InstanceNode W("W", Shape(1), TENSOR_TYPE_TSR);
  InstanceNode B("B", Shape(1), TENSOR_TYPE_TSR);
  MulNode XW("XW", &X, &W);
  AddNode Z("Z", &XW, &B);
  // 编译计算图
  DXCHECK_THROW(graph.Compile({&Z}, 0));
  ```

- 初始化OpContext对象.

  ```c++
  OpContext op_context;
  op_context.Init(&graph, &param);
  DXCHECK_THROW(op_context.InitOp(std::vector<int>{0}, -1));
  // 初始化X, W, B
  auto& _X = op_context.mutable_inst()->insert<tsr_t>(X.name());
  auto& _W = op_context.mutable_inst()->insert<tsr_t>(W.name());
  auto& _B = op_context.mutable_inst()->insert<tsr_t>(B.name());
  _X.resize(X.shape());
  _W.resize(W.shape());
  _B.resize(B.shape());
  // 初始化前向计算
  op_context.InitForward();
  ```

- 输入, 前向计算, 输出.

  ```c++
  // 输入
  _X.data(0) = x;
  _W.data(0) = w;
  _B.data(0) = b;
  // 前向计算
  op_context.Forward();
  // 输出
  const auto& _Z = op_context.hidden().get<tsr_t>(Z.name());
  float_t z = _Z.data(0);
  std::cout << "Z=" << z << std::endl;
  ```

## [example2](example2_main.cc)

计算数学表达式"Z = X * W + B".

Z, X, W, B是标量.

W, B是参数, 输入X, 输出Z.

- 初始化计算图.

  ```c++
  // 构造计算图
  InstanceNode X("X", Shape(1), TENSOR_TYPE_TSR);
  VariableNode W("W", Shape(1), TENSOR_TYPE_TSR);
  VariableNode B("B", Shape(1), TENSOR_TYPE_TSR);
  MulNode XW("XW", &X, &W);
  AddNode Z("Z", &XW, &B);
  // 编译计算图
  DXCHECK_THROW(graph.Compile({&Z}, 0));
  ```

- 初始化参数

  ```c++
  auto& _W = param.insert<tsr_t>(W.name());
  _W.resize(W.shape());
  // 硬编码. 生产环境中, 通常从文件或网络加载参数.
  _W.data(0) = 2;
  auto& _B = param.insert<tsr_t>(B.name());
  _B.resize(B.shape());
  _B.data(0) = 3;
  ```

- 初始化OpContext对象.

  类似example1.

- 输入, 前向计算, 输出.

  类似example1.

## [example3](example3_main.cc)

计算数学表达式"Z = X * W + B".

Z, X, W是矩阵, B是标量.

W, B是参数, 输入X, 输出Z.

- 初始化计算图.

  ```c++
  // 构造计算图
  InstanceNode X("X", Shape(2, 10), TENSOR_TYPE_TSR);
  VariableNode W("W", Shape(10, 1), TENSOR_TYPE_TSR);
  VariableNode B("B", Shape(1), TENSOR_TYPE_TSR);
  MatmulNode XW("XW", &X, &W);
  BroadcastAddNode Z("Z", &XW, &B);
  // 编译计算图
  DXCHECK_THROW(graph.Compile({&Z}, 0));
  ```

- 初始化参数

  类似example2.

- 初始化OpContext对象.

  类似example2.

- 输入, 前向计算, 输出.

  类似example2.

## [example4](example4_main.cc)

计算数学表达式"Z = X * W + B".

Z, X, W是矩阵, B是标量.

W, B是参数, 输入X, 输出Z.

- 初始化计算图.

  ```c++
  // 构造计算图
  // X的形状是(-1, 10), 表示(batch, 10).
  InstanceNode X("X", Shape(-1, 10), TENSOR_TYPE_TSR);
  VariableNode W("W", Shape(10, 1), TENSOR_TYPE_TSR);
  VariableNode B("B", Shape(1), TENSOR_TYPE_TSR);
  MatmulNode XW("XW", &X, &W);
  BroadcastAddNode Z("Z", &XW, &B);
  // 编译计算图
  DXCHECK_THROW(graph.Compile({&XW, &Z}, 0));
  ```

- 初始化参数

  类似example3.

- 初始化OpContext对象.

  类似example3.

- 输入, 前向计算, 输出.

  ```c++
  auto& _X = op_context.mutable_inst()->insert<tsr_t>(X.name());
  _X.resize(2 + i, 10);
  _X.randn(engine);
  // 只要X的行数变化, 就要重新调用OpContext::InitForward.
  op_context.InitForward();
  op_context.Forward();
  const auto& _XW = op_context.hidden().get<tsr_t>(XW.name());
  const auto& _Z = op_context.hidden().get<tsr_t>(Z.name());
  std::cout << "XW=" << _XW << std::endl;
  std::cout << "Z=" << _Z << std::endl;
  ```

## [example5](example5_main.cc)

使用以上所有特性, 实现self attention的前向计算.
