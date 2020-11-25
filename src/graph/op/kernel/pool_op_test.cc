// Copyright 2019 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class PoolForwardTest : public testing::Test, public DataType {
 protected:
  struct TestCase {
    int pool_rank = 0;
    Shape Xshape, Zshape;
    int data_format = 0;
    std::vector<int> kernel_sizes;
    std::vector<int> strides;
    std::vector<int> dilations;
    int padding_mode = 0;
    std::vector<int> paddings;
    int ceil_mode = 0;
    int count_include_pad = 0;
    std::vector<float_t> Xvalues, Zvalues;
    tsr_t X, Z;

   public:
    void ParseFrom(int _pool_rank, const std::string& param,
                   const std::string& Xline, const std::string& Zline) {
      pool_rank = _pool_rank;
      std::vector<int> Xdims(pool_rank + 2);
      std::vector<int> Zdims(pool_rank + 2);
      std::istringstream iss;
      iss.str(param);

      for (int& Xdim : Xdims) {
        iss >> Xdim;
      }
      ASSERT_TRUE(iss);
      Xshape.assign(Xdims.begin(), Xdims.end());

      for (int& Zdim : Zdims) {
        iss >> Zdim;
      }
      ASSERT_TRUE(iss);
      Zshape.assign(Zdims.begin(), Zdims.end());

      iss >> data_format;
      ASSERT_TRUE(iss);

      kernel_sizes.resize(pool_rank);
      for (int& kernel_size : kernel_sizes) {
        iss >> kernel_size;
      }
      ASSERT_TRUE(iss);

      strides.resize(pool_rank);
      for (int& stride : strides) {
        iss >> stride;
      }
      ASSERT_TRUE(iss);

      dilations.resize(pool_rank);
      for (int& dilation : dilations) {
        iss >> dilation;
      }
      ASSERT_TRUE(iss);

      iss >> padding_mode;
      ASSERT_TRUE(iss);

      paddings.resize(pool_rank);
      for (int& padding : paddings) {
        iss >> padding;
      }
      ASSERT_TRUE(iss);

      iss >> ceil_mode;
      ASSERT_TRUE(iss);

      iss >> count_include_pad;
      ASSERT_TRUE(iss);

      ASSERT_TRUE(Split(Xline, " ", &Xvalues));
      ASSERT_EQ(Xshape.total_dim(), (int)Xvalues.size());
      X.view(Xshape, Xvalues.data());

      ASSERT_TRUE(Split(Zline, " ", &Zvalues));
      ASSERT_EQ(Zshape.total_dim(), (int)Zvalues.size());
      Z.view(Zshape, Zvalues.data());
    }
  };

 protected:
  void LoadTestCase(int pool_rank, const std::string& file,
                    std::vector<TestCase>* test_cases) {
    test_cases->clear();

    std::ifstream is(file);
    ASSERT_TRUE(is.is_open());

    std::string param, Xline, Zline;
    while (std::getline(is, param)) {
      ASSERT_TRUE(std::getline(is, Xline));
      ASSERT_TRUE(std::getline(is, Zline));
      TestCase test_case;
      test_case.ParseFrom(pool_rank, param, Xline, Zline);
      test_cases->emplace_back(std::move(test_case));
    }
    is.close();
  }

  void Test(const std::vector<TestCase>& test_cases, int pool_type) {
    for (const auto& test_case : test_cases) {
      auto* X = new VariableNode("X", test_case.Xshape);
      GraphNode* Z;
      if (pool_type == GraphNodePoolBase::POOL_TYPE_MAX) {
        if (test_case.pool_rank == 1) {
          Z = new MaxPool1dNode("Z", X, test_case.data_format,
                                test_case.kernel_sizes[0], test_case.strides[0],
                                test_case.dilations[0], test_case.padding_mode,
                                test_case.paddings[0], test_case.ceil_mode);
        } else if (test_case.pool_rank == 2) {
          Z = new MaxPool2dNode("Z", X, test_case.data_format,
                                test_case.kernel_sizes, test_case.strides,
                                test_case.dilations, test_case.padding_mode,
                                test_case.paddings, test_case.ceil_mode);
        } else {
          Z = new MaxPool3dNode("Z", X, test_case.data_format,
                                test_case.kernel_sizes, test_case.strides,
                                test_case.dilations, test_case.padding_mode,
                                test_case.paddings, test_case.ceil_mode);
        }
      } else {
        if (test_case.pool_rank == 1) {
          Z = new AvgPool1dNode("Z", X, test_case.data_format,
                                test_case.kernel_sizes[0], test_case.strides[0],
                                test_case.padding_mode, test_case.paddings[0],
                                test_case.ceil_mode,
                                test_case.count_include_pad);
        } else if (test_case.pool_rank == 2) {
          Z = new AvgPool2dNode(
              "Z", X, test_case.data_format, test_case.kernel_sizes,
              test_case.strides, test_case.padding_mode, test_case.paddings,
              test_case.ceil_mode, test_case.count_include_pad);
        } else {
          Z = new AvgPool3dNode(
              "Z", X, test_case.data_format, test_case.kernel_sizes,
              test_case.strides, test_case.padding_mode, test_case.paddings,
              test_case.ceil_mode, test_case.count_include_pad);
        }
      }
      auto post_param_initializer = [&test_case](
                                        std::default_random_engine /*engine*/,
                                        TensorMap* param) {
        param->get<tsr_t>("X") = test_case.X;
      };
      CheckOpForward(Z, 1, test_case.Z, nullptr, post_param_initializer,
                     nullptr);
    }
  }
};

TEST_F(PoolForwardTest, MaxPool1d) {
  std::vector<TestCase> test_cases;
  LoadTestCase(1, "testdata/graph/op/kernel/max_pool1d.txt", &test_cases);
  Test(test_cases, GraphNodePoolBase::POOL_TYPE_MAX);
}

TEST_F(PoolForwardTest, MaxPool2d) {
  std::vector<TestCase> test_cases;
  LoadTestCase(2, "testdata/graph/op/kernel/max_pool2d.txt", &test_cases);
  Test(test_cases, GraphNodePoolBase::POOL_TYPE_MAX);
}

TEST_F(PoolForwardTest, MaxPool3d) {
  std::vector<TestCase> test_cases;
  LoadTestCase(3, "testdata/graph/op/kernel/max_pool3d.txt", &test_cases);
  Test(test_cases, GraphNodePoolBase::POOL_TYPE_MAX);
}

TEST_F(PoolForwardTest, AvgPool1d) {
  std::vector<TestCase> test_cases;
  LoadTestCase(1, "testdata/graph/op/kernel/avg_pool1d.txt", &test_cases);
  Test(test_cases, GraphNodePoolBase::POOL_TYPE_AVG);
}

TEST_F(PoolForwardTest, AvgPool2d) {
  std::vector<TestCase> test_cases;
  LoadTestCase(2, "testdata/graph/op/kernel/avg_pool2d.txt", &test_cases);
  Test(test_cases, GraphNodePoolBase::POOL_TYPE_AVG);
}

TEST_F(PoolForwardTest, AvgPool3d) {
  std::vector<TestCase> test_cases;
  LoadTestCase(3, "testdata/graph/op/kernel/avg_pool3d.txt", &test_cases);
  Test(test_cases, GraphNodePoolBase::POOL_TYPE_AVG);
}

class AvgPoolBackwardTest : public testing::Test {
 protected:
  struct TestCase {
    int pool_rank;
    Shape Xshape;
    int data_format;
    std::vector<int> kernel_sizes;
    std::vector<int> strides;
    std::vector<int> paddings;
  };
  const std::vector<TestCase> AVG_POOL1D_TEST_CASES = {
      {1, Shape(4, 2, 11), GraphNodePoolBase::DATA_FORMAT_NCW, {3}, {2}, {1}},
      {1, Shape(4, 11, 2), GraphNodePoolBase::DATA_FORMAT_NWC, {3}, {2}, {1}}};
  const std::vector<TestCase> AVG_POOL2D_TEST_CASES = {
      {2,
       Shape(2, 3, 10, 20),
       GraphNodePoolBase::DATA_FORMAT_NCHW,
       {4, 6},
       {2, 3},
       {2, 3}},
      {2,
       Shape(2, 10, 20, 3),
       GraphNodePoolBase::DATA_FORMAT_NHWC,
       {4, 6},
       {2, 3},
       {2, 3}}};
  const std::vector<TestCase> AVG_POOL3D_TEST_CASES = {
      {3,
       Shape(2, 2, 6, 10, 9),
       GraphNodePoolBase::DATA_FORMAT_NCDHW,
       {3, 4, 4},
       {2, 1, 3},
       {1, 2, 2}},
      {3,
       Shape(2, 6, 10, 9, 2),
       GraphNodePoolBase::DATA_FORMAT_NDHWC,
       {3, 4, 4},
       {2, 1, 3},
       {1, 2, 2}}};

 protected:
  void Test(const std::vector<TestCase>& test_cases, int padding_mode,
            int ceil_mode = 0, int count_include_pad = 0) {
    for (const auto& test_case : test_cases) {
      auto* X = new VariableNode("X", test_case.Xshape,
                                 TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
      GraphNode* Z = nullptr;
      if (test_case.pool_rank == 1) {
        Z = new AvgPool1dNode("Z", X, test_case.data_format,
                              test_case.kernel_sizes[0], test_case.strides[0],
                              padding_mode, test_case.paddings[0], ceil_mode,
                              count_include_pad);
      } else if (test_case.pool_rank == 2) {
        Z = new AvgPool2dNode("Z", X, test_case.data_format,
                              test_case.kernel_sizes, test_case.strides,
                              padding_mode, test_case.paddings, ceil_mode,
                              count_include_pad);
      } else {
        Z = new AvgPool3dNode("Z", X, test_case.data_format,
                              test_case.kernel_sizes, test_case.strides,
                              padding_mode, test_case.paddings, ceil_mode,
                              count_include_pad);
      }
      CheckOpBackward(Z, 1);
    }
  }
};

TEST_F(AvgPoolBackwardTest, AvgPool1d) {
  Test(AVG_POOL1D_TEST_CASES, GraphNodePoolBase::PADDING_MODE_SAME);
  Test(AVG_POOL1D_TEST_CASES, GraphNodePoolBase::PADDING_MODE_VALID);
  Test(AVG_POOL1D_TEST_CASES, GraphNodePoolBase::PADDING_MODE_USE_PADDINGS);
  Test(AVG_POOL1D_TEST_CASES, GraphNodePoolBase::PADDING_MODE_USE_PADDINGS, 1,
       1);
}

TEST_F(AvgPoolBackwardTest, AvgPool2d) {
  Test(AVG_POOL2D_TEST_CASES, GraphNodePoolBase::PADDING_MODE_SAME);
  Test(AVG_POOL2D_TEST_CASES, GraphNodePoolBase::PADDING_MODE_VALID);
  Test(AVG_POOL2D_TEST_CASES, GraphNodePoolBase::PADDING_MODE_USE_PADDINGS);
  Test(AVG_POOL2D_TEST_CASES, GraphNodePoolBase::PADDING_MODE_USE_PADDINGS, 1,
       1);
}

TEST_F(AvgPoolBackwardTest, AvgPool3d) {
  Test(AVG_POOL3D_TEST_CASES, GraphNodePoolBase::PADDING_MODE_SAME);
  Test(AVG_POOL3D_TEST_CASES, GraphNodePoolBase::PADDING_MODE_VALID);
  Test(AVG_POOL3D_TEST_CASES, GraphNodePoolBase::PADDING_MODE_USE_PADDINGS);
  Test(AVG_POOL3D_TEST_CASES, GraphNodePoolBase::PADDING_MODE_USE_PADDINGS, 1,
       1);
}

}  // namespace deepx_core
