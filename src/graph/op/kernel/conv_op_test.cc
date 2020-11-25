// Copyright 2019 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class ConvForwardTest : public testing::Test, public DataType {
 protected:
  struct TestCase {
    int conv_rank = 0;
    Shape Xshape, Kshape, Zshape;
    int data_format = 0;
    std::vector<int> strides;
    std::vector<int> dilations;
    int padding_mode = 0;
    std::vector<int> paddings;
    std::vector<float_t> Xvalues, Kvalues, Zvalues;
    tsr_t X, K, Z;

   public:
    void ParseFrom(int _conv_rank, const std::string& param,
                   const std::string& Xline, const std::string& Kline,
                   const std::string& Zline) {
      conv_rank = _conv_rank;
      std::vector<int> Xdims(conv_rank + 2);
      std::vector<int> Kdims(conv_rank + 2);
      std::vector<int> Zdims(conv_rank + 2);
      std::istringstream iss;
      iss.str(param);

      for (int& Xdim : Xdims) {
        iss >> Xdim;
      }
      ASSERT_TRUE(iss);
      Xshape.assign(Xdims.begin(), Xdims.end());

      for (int& Kdim : Kdims) {
        iss >> Kdim;
      }
      ASSERT_TRUE(iss);
      Kshape.assign(Kdims.begin(), Kdims.end());

      for (int& Zdim : Zdims) {
        iss >> Zdim;
      }
      ASSERT_TRUE(iss);
      Zshape.assign(Zdims.begin(), Zdims.end());

      iss >> data_format;
      ASSERT_TRUE(iss);

      strides.resize(conv_rank);
      for (int& stride : strides) {
        iss >> stride;
      }
      ASSERT_TRUE(iss);

      dilations.resize(conv_rank);
      for (int& dilation : dilations) {
        iss >> dilation;
      }
      ASSERT_TRUE(iss);

      iss >> padding_mode;
      ASSERT_TRUE(iss);

      paddings.resize(conv_rank);
      for (int& padding : paddings) {
        iss >> padding;
      }
      ASSERT_TRUE(iss);

      ASSERT_TRUE(Split(Xline, " ", &Xvalues));
      ASSERT_EQ(Xshape.total_dim(), (int)Xvalues.size());
      X.view(Xshape, Xvalues.data());

      ASSERT_TRUE(Split(Kline, " ", &Kvalues));
      ASSERT_EQ(Kshape.total_dim(), (int)Kvalues.size());
      K.view(Kshape, Kvalues.data());

      ASSERT_TRUE(Split(Zline, " ", &Zvalues));
      ASSERT_EQ(Zshape.total_dim(), (int)Zvalues.size());
      Z.view(Zshape, Zvalues.data());
    }
  };

 protected:
  void LoadTestCase(int conv_rank, const std::string& file,
                    std::vector<TestCase>* test_cases) {
    test_cases->clear();

    std::ifstream is(file);
    ASSERT_TRUE(is.is_open());

    std::string param, Xline, Kline, Zline;
    while (std::getline(is, param)) {
      ASSERT_TRUE(std::getline(is, Xline));
      ASSERT_TRUE(std::getline(is, Kline));
      ASSERT_TRUE(std::getline(is, Zline));
      TestCase test_case;
      test_case.ParseFrom(conv_rank, param, Xline, Kline, Zline);
      test_cases->emplace_back(std::move(test_case));
    }
    is.close();
  }

  void Test(const std::vector<TestCase>& test_cases) {
    for (const TestCase& test_case : test_cases) {
      auto* X = new VariableNode("X", test_case.Xshape);
      auto* K = new VariableNode("K", test_case.Kshape);
      GraphNode* Z;
      if (test_case.conv_rank == 1) {
        Z = new Conv1dNode("Z", X, K, test_case.data_format,
                           test_case.strides[0], test_case.dilations[0],
                           test_case.padding_mode, test_case.paddings[0]);
      } else if (test_case.conv_rank == 2) {
        Z = new Conv2dNode("Z", X, K, test_case.data_format, test_case.strides,
                           test_case.dilations, test_case.padding_mode,
                           test_case.paddings);
      } else {
        Z = new Conv3dNode("Z", X, K, test_case.data_format, test_case.strides,
                           test_case.dilations, test_case.padding_mode,
                           test_case.paddings);
      }
      auto post_param_initializer = [&test_case](
                                        std::default_random_engine /*engine*/,
                                        TensorMap* param) {
        param->get<tsr_t>("X") = test_case.X;
        param->get<tsr_t>("K") = test_case.K;
      };
      CheckOpForward(Z, 1, test_case.Z, nullptr, post_param_initializer,
                     nullptr);
    }
  }
};

TEST_F(ConvForwardTest, Conv1d) {
  std::vector<TestCase> test_cases;
  LoadTestCase(1, "testdata/graph/op/kernel/conv1d.txt", &test_cases);
  Test(test_cases);
}

TEST_F(ConvForwardTest, Conv2d) {
  std::vector<TestCase> test_cases;
  LoadTestCase(2, "testdata/graph/op/kernel/conv2d.txt", &test_cases);
  Test(test_cases);
}

TEST_F(ConvForwardTest, Conv3d) {
  std::vector<TestCase> test_cases;
  LoadTestCase(3, "testdata/graph/op/kernel/conv3d.txt", &test_cases);
  Test(test_cases);
}

class ConvBackwardTest : public testing::Test {
 protected:
  struct TestCase {
    int conv_rank;
    Shape Xshape;
    Shape Kshape;
    int data_format;
    std::vector<int> strides;
    std::vector<int> dilations;
    std::vector<int> paddings;
  };
  const std::vector<TestCase> CONV1D_TEST_CASES = {
      {1,
       Shape(1, 1, 11),
       Shape(1, 1, 1),
       GraphNodeConvBase::DATA_FORMAT_NCW,
       {1},
       {1},
       {0}},
      {1,
       Shape(2, 1, 11),
       Shape(1, 1, 3),
       GraphNodeConvBase::DATA_FORMAT_NCW,
       {1},
       {1},
       {0}},
      {1,
       Shape(4, 2, 11),
       Shape(3, 2, 3),
       GraphNodeConvBase::DATA_FORMAT_NCW,
       {2},
       {1},
       {1}},
      {1,
       Shape(1, 11, 1),
       Shape(1, 1, 1),
       GraphNodeConvBase::DATA_FORMAT_NWC,
       {1},
       {1},
       {0}},
      {1,
       Shape(2, 11, 1),
       Shape(3, 1, 1),
       GraphNodeConvBase::DATA_FORMAT_NWC,
       {1},
       {1},
       {0}},
      {1,
       Shape(4, 11, 2),
       Shape(3, 2, 3),
       GraphNodeConvBase::DATA_FORMAT_NWC,
       {2},
       {1},
       {1}}};
  const std::vector<TestCase> CONV2D_TEST_CASES = {
      {2,
       Shape(1, 1, 5, 5),
       Shape(1, 1, 1, 1),
       GraphNodeConvBase::DATA_FORMAT_NCHW,
       {1, 1},
       {1, 1},
       {0, 0}},
      {2,
       Shape(2, 1, 5, 5),
       Shape(1, 1, 3, 3),
       GraphNodeConvBase::DATA_FORMAT_NCHW,
       {1, 1},
       {1, 1},
       {0, 0}},
      {2,
       Shape(3, 2, 7, 8),
       Shape(4, 2, 3, 2),
       GraphNodeConvBase::DATA_FORMAT_NCHW,
       {1, 2},
       {3, 2},
       {3, 2}},
      {2,
       Shape(5, 5, 7, 8),
       Shape(6, 5, 2, 3),
       GraphNodeConvBase::DATA_FORMAT_NCHW,
       {2, 3},
       {2, 3},
       {4, 3}},
      {2,
       Shape(1, 5, 5, 1),
       Shape(1, 1, 1, 1),
       GraphNodeConvBase::DATA_FORMAT_NHWC,
       {1, 1},
       {1, 1},
       {0, 0}},
      {2,
       Shape(2, 5, 5, 1),
       Shape(3, 3, 1, 1),
       GraphNodeConvBase::DATA_FORMAT_NHWC,
       {1, 1},
       {1, 1},
       {0, 0}},
      {2,
       Shape(3, 7, 8, 2),
       Shape(3, 2, 2, 4),
       GraphNodeConvBase::DATA_FORMAT_NHWC,
       {1, 2},
       {3, 2},
       {3, 2}},
      {2,
       Shape(5, 7, 8, 5),
       Shape(2, 3, 5, 6),
       GraphNodeConvBase::DATA_FORMAT_NHWC,
       {2, 3},
       {2, 3},
       {4, 3}}};
  const std::vector<TestCase> CONV3D_TEST_CASES = {
      {3,
       Shape(1, 1, 3, 3, 3),
       Shape(1, 1, 1, 1, 1),
       GraphNodeConvBase::DATA_FORMAT_NCDHW,
       {1, 1, 1},
       {1, 1, 1},
       {0, 0, 0}},
      {3,
       Shape(2, 1, 5, 7, 9),
       Shape(3, 1, 3, 3, 3),
       GraphNodeConvBase::DATA_FORMAT_NCDHW,
       {1, 1, 1},
       {1, 1, 1},
       {0, 0, 0}},
      {3,
       Shape(2, 3, 6, 7, 5),
       Shape(5, 3, 3, 3, 2),
       GraphNodeConvBase::DATA_FORMAT_NCDHW,
       {2, 1, 3},
       {2, 3, 2},
       {1, 2, 3}},
      {3,
       Shape(1, 3, 3, 3, 1),
       Shape(1, 1, 1, 1, 1),
       GraphNodeConvBase::DATA_FORMAT_NDHWC,
       {1, 1, 1},
       {1, 1, 1},
       {0, 0, 0}},
      {3,
       Shape(2, 5, 7, 9, 1),
       Shape(3, 3, 3, 1, 3),
       GraphNodeConvBase::DATA_FORMAT_NDHWC,
       {1, 1, 1},
       {1, 1, 1},
       {0, 0, 0}},
      {3,
       Shape(2, 6, 7, 5, 3),
       Shape(3, 3, 2, 3, 5),
       GraphNodeConvBase::DATA_FORMAT_NDHWC,
       {2, 1, 3},
       {2, 3, 2},
       {1, 2, 3}}};

 protected:
  void Test(const std::vector<TestCase>& test_cases, int padding_mode) {
    for (const TestCase& test_case : test_cases) {
      auto* X = new VariableNode("X", test_case.Xshape,
                                 TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
      auto* K = new VariableNode("K", test_case.Kshape,
                                 TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
      GraphNode* Z;
      if (test_case.conv_rank == 1) {
        Z = new Conv1dNode("Z", X, K, test_case.data_format,
                           test_case.strides[0], test_case.dilations[0],
                           padding_mode, test_case.paddings[0]);
      } else if (test_case.conv_rank == 2) {
        Z = new Conv2dNode("Z", X, K, test_case.data_format, test_case.strides,
                           test_case.dilations, padding_mode,
                           test_case.paddings);
      } else {
        Z = new Conv3dNode("Z", X, K, test_case.data_format, test_case.strides,
                           test_case.dilations, padding_mode,
                           test_case.paddings);
      }
      CheckOpBackward(Z, 1);
    }
  }
};

TEST_F(ConvBackwardTest, Conv1d) {
  Test(CONV1D_TEST_CASES, GraphNodeConvBase::PADDING_MODE_SAME);
  Test(CONV1D_TEST_CASES, GraphNodeConvBase::PADDING_MODE_VALID);
  Test(CONV1D_TEST_CASES, GraphNodeConvBase::PADDING_MODE_USE_PADDINGS);
}

TEST_F(ConvBackwardTest, Conv2d) {
  Test(CONV2D_TEST_CASES, GraphNodeConvBase::PADDING_MODE_SAME);
  Test(CONV2D_TEST_CASES, GraphNodeConvBase::PADDING_MODE_VALID);
  Test(CONV2D_TEST_CASES, GraphNodeConvBase::PADDING_MODE_USE_PADDINGS);
}

TEST_F(ConvBackwardTest, Conv3d) {
  Test(CONV3D_TEST_CASES, GraphNodeConvBase::PADDING_MODE_SAME);
  Test(CONV3D_TEST_CASES, GraphNodeConvBase::PADDING_MODE_VALID);
  Test(CONV3D_TEST_CASES, GraphNodeConvBase::PADDING_MODE_USE_PADDINGS);
}

}  // namespace deepx_core
