// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <gtest/gtest.h>

namespace deepx_core {

class TensorMapTest : public testing::Test, public DataType {
 protected:
  TensorMap tensor_map;
  TensorMap read_tensor_map;

 protected:
  void SetUp() override {
    auto& tsr = tensor_map.insert<tsr_t>("0");
    auto& srm = tensor_map.insert<srm_t>("1");
    auto& csr = tensor_map.insert<csr_t>("2");
    auto& tsri = tensor_map.insert<tsri_t>("3");
    auto& tsrs = tensor_map.insert<tsrs_t>("4");

    tsr.resize(100).arange();
    srm = {{0, 1, 2}, {{0, 0, 0}, {1, 1, 1}, {2, 2, 2}}};
    csr = {{0, 1, 4, 6, 7},
           {1, 2, 3, 4, 5, 6, 7},
           {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
    tsri = tsri_t{{0, 1, 2}, {3, 4, 5}};
    tsrs = tsrs_t{{"0", "1", "2"}, {"3", "4", "5"}};
  }
};

TEST_F(TensorMapTest, WriteRead) {
  OutputStringStream os;
  InputStringStream is;

  os << tensor_map;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  is >> read_tensor_map;
  ASSERT_TRUE(is);

  EXPECT_EQ(tensor_map.get<tsr_t>("0"), read_tensor_map.get<tsr_t>("0"));
  EXPECT_EQ(tensor_map.get<srm_t>("1"), read_tensor_map.get<srm_t>("1"));
  EXPECT_EQ(tensor_map.get<csr_t>("2"), read_tensor_map.get<csr_t>("2"));
  EXPECT_EQ(tensor_map.get<tsri_t>("3"), read_tensor_map.get<tsri_t>("3"));
  EXPECT_EQ(tensor_map.get<tsrs_t>("4"), read_tensor_map.get<tsrs_t>("4"));
}

TEST_F(TensorMapTest, WriteReadView) {
  OutputStringStream os;
  InputStringStream is;

  os << tensor_map;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  ReadView(is, read_tensor_map);
  ASSERT_TRUE(is);

  EXPECT_EQ(tensor_map.get<tsr_t>("0"), read_tensor_map.get<tsr_t>("0"));
  EXPECT_EQ(tensor_map.get<srm_t>("1"), read_tensor_map.get<srm_t>("1"));
  EXPECT_EQ(tensor_map.get<csr_t>("2"), read_tensor_map.get<csr_t>("2"));
  EXPECT_EQ(tensor_map.get<tsri_t>("3"), read_tensor_map.get<tsri_t>("3"));
  EXPECT_EQ(tensor_map.get<tsrs_t>("4"), read_tensor_map.get<tsrs_t>("4"));
}

TEST_F(TensorMapTest, ClearSRMValue) {
  tensor_map.ClearSRMValue();

  auto& tsr = tensor_map.get<tsr_t>("0");
  auto& srm = tensor_map.get<srm_t>("1");
  auto& csr = tensor_map.get<csr_t>("2");
  auto& tsri = tensor_map.get<tsri_t>("3");
  auto& tsrs = tensor_map.get<tsrs_t>("4");

  EXPECT_FALSE(tsr.empty());
  EXPECT_TRUE(srm.empty());
  EXPECT_FALSE(srm.shape().empty());
  EXPECT_FALSE(csr.empty());
  EXPECT_FALSE(tsri.empty());
  EXPECT_FALSE(tsrs.empty());
}

TEST_F(TensorMapTest, ClearValue) {
  tensor_map.ClearValue();

  auto& tsr = tensor_map.get<tsr_t>("0");
  auto& srm = tensor_map.get<srm_t>("1");
  auto& csr = tensor_map.get<csr_t>("2");
  auto& tsri = tensor_map.get<tsri_t>("3");
  auto& tsrs = tensor_map.get<tsrs_t>("4");

  EXPECT_TRUE(tsr.empty());
  EXPECT_TRUE(srm.empty());
  EXPECT_FALSE(srm.shape().empty());
  EXPECT_TRUE(csr.empty());
  EXPECT_TRUE(tsri.empty());
  EXPECT_TRUE(tsrs.empty());
}

TEST_F(TensorMapTest, ZerosValue) {
  tensor_map.ZerosValue();

  auto& tsr = tensor_map.get<tsr_t>("0");
  auto& srm = tensor_map.get<srm_t>("1");
  auto& csr = tensor_map.get<csr_t>("2");
  auto& tsri = tensor_map.get<tsri_t>("3");
  auto& tsrs = tensor_map.get<tsrs_t>("4");

  EXPECT_FALSE(tsr.empty());
  EXPECT_TRUE(srm.empty());
  EXPECT_FALSE(srm.shape().empty());
  EXPECT_FALSE(csr.empty());
  EXPECT_FALSE(tsri.empty());
  EXPECT_FALSE(tsrs.empty());
}

TEST_F(TensorMapTest, RemoveEmptyValue_1) {
  tensor_map.RemoveEmptyValue();

  auto& tsr = tensor_map.get<tsr_t>("0");
  auto& srm = tensor_map.get<srm_t>("1");
  auto& csr = tensor_map.get<csr_t>("2");
  auto& tsri = tensor_map.get<tsri_t>("3");
  auto& tsrs = tensor_map.get<tsrs_t>("4");

  EXPECT_FALSE(tsr.empty());
  EXPECT_FALSE(srm.empty());
  EXPECT_FALSE(csr.empty());
  EXPECT_FALSE(tsri.empty());
  EXPECT_FALSE(tsrs.empty());
}

TEST_F(TensorMapTest, RemoveEmptyValue_2) {
  tensor_map.ClearValue();
  tensor_map.RemoveEmptyValue();

  EXPECT_EQ(tensor_map.find("0"), tensor_map.end());
  EXPECT_EQ(tensor_map.find("1"), tensor_map.end());
  EXPECT_EQ(tensor_map.find("2"), tensor_map.end());
  EXPECT_EQ(tensor_map.find("3"), tensor_map.end());
  EXPECT_EQ(tensor_map.find("4"), tensor_map.end());
}

}  // namespace deepx_core
