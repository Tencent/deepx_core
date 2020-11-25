// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/any_map.h>
#include <deepx_core/graph/instance_reader.h>
#include <gtest/gtest.h>
#include <memory>
#include <string>

namespace deepx_core {

class LibsvmInstanceReaderTest : public testing::Test {
 protected:
  void TestGetBatch(const std::string& file, int batch, int has_w, int has_uuid,
                    int expected_m, int expected_n) {
    std::unique_ptr<InstanceReader> reader(NewInstanceReader("libsvm"));
    StringMap config;
    config["batch"] = std::to_string(batch);
    config["w"] = std::to_string(has_w);
    config["uuid"] = std::to_string(has_uuid);
    ASSERT_TRUE(reader->InitConfig(config));

    int m = 0;  // # of batch
    int n = 0;  // # of inst
    Instance inst;
    ASSERT_TRUE(reader->Open(file));
    while (reader->GetBatch(&inst)) {
      if (has_w) {
        EXPECT_GT(inst.count(W_NAME), 0u);
      } else {
        EXPECT_EQ(inst.count(W_NAME), 0u);
      }
      if (has_uuid) {
        EXPECT_GT(inst.count(UUID_NAME), 0u);
      } else {
        EXPECT_EQ(inst.count(UUID_NAME), 0u);
      }
      ++m;
      n += inst.batch();
    }
    n += inst.batch();
    EXPECT_EQ(m, expected_m);
    EXPECT_EQ(n, expected_n);
  }
};

TEST_F(LibsvmInstanceReaderTest, GetBatch) {
  TestGetBatch("testdata/graph/instance_reader/libsvm.txt", 1, 0, 0, 60, 60);
  TestGetBatch("testdata/graph/instance_reader/libsvm.txt", 10, 0, 0, 6, 60);
  TestGetBatch("testdata/graph/instance_reader/libsvm.txt", 15, 0, 0, 4, 60);
  TestGetBatch("testdata/graph/instance_reader/libsvm.txt", 20, 0, 0, 3, 60);
  TestGetBatch("testdata/graph/instance_reader/libsvm.txt", 32, 0, 1, 1, 60);
  TestGetBatch("testdata/graph/instance_reader/libsvm.txt", 60, 1, 0, 1, 60);
  TestGetBatch("testdata/graph/instance_reader/libsvm.txt", 64, 1, 1, 0, 60);
}

}  // namespace deepx_core
