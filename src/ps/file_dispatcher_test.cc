// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/ps/file_dispatcher.h>
#include <gtest/gtest.h>
#include <algorithm>  // std::sort
#include <string>
#include <vector>

namespace deepx_core {

class FileDispatcherTest : public testing::Test {};

TEST_F(FileDispatcherTest, reverse0_shuffle0_timeout0) {
  FileDispatcher file_dispatcher;
  file_dispatcher.set_reverse(0);
  file_dispatcher.set_shuffle(0);
  file_dispatcher.set_timeout(0);

  std::vector<std::string> files = {"1", "2", "3"};
  file_dispatcher.PreTrain(files);
  file_dispatcher.PreEpoch();

  std::string file;
  ASSERT_TRUE(file_dispatcher.WorkerDispatchFile(&file));
  EXPECT_EQ(file, "1");
  ASSERT_FALSE(file_dispatcher.WorkerFinishFile(file));

  ASSERT_TRUE(file_dispatcher.WorkerDispatchFile(&file));
  EXPECT_EQ(file, "2");
  ASSERT_FALSE(file_dispatcher.WorkerFinishFile(file));

  ASSERT_TRUE(file_dispatcher.WorkerDispatchFile(&file));
  EXPECT_EQ(file, "3");
  file_dispatcher.WorkerFailureFile(file);
  file_dispatcher.WorkerFailureFile(file);

  ASSERT_TRUE(file_dispatcher.WorkerDispatchFile(&file));
  EXPECT_EQ(file, "3");
  ASSERT_TRUE(file_dispatcher.WorkerFinishFile(file));

  ASSERT_FALSE(file_dispatcher.WorkerDispatchFile(&file));
}

TEST_F(FileDispatcherTest, reverse1_shuffle0_timeout0) {
  FileDispatcher file_dispatcher;
  file_dispatcher.set_reverse(1);
  file_dispatcher.set_shuffle(0);
  file_dispatcher.set_timeout(0);

  std::vector<std::string> files = {"1", "2", "3"};
  file_dispatcher.PreTrain(files);
  file_dispatcher.PreEpoch();

  std::string file;
  ASSERT_TRUE(file_dispatcher.WorkerDispatchFile(&file));
  EXPECT_EQ(file, "3");
  ASSERT_FALSE(file_dispatcher.WorkerFinishFile(file));

  ASSERT_TRUE(file_dispatcher.WorkerDispatchFile(&file));
  EXPECT_EQ(file, "2");
  ASSERT_FALSE(file_dispatcher.WorkerFinishFile(file));

  ASSERT_TRUE(file_dispatcher.WorkerDispatchFile(&file));
  EXPECT_EQ(file, "1");
  file_dispatcher.WorkerFailureFile(file);
  file_dispatcher.WorkerFailureFile(file);

  ASSERT_TRUE(file_dispatcher.WorkerDispatchFile(&file));
  EXPECT_EQ(file, "1");
  ASSERT_TRUE(file_dispatcher.WorkerFinishFile(file));

  ASSERT_FALSE(file_dispatcher.WorkerDispatchFile(&file));
}

TEST_F(FileDispatcherTest, reverse0_shuffle1_timeout0) {
  FileDispatcher file_dispatcher;
  file_dispatcher.set_reverse(0);
  file_dispatcher.set_shuffle(1);
  file_dispatcher.set_timeout(0);

  std::vector<std::string> files;
  for (int i = 0; i < 100; ++i) {
    files.emplace_back(std::to_string(i));
  }
  file_dispatcher.PreTrain(files);
  file_dispatcher.PreEpoch();

  std::vector<std::string> dispatched;
  for (;;) {
    std::string file;
    ASSERT_TRUE(file_dispatcher.WorkerDispatchFile(&file));
    dispatched.emplace_back(file);
    if (file_dispatcher.WorkerFinishFile(file)) {
      break;
    }
  }

  EXPECT_NE(files, dispatched);

  std::sort(files.begin(), files.end());
  std::sort(dispatched.begin(), dispatched.end());
  EXPECT_EQ(files, dispatched);
}

}  // namespace deepx_core
