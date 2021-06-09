// Copyright 2021 the deepx authors.
// Author: Yalong Wang (vinceywang@tencent.com)
// Author: Chunchen Su (hillsu@tencent.com)
//

#pragma once
#include <deepx_core/common/stream.h>
#include <array>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* FileMetric */
/************************************************************************/
class FileMetric {
 private:
  struct TaskMetric {
    static constexpr size_t BUCKET_SIZE = 1000000;  // magic number
    std::array<double, BUCKET_SIZE> bucket;
    std::array<double, BUCKET_SIZE> positive_bucket;
    double tp = 0;  // true positive
    double tn = 0;  // true negative
    double fp = 0;  // false positive
    double fn = 0;  // false negative
    double auc = 0;

    void add_label_score(double label, double score);

    TaskMetric();

    void clear() noexcept;
    void Merge(const TaskMetric& other);
    void ComputeAUC() noexcept;
  };
  friend OutputStream& operator<<(OutputStream& os,
                                  const TaskMetric& task_metric);
  friend InputStream& operator>>(InputStream& is, TaskMetric& task_metric);

 private:
  double num_inst_ = 0;
  double loss_ = 0;  // sum, not mean
  std::vector<TaskMetric> task_metrics_;

  friend OutputStream& operator<<(OutputStream& os,
                                  const FileMetric& file_metric);
  friend InputStream& operator>>(InputStream& is, FileMetric& file_metric);

 private:
  template <typename Int>
  TaskMetric& safe_task_metric(Int task_id) {
    if (task_metrics_.size() <= (size_t)task_id) {
      task_metrics_.resize((size_t)task_id + 1);
    }
    return task_metrics_[(size_t)task_id];
  }

 public:
  void add_num_inst(double num_inst) noexcept { num_inst_ += num_inst; }
  double num_inst() const noexcept { return num_inst_; }
  void add_loss(double loss) noexcept { loss_ += loss; }
  double loss() const noexcept { return loss_; }
  double mean_loss() const noexcept {
    return num_inst_ == 0 ? 0 : loss_ / num_inst_;
  }
  template <typename Int>
  void add_label_score(Int task_id, double label, double score) {
    safe_task_metric(task_id).add_label_score(label, score);
  }
  int task_size() const noexcept { return (int)task_metrics_.size(); }
  template <typename Int>
  double auc(Int task_id) const noexcept {
    return task_metrics_[(size_t)task_id].auc;
  }
  template <typename Int>
  double num_positive(Int task_id) {
    return task_metrics_[(size_t)task_id].tp +
           task_metrics_[(size_t)task_id].fn;
  }
  template <typename Int>
  double num_negative(Int task_id) {
    return task_metrics_[(size_t)task_id].tn +
           task_metrics_[(size_t)task_id].fp;
  }

 public:
  void clear() noexcept;
  void Merge(const FileMetric& other);
  void ComputeTaskMetric() noexcept;
};

}  // namespace deepx_core
