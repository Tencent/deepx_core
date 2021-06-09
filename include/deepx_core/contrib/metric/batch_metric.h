// Copyright 2021 the deepx authors.
// Author: Yalong Wang (vinceywang@tencent.com)
// Author: Chunchen Su (hillsu@tencent.com)
//

#pragma once
#include <deepx_core/common/stream.h>
#include <utility>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* BatchMetric */
/************************************************************************/
class BatchMetric {
 private:
  struct TaskMetric {
    std::vector<std::pair<double, double>> label_scores;
    double auc = 0;
    double copc = 0;

    void clear() noexcept;
    void Merge(const TaskMetric& other);
    void ComputeAUC() noexcept;
    void ComputeCOPC() noexcept;
  };
  friend OutputStream& operator<<(OutputStream& os,
                                  const TaskMetric& task_metric);
  friend InputStream& operator>>(InputStream& is, TaskMetric& task_metric);

 private:
  double num_inst_ = 0;
  double loss_ = 0;  // sum, not mean
  std::vector<TaskMetric> task_metrics_;

  friend OutputStream& operator<<(OutputStream& os,
                                  const BatchMetric& batch_metric);
  friend InputStream& operator>>(InputStream& is, BatchMetric& batch_metric);

 private:
  template <typename Int>
  TaskMetric& safe_task_metric(Int task_id) {
    if (task_metrics_.size() <= (size_t)task_id) {
      task_metrics_.resize((size_t)task_id + 1);
    }
    return task_metrics_[(size_t)task_id];
  }

 public:
  void set_num_inst(double num_inst) noexcept { num_inst_ = num_inst; }
  double num_inst() const noexcept { return num_inst_; }
  void set_loss(double loss) noexcept { loss_ = loss; }
  double loss() const noexcept { return loss_; }
  double mean_loss() const noexcept {
    return num_inst_ == 0 ? 0 : loss_ / num_inst_;
  }
  template <typename Int>
  void add_label_score(Int task_id, double label, double score) {
    safe_task_metric(task_id).label_scores.emplace_back(label, score);
  }
  int task_size() const noexcept { return (int)task_metrics_.size(); }
  template <typename Int>
  double auc(Int task_id) const noexcept {
    return task_metrics_[(size_t)task_id].auc;
  }
  template <typename Int>
  double copc(Int task_id) const noexcept {
    return task_metrics_[(size_t)task_id].copc;
  }

 public:
  void clear() noexcept;
  void Merge(const BatchMetric& other);
  void ComputeTaskMetric() noexcept;
};

}  // namespace deepx_core
