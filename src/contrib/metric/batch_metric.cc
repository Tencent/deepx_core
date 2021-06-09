// Copyright 2021 the deepx authors.
// Author: Yalong Wang (vinceywang@tencent.com)
// Author: Chunchen Su (hillsu@tencent.com)
//

#include <deepx_core/contrib/metric/batch_metric.h>
#include <algorithm>  // std::sort

namespace deepx_core {

/************************************************************************/
/* BatchMetric::TaskMetric */
/************************************************************************/
OutputStream& operator<<(OutputStream& os,
                         const BatchMetric::TaskMetric& task_metric) {
  os << task_metric.label_scores;
  // no 'auc' and 'copc'
  return os;
}

InputStream& operator>>(InputStream& is, BatchMetric::TaskMetric& task_metric) {
  is >> task_metric.label_scores;
  // no 'auc' and 'copc'
  return is;
}

void BatchMetric::TaskMetric::clear() noexcept {
  label_scores.clear();
  auc = 0;
  copc = 0;
}

void BatchMetric::TaskMetric::Merge(const TaskMetric& other) {
  label_scores.insert(label_scores.end(), other.label_scores.begin(),
                      other.label_scores.end());
  // no 'auc' and 'copc'
}

void BatchMetric::TaskMetric::ComputeAUC() noexcept {
  if (label_scores.empty()) {
    auc = 0;
    return;
  }

  std::sort(
      label_scores.begin(), label_scores.end(),
      [](const std::pair<double, double>& a,
         const std::pair<double, double>& b) { return a.second > b.second; });

  size_t num_positive = 0;
  size_t accumulator = 0;
  for (const auto& label_score : label_scores) {
    if (label_score.first > 0) {
      num_positive += 1;
    } else {
      accumulator += num_positive;
    }
  }
  size_t num_inst = label_scores.size();
  if (num_positive == 0 || num_positive == num_inst) {
    auc = 1;
  } else {
    size_t num_negative = num_inst - num_positive;
    auc = 1.0 * accumulator / num_positive / num_negative;
  }
}

void BatchMetric::TaskMetric::ComputeCOPC() noexcept {
  if (label_scores.empty()) {
    copc = 0;
    return;
  }

  double sum_label = 0;
  double sum_score = 0;
  for (const auto& label_score : label_scores) {
    sum_label += label_score.first;
    sum_score += label_score.second;
  }
  copc = sum_score == 0 ? 0 : sum_label / sum_score;
}

/************************************************************************/
/* BatchMetric */
/************************************************************************/
OutputStream& operator<<(OutputStream& os, const BatchMetric& batch_metric) {
  os << batch_metric.num_inst_ << batch_metric.loss_
     << batch_metric.task_metrics_;
  return os;
}

InputStream& operator>>(InputStream& is, BatchMetric& batch_metric) {
  is >> batch_metric.num_inst_ >> batch_metric.loss_ >>
      batch_metric.task_metrics_;
  return is;
}

void BatchMetric::clear() noexcept {
  num_inst_ = 0;
  loss_ = 0;
  task_metrics_.clear();
}

void BatchMetric::Merge(const BatchMetric& other) {
  num_inst_ += other.num_inst_;
  loss_ += other.loss_;
  for (size_t i = 0; i < other.task_metrics_.size(); ++i) {
    safe_task_metric(i).Merge(other.task_metrics_[i]);
  }
}

void BatchMetric::ComputeTaskMetric() noexcept {
  for (TaskMetric& task_metric : task_metrics_) {
    task_metric.ComputeAUC();
    task_metric.ComputeCOPC();
  }
}

}  // namespace deepx_core
