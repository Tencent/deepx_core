// Copyright 2021 the deepx authors.
// Author: Yalong Wang (vinceywang@tencent.com)
// Author: Chunchen Su (hillsu@tencent.com)
//

#include <deepx_core/contrib/metric/file_metric.h>

namespace deepx_core {

/************************************************************************/
/* FileMetric::TaskMetric */
/************************************************************************/
OutputStream& operator<<(OutputStream& os,
                         const FileMetric::TaskMetric& task_metric) {
  for (size_t i = 0; i < FileMetric::TaskMetric::BUCKET_SIZE; ++i) {
    os << task_metric.bucket[i];
  }
  for (size_t i = 0; i < FileMetric::TaskMetric::BUCKET_SIZE; ++i) {
    os << task_metric.positive_bucket[i];
  }
  os << task_metric.tp << task_metric.tn << task_metric.fp << task_metric.fn;
  // no 'auc'
  return os;
}

InputStream& operator>>(InputStream& is, FileMetric::TaskMetric& task_metric) {
  for (size_t i = 0; i < FileMetric::TaskMetric::BUCKET_SIZE; ++i) {
    is >> task_metric.bucket[i];
  }
  for (size_t i = 0; i < FileMetric::TaskMetric::BUCKET_SIZE; ++i) {
    is >> task_metric.positive_bucket[i];
  }
  is >> task_metric.tp >> task_metric.tn >> task_metric.fp >> task_metric.fn;
  // no 'auc'
  return is;
}

void FileMetric::TaskMetric::add_label_score(double label, double score) {
  int _label = label > 0 ? 1 : 0;
  // TODO(hillsu): '0.5' is unreasonable.
  int _score = score >= 0.5 ? 1 : 0;
  size_t index;
  if (score <= 0) {
    index = 0;
  } else if (score < 1) {
    index = (size_t)(BUCKET_SIZE * score);
  } else {
    index = BUCKET_SIZE - 1;
  }
  bucket[index] += 1;
  positive_bucket[index] += (double)_label;
  if (_label) {
    if (_score) {
      ++tp;
    } else {
      ++fn;
    }
  } else {
    if (_score) {
      ++fp;
    } else {
      ++tn;
    }
  }
}

FileMetric::TaskMetric::TaskMetric() { clear(); }

void FileMetric::TaskMetric::clear() noexcept {
  bucket.fill(0);
  positive_bucket.fill(0);
  tp = 0;
  tn = 0;
  fp = 0;
  fn = 0;
  auc = 0;
}

void FileMetric::TaskMetric::Merge(const TaskMetric& other) {
  for (size_t i = 0; i < BUCKET_SIZE; ++i) {
    bucket[i] += other.bucket[i];
    positive_bucket[i] += other.positive_bucket[i];
  }
  tp += other.tp;
  tn += other.tn;
  fp += other.fp;
  fn += other.fn;
  // no 'auc'
}

void FileMetric::TaskMetric::ComputeAUC() noexcept {
  double num_positive = tp + fn;
  double num_negative = tn + fp;
  double accumulated_tp = 0;
  double accumulated_fp = 0;
  double tpr = 0;  // true positive rate
  double fpr = 0;  // false positive rate
  double prev_tpr = 0;
  double prev_fpr = 0;
  auc = 0;
  if (num_positive > 0 && num_negative > 0) {
    for (size_t i = 0; i < BUCKET_SIZE; ++i) {
      size_t j = BUCKET_SIZE - i - 1;
      accumulated_tp += positive_bucket[j];
      accumulated_fp += bucket[j] - positive_bucket[j];
      tpr = accumulated_tp / num_positive;
      fpr = accumulated_fp / num_negative;
      auc += 0.5 * (tpr + prev_tpr) * (fpr - prev_fpr);
      prev_tpr = tpr;
      prev_fpr = fpr;
    }
  }
}

/************************************************************************/
/* FileMetric */
/************************************************************************/
OutputStream& operator<<(OutputStream& os, const FileMetric& file_metric) {
  os << file_metric.num_inst_ << file_metric.loss_ << file_metric.task_metrics_;
  return os;
}

InputStream& operator>>(InputStream& is, FileMetric& file_metric) {
  is >> file_metric.num_inst_ >> file_metric.loss_ >> file_metric.task_metrics_;
  return is;
}

void FileMetric::clear() noexcept {
  num_inst_ = 0;
  loss_ = 0;
  task_metrics_.clear();
}

void FileMetric::Merge(const FileMetric& other) {
  num_inst_ += other.num_inst_;
  loss_ += other.loss_;
  for (size_t i = 0; i < other.task_metrics_.size(); ++i) {
    safe_task_metric(i).Merge(other.task_metrics_[i]);
  }
}

void FileMetric::ComputeTaskMetric() noexcept {
  for (TaskMetric& task_metric : task_metrics_) {
    task_metric.ComputeAUC();
  }
}

}  // namespace deepx_core
