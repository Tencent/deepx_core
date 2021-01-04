// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#pragma once
#include <deepx_core/graph/graph_node.h>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "simp_item.h"

namespace deepx_core {

#define DEFINE_SIMP_STAGE_LIKE_BASE(clazz_name)                           \
  clazz_name(const std::string& simp_name, const std::string& stage_name, \
             SimpContext* ctx)                                            \
      : SimpStage(simp_name, stage_name, ctx) {}                          \
  clazz_name(const clazz_name&) = delete;                                 \
  clazz_name& operator=(const clazz_name&) = delete

#define DEFINE_SIMP_STAGE_LIKE(clazz_name)                   \
  clazz_name(const std::string& simp_name, SimpContext* ctx) \
      : SimpStage(simp_name, #clazz_name, ctx) {}            \
  clazz_name(const clazz_name&) = delete;                    \
  clazz_name& operator=(const clazz_name&) = delete

/************************************************************************/
/* SimpContext */
/************************************************************************/
struct SimpContext {
  // CR check T is POD, or remove template
  template <typename T>
  class SetVector {
   private:
    std::unordered_set<T> set_;
    std::vector<T> vector_;

   public:
    bool PushBack(const T& value) {
      if (!set_.insert(value).second) {
        return false;
      }
      vector_.emplace_back(value);
      return true;
    }

    T PopBack() {
      T back = vector_.back();
      set_.erase(back);
      vector_.pop_back();
      return back;
    }

    bool Empty() const noexcept { return vector_.empty(); }

    void Clear() noexcept {
      set_.clear();
      vector_.clear();
    }
  };

  SimpItem* item = nullptr;
  SetVector<GraphNode*> nodes_to_simp;

 public:
  void Init(SimpItem* mutable_item);
};

/************************************************************************/
/* SimpStage */
/************************************************************************/
class SimpStage {
 protected:
  std::string simp_name_;
  std::string stage_name_;
  SimpContext* ctx_;

 public:
  const std::string& simp_name() const noexcept { return simp_name_; }
  const std::string& stage_name() const noexcept { return stage_name_; }

 public:
  SimpStage(std::string simp_name, std::string stage_name, SimpContext* ctx)
      : simp_name_(std::move(simp_name)),
        stage_name_(std::move(stage_name)),
        ctx_(ctx) {}
  virtual ~SimpStage() = default;

 public:
  std::string NewNodeName(const std::string& old_name,
                          const std::string& suffix = "") const noexcept;
  virtual bool MaySimplify(const GraphNode* node) const noexcept = 0;
  virtual bool TrySimplify(GraphNode* node) = 0;

 protected:
  bool IsTarget(const GraphNode* node) const noexcept;
  bool IsSingleOutput(const GraphNode* node) const noexcept;
  static void SortByName(std::vector<GraphNode*>* nodes);
};

/************************************************************************/
/* SimpPipeline */
/************************************************************************/
class SimpPipeline {
 protected:
  SimpContext* ctx_;
  std::vector<std::unique_ptr<SimpStage>> stages_;

 public:
  SimpPipeline(SimpContext* ctx, std::vector<std::unique_ptr<SimpStage>> stages)
      : ctx_(ctx), stages_(std::move(stages)) {}

 public:
  bool TrySimplify(GraphNode* node);
};

}  // namespace deepx_core
