// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#include "simp_stage.h"
#include <algorithm>  // std::sort
#include <utility>

namespace deepx_core {

/************************************************************************/
/* SimpStage */
/************************************************************************/
void SimpContext::Init(SimpItem* mutable_item) {
  item = mutable_item;
  nodes_to_simp.Clear();

  std::vector<GraphNode*> sorted;
  item->GetTopologicalSortedNodes(&sorted);
  for (auto* node : sorted) {
    nodes_to_simp.PushBack(node);
  }
}

std::string SimpStage::NewNodeName(const std::string& old_name,
                                   const std::string& suffix) const noexcept {
  return ctx_->item->NewNodeName(old_name, {simp_name(), stage_name()}, "",
                                 suffix);
}

bool SimpStage::IsTarget(const GraphNode* node) const noexcept {
  return ctx_->item->is_target(node->name());
}

bool SimpStage::IsSingleOutput(const GraphNode* node) const noexcept {
  return (int)ctx_->item->find_output(node->name()).size() == 1;
}

void SimpStage::SortByName(std::vector<GraphNode*>* nodes) {
  std::sort(nodes->begin(), nodes->end(),
            [](const GraphNode* a, const GraphNode* b) {
              return a->name() < b->name();
            });
}

/************************************************************************/
/* SimpPipeline */
/************************************************************************/
bool SimpPipeline::TrySimplify(GraphNode* node) {
  for (auto& stage : stages_) {
    if (stage->MaySimplify(node)) {
      if (stage->TrySimplify(node)) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace deepx_core
