// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#pragma once
#include <string>
#include "cf.h"
#include "simp_stage.h"

namespace deepx_core {

class GraphFolding {
 private:
  CFConfig config_;

 public:
  explicit GraphFolding(const CFConfig& config);
  ~GraphFolding() = default;
  bool FoldGraph(SimpItem* item);

 private:
  bool IsFoldable(const GraphNode* node, const SimpItem* item) const noexcept;
  bool FoldNode(GraphNode* node, SimpItem* item);
};

#define DEFINE_IDENTICAL_NODE_LIKE(clazz_name)               \
  clazz_name(const std::string& simp_name, SimpContext* ctx) \
      : RemoveIdenticalNodeBase(simp_name, #clazz_name, ctx) {}

class RemoveIdenticalNodeBase : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE_BASE(RemoveIdenticalNodeBase);

 public:
  bool TrySimplify(GraphNode* node) override;
};

class RemoveIdenticalTransposeStage : public RemoveIdenticalNodeBase {
 public:
  DEFINE_IDENTICAL_NODE_LIKE(RemoveIdenticalTransposeStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
};

class RemoveIdenticalTileStage : public RemoveIdenticalNodeBase {
 public:
  DEFINE_IDENTICAL_NODE_LIKE(RemoveIdenticalTileStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
};

class RemoveIdenticalSubscriptRangeStage : public RemoveIdenticalNodeBase {
 public:
  DEFINE_IDENTICAL_NODE_LIKE(RemoveIdenticalSubscriptRangeStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
};

class RemoveIdenticalReshapeStage : public RemoveIdenticalNodeBase {
 public:
  DEFINE_IDENTICAL_NODE_LIKE(RemoveIdenticalReshapeStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
};

class DivToReciprocalMulFoldingStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(DivToReciprocalMulFoldingStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;
};

class PartialAddNFoldingStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(PartialAddNFoldingStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;

 private:
  bool IsFoldableAddNInput(const GraphNode* node) const noexcept;
};

class PartialConcatFoldingStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(PartialConcatFoldingStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;

 private:
  bool IsFoldableConcatInput(const GraphNode* node) const noexcept;
};

class ArithmeticOperationsFoldingStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(ArithmeticOperationsFoldingStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;
};

class ConstantPushDownStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(ConstantPushDownStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;
};

#undef DEFINE_IDENTICAL_NODE_LIKE

}  // namespace deepx_core
