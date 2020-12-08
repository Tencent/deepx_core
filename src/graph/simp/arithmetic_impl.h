// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "arithmetic.h"
#include "simp_stage.h"

namespace deepx_core {

#define DEFINE_REWRITE_GROUPED_NODES_LIKE(clazz_name)   \
  clazz_name(const std::string& name, SimpContext* ctx) \
      : RewriteGroupedNodesBase(name, #clazz_name, ctx) {}

struct NodeGroup {
  GraphNode* root = nullptr;
  std::vector<GraphNode*> non_root_members;
  std::vector<GraphNode*> inputs;
};

class RewriteGroupedNodesBase : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE_BASE(RewriteGroupedNodesBase);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;

 protected:
  virtual bool CanBeRoot(const GraphNode* node) const noexcept = 0;
  virtual bool Absorbable(const GraphNode* node, const NodeGroup* group) const
      noexcept = 0;
  virtual bool CanBeMember(const GraphNode* node) const noexcept = 0;
  virtual bool SimplifyGroup(NodeGroup* group) = 0;
  virtual void TryAbsorbNodeToGroup(GraphNode* node, NodeGroup* group);
};

class RewriteGroupedAddStage : public RewriteGroupedNodesBase {
 public:
  DEFINE_REWRITE_GROUPED_NODES_LIKE(RewriteGroupedAddStage);

 protected:
  bool CanBeRoot(const GraphNode* node) const noexcept override;
  bool Absorbable(const GraphNode* node, const NodeGroup* /*group*/) const
      noexcept override;
  bool CanBeMember(const GraphNode* node) const noexcept override;
  bool SimplifyGroup(NodeGroup* group) override;
};

class RewriteGroupedBroadcastStage : public RewriteGroupedNodesBase {
 public:
  DEFINE_REWRITE_GROUPED_NODES_LIKE(RewriteGroupedBroadcastStage);

 protected:
  bool CanBeRoot(const GraphNode* node) const noexcept override;
  bool Absorbable(const GraphNode* node, const NodeGroup* /*group*/) const
      noexcept override;
  bool CanBeMember(const GraphNode* node) const noexcept override;
  bool SimplifyGroup(NodeGroup* group) override;

 private:
  static bool CompareByShapeAndName(const GraphNode* a,
                                    const GraphNode* b) noexcept;
  static bool CompareStrideMoveToTarget(const GraphNode* target,
                                        const GraphNode* a,
                                        const GraphNode* b) noexcept;
  static int CountStrideMoveInBroadcast(const GraphNode* a,
                                        const GraphNode* b) noexcept;
};

class RewritePowStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(RewritePowStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;
};

class RewriteMaxOrMinOfMonotonicStage : public SimpStage {
 private:
  static const std::vector<std::type_index> MONOTONIC_NON_INCREASING_TYPEIDS_;
  static const std::vector<std::type_index> MONOTONIC_NON_DECREASING_TYPEIDS_;

 public:
  DEFINE_SIMP_STAGE_LIKE(RewriteMaxOrMinOfMonotonicStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;

 private:
  bool IsMaxPool(const GraphNode* node) const noexcept;
  bool IsMonotonicNonIncreasing(const GraphNode* node) const noexcept;
  bool IsMonotonicNonDecreasing(const GraphNode* node) const noexcept;
};

class RewriteAggregatableAddNStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(RewriteAggregatableAddNStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;

 private:
  void GetAggregatableInputs(GraphNode* node,
                             std::unordered_map<GraphNode*, int>* inputs);
};

class RewriteSquareMulStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(RewriteSquareMulStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;
};

class RewriteCubicMulStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(RewriteCubicMulStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;
};

class RewriteNegateStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(RewriteNegateStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;

 private:
  bool IsRewritableNegate(const GraphNode* node) const noexcept;
};

class RewriteInvStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(RewriteInvStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;

 private:
  bool IsRewritableInv(const GraphNode* node) const noexcept;
};

class RewriteSuccessiveReshapeStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(RewriteSuccessiveReshapeStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;

 private:
  bool IsRewritableReshape(const GraphNode* node) const noexcept;
};

class FuseTransposeIntoMatmulOrGEMMStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(FuseTransposeIntoMatmulOrGEMMStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;
};

#define DEFINE_REMOVE_INVOLUTION_LIKE(clazz_name)            \
  clazz_name(const std::string& simp_name, SimpContext* ctx) \
      : RemoveInvolutionBase(simp_name, #clazz_name, ctx) {}

class RemoveInvolutionBase : public SimpStage {
 private:
  static const std::vector<std::type_index> VALUE_PRESERVING_TYPEIDS_;

 public:
  DEFINE_SIMP_STAGE_LIKE_BASE(RemoveInvolutionBase);

 public:
  bool TrySimplify(GraphNode* node) override;

 private:
  bool IsRemovableValuePreserving(const GraphNode* node) const noexcept;
};

class RemoveInvInvolutionStage : public RemoveInvolutionBase {
 public:
  DEFINE_REMOVE_INVOLUTION_LIKE(RemoveInvInvolutionStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
};

class RemoveNegateInvolutionStage : public RemoveInvolutionBase {
 public:
  DEFINE_REMOVE_INVOLUTION_LIKE(RemoveNegateInvolutionStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
};

class RemoveIneffectiveAdjacentTransposeStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(RemoveIneffectiveAdjacentTransposeStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;
};

class RemoveIdempotentStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(RemoveIdempotentStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;
};

class HoistCommonFactorOutOfAggregationStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(HoistCommonFactorOutOfAggregationStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;
};

class HoistCommonDenominatorOutOfAggregationStage : public SimpStage {
 public:
  DEFINE_SIMP_STAGE_LIKE(HoistCommonDenominatorOutOfAggregationStage);

 public:
  bool MaySimplify(const GraphNode* node) const noexcept override;
  bool TrySimplify(GraphNode* node) override;
};

#undef DEFINE_REWRITE_GROUPED_NODES_LIKE
#undef DEFINE_REMOVE_INVOLUTION_LIKE

}  // namespace deepx_core
