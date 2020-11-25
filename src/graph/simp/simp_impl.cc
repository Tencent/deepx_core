// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#include "simp_impl.h"
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/graph_simp.h>
#include <memory>
#include <vector>
#include "arithmetic.h"
#include "cf.h"
#include "cse.h"

namespace deepx_core {
namespace {

using simps_t = std::vector<std::unique_ptr<Simp>>;

simps_t GetSimps(const SimpConfig& config) {
  simps_t simps;
  simps.emplace_back(new ArithmeticSimp);

  CFConfig cf_config;
  cf_config.use_static_shape = config.use_static_shape;
  simps.emplace_back(new CFSimp(cf_config));

  simps.emplace_back(new CSESimp);
  return simps;
}

}  // namespace

bool Simplify(const Graph& from, const SimpConfig& config, Graph* to) {
  DXCHECK_THROW(from.compiled());

  SimpItem item;
  DXCHECK_THROW(item.FromGraph(from));

  bool outer = false;
  for (int i = 0; i < config.max_iteration; ++i) {
    bool inner = false;
    for (const auto& simp : GetSimps(config)) {
      if (simp->Simplify(&item)) {
        inner = true;
      }
    }
    if (!inner) {
      break;
    }
    outer = true;
  }

  if (outer) {
    DXCHECK_THROW(item.ToGraph(to));
  }
  return outer;
}

}  // namespace deepx_core
