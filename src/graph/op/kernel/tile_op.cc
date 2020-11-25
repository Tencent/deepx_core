// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

struct TileAux {
  Shape Z;
  std::vector<int> replicates;
  std::vector<int> steps;
  std::vector<int> loops;
};

template <typename T>
struct TileMutableAux {
  Tensor<T> buf;
};

bool TileCheckAttr(const std::vector<int>& replicates) noexcept {
  for (int replicate : replicates) {
    if (replicate <= 0) {
      DXERROR("Invalid replicates: replicate %d must be positive.", replicate);
      return false;
    }
  }
  return true;
}

bool TilePrepare(const Shape& X, const std::vector<int>& replicates,
                 TileAux* aux) noexcept {
  int Xrank = X.rank();
  if (Xrank == 0) {
    DXERROR("Invalid X: rank of X is zero.");
    return false;
  }

  if ((int)replicates.size() != Xrank) {
    DXERROR("Invalid replicates: size of replicates %d must be %d.",
            (int)replicates.size(), Xrank);
    return false;
  }

  int Zdims[SHAPE_MAX_RANK];
  for (int i = 0; i < Xrank; ++i) {
    Zdims[i] = X[i] * replicates[i];
  }
  aux->Z.assign(&Zdims[0], &Zdims[Xrank]);
  aux->replicates.resize(Xrank);
  aux->steps.resize(Xrank);
  aux->loops.resize(Xrank);
  int step = X.back();
  int block = X.total_dim();
  int replicate = replicates.back();
  aux->replicates[0] = replicate;
  aux->steps[0] = step;
  aux->loops[0] = block / step;
  for (int i = Xrank - 2, j = 1; i >= 0; --i, ++j) {
    block *= replicate;
    step *= X[i] * replicate;
    replicate = replicates[i];
    aux->replicates[j] = replicate;
    aux->steps[j] = step;
    aux->loops[j] = block / step;
  }
  return true;
}

bool TileInferShape(const Shape& X, const std::vector<int>& replicates,
                    Shape* Z) noexcept {
  TileAux aux;
  if (!TilePrepare(X, replicates, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
void TilePrepareBackward(const TileAux& aux, TileMutableAux<T>* maux) {
  maux->buf.resize(aux.Z);
}

template <typename T>
void Tile(const Tensor<T>& X, Tensor<T>* Z, const TileAux& aux) noexcept {
  auto tile_axis =
      [](const T* in, T* out, int replicate, int step, int loop) noexcept {
    if (replicate == 1) {
      LLMath<T>::copy(step * loop, in, out);
    } else {
      const T* _in = in + (loop - 1) * step;
      T* _out = out + (loop * replicate - 1) * step;
      for (int j = 0; j < loop; ++j) {
        for (int k = 0; k < replicate; ++k) {
          LLMath<T>::copy(step, _in, _out);
          _out -= step;
        }
        _in -= step;
      }
    }
  };

  const T* _X = X.data();
  T* _Z = Z->data();
  for (int i = 0; i < X.rank(); ++i) {
    tile_axis(i == 0 ? _X : _Z, _Z, aux.replicates[i], aux.steps[i],
              aux.loops[i]);
  }
}

template <typename T>
void TileBackward(const Tensor<T>& X, const Tensor<T>& /*Z*/,
                  const Tensor<T>& gZ, Tensor<T>* gX, const TileAux& aux,
                  TileMutableAux<T>* maux) noexcept {
  auto tile_axis =
      [](const T* in, T* out, int replicate, int step, int loop) noexcept {
    if (in != out && replicate == 1) {
      LLMath<T>::add(step * loop, in, out, out);
    } else if (in != out) {
      const T* _in = in;
      T* _out = out;
      for (int j = 0; j < loop; ++j) {
        for (int k = 0; k < replicate; ++k) {
          LLMath<T>::add(step, _in, _out, _out);
          _in += step;
        }
        _out += step;
      }
    } else {
      const T* _in = in;
      T* _out = out;
      for (int j = 0; j < loop; ++j) {
        _in += step;  // skip k = 0
        for (int k = 1; k < replicate; ++k) {
          LLMath<T>::add(step, _in, _out, _out);
          _in += step;
        }
        _out += step;
      }
    }
  };

  maux->buf.set_data(gZ);
  T* _gZ = maux->buf.data();
  T* _gX = gX->data();
  for (int i = X.rank() - 1; i >= 0; --i) {
    tile_axis(_gZ, i == 0 ? _gX : _gZ, aux.replicates[i], aux.steps[i],
              aux.loops[i]);
  }
}

}  // namespace

TileNode::TileNode(std::string name, GraphNode* X, int replicate)
    : TileNode(std::move(name), X, std::vector<int>{replicate}) {}

TileNode::TileNode(std::string name, GraphNode* X, std::vector<int> replicates)
    : GraphNodeUnaryBase(std::move(name), X),
      replicates_(std::move(replicates)) {
  DXCHECK_THROW(TileCheckAttr(replicates_));
  if (!X->shape().empty()) {
    (void)TileInferShape(X->shape(), replicates_, &shape_);
  }
}

class TileOp : public OpUnaryBase {
 private:
  TileAux aux_;
  TileMutableAux<float_t> maux_;

 public:
  DEFINE_OP_LIKE(TileOp);

  const Shape& InferShape() override {
    DXCHECK_THROW(TilePrepare(X_->shape(),
                              ((const TileNode*)node_)->replicates(), &aux_));
    return aux_.Z;
  }

  void InitBackward() override {
    OpUnaryBase::InitBackward();
    TilePrepareBackward(aux_, &maux_);
  }

 public:
  void Forward() override { Tile(*X_, Z_, aux_); }

  void Backward() override {
    if (gX_) {
      TileBackward(*X_, *Z_, *gZ_, gX_, aux_, &maux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Tile);

}  // namespace deepx_core
