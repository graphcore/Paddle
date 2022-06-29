// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/platform/device/ipu/ipu_backend.h"
#include "paddle/fluid/platform/device/ipu/popart_canonicalization/canonicalization_utils.h"
#include "paddle/fluid/platform/device/ipu/popart_canonicalization/op_builder.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace ipu {
namespace {

bool is_dynamic_graph() {
  auto *ipu_backend = platform::ipu::IpuBackend::GetInstance();
  return ipu_backend->GetIpuStrategy()->is_dynamic;
}

Node *identity_loss_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto reduction = BOOST_GET_CONST(int, op->GetAttr("reduction"));
  return CreateIdentityLossOp(
      graph, node, node->inputs, node->outputs, reduction);
}

Node *cross_entropy2_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto ignoreIndex = BOOST_GET_CONST(int, op->GetAttr("ignore_index"));
  Node *cast_and_reshape = nullptr;
  Node *final_loss_node = nullptr;
  int reduction = RemoveTailReduction(graph, node, "Y");
  bool append_identity_loss = is_dynamic_graph();
  bool is_last_var_node = IsLastVarNode(GetOutputVarNode("Y", node));
  append_identity_loss = append_identity_loss && is_last_var_node;

  if (GetInputVarNode("Label", node)->Var()->GetDataType() ==
      framework::proto::VarType::INT32) {
    cast_and_reshape = GetInputVarNode("Label", node);
  } else {
    auto cast_op = CreateCast(graph,
                              node,
                              {GetInputVarNode("Label", node)},
                              {},
                              framework::proto::VarType::INT32);
    cast_and_reshape = cast_op->outputs[0];
  }

  auto label_shape_ = GetInputVarNode("Label", node)->Var()->GetShape();

  if (label_shape_.back() == 1) {
    // input shape: [N1, N2, ... , Nk, C]
    // label shape: [N1, N2, ... , Nk, 1]
    // reshape label shape to [N1, N2, ... , Nk]
    std::vector<int64_t> new_shape_{label_shape_[0]};
    auto const_before_loss = CreateBaseOp(
        graph,
        node,
        "popart_constant",
        {},
        {},
        {{"value", new_shape_},
         {"dims",
          std::vector<int64_t>{static_cast<int64_t>(new_shape_.size())}},
         {"dtype", ONNXDataType::INT64}});

    auto reshape_op =
        CreateBaseOp(graph,
                     node,
                     "popart_reshape",
                     {cast_and_reshape, const_before_loss->outputs[0]},
                     {},
                     {});
    cast_and_reshape = reshape_op->outputs[0];
  }

  auto log = CreateBaseOp(
      graph, node, "popart_log", {GetInputVarNode("X", node)}, {}, {});
  bool reshape_back = reduction == 2 && label_shape_.back() == 1;
  final_loss_node =
      CreateBaseOp(graph,
                   node,
                   "popart_nllloss_v2",
                   {log->outputs[0], cast_and_reshape},
                   !(reshape_back || append_identity_loss)
                       ? std::vector<Node *>{GetOutputVarNode("Y", node)}
                       : std::vector<Node *>{},
                   {
                       {"reduction", reduction},
                       {"ignoreIndex", ignoreIndex},
                       {"inputIsLogProbability", true},
                   });

  if (reshape_back) {
    // reshape output to the shape of input label.
    auto const_after_loss = CreateBaseOp(
        graph,
        node,
        "popart_constant",
        {},
        {},
        {{"value", label_shape_},
         {"dims",
          std::vector<int64_t>{static_cast<int64_t>(label_shape_.size())}},
         {"dtype", ONNXDataType::INT64}});
    final_loss_node = CreateBaseOp(
        graph,
        node,
        "popart_reshape",
        {final_loss_node->outputs[0], const_after_loss->outputs[0]},
        append_identity_loss ? std::vector<Node *>{}
                             : std::vector<Node *>{GetOutputVarNode("Y", node)},
        {});
  }

  if (append_identity_loss) {
    final_loss_node = CreateIdentityLossOp(graph,
                                           node,
                                           final_loss_node->outputs,
                                           {GetOutputVarNode("Y", node)},
                                           2);
  }

  return final_loss_node;
}

Node *softmax_with_cross_entropy_handler(Graph *graph, Node *node) {
  // TODO(czr): reuse cross_entropy2 code.
  auto *op = node->Op();
  auto ignoreIndex = BOOST_GET_CONST(int, op->GetAttr("ignore_index"));
  Node *cast_and_reshape = nullptr;
  Node *final_loss_node = nullptr;
  int reduction = RemoveTailReduction(graph, node, "Loss");
  auto axis = BOOST_GET_CONST(int, op->GetAttr("axis"));
  auto soft_label = BOOST_GET_CONST(bool, op->GetAttr("soft_label"));
  if (soft_label) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "soft_label is not supported yet in IPU"));
  }
  bool append_identity_loss = is_dynamic_graph();
  bool is_last_var_node = IsLastVarNode(GetOutputVarNode("Loss", node));
  append_identity_loss = append_identity_loss && is_last_var_node;
  if (GetInputVarNode("Label", node)->Var()->GetDataType() ==
      framework::proto::VarType::INT32) {
    cast_and_reshape = GetInputVarNode("Label", node);
  } else {
    auto cast_op = CreateCast(graph,
                              node,
                              {GetInputVarNode("Label", node)},
                              {},
                              framework::proto::VarType::INT32);
    cast_and_reshape = cast_op->outputs[0];
  }
  auto softmax_node = CreateSoftmaxOpset11(graph,
                                           node,
                                           {GetInputVarNode("Logits", node)},
                                           {GetOutputVarNode("Softmax", node)},
                                           axis);

  auto label_shape_ = GetInputVarNode("Label", node)->Var()->GetShape();
  if (label_shape_.back() == 1) {
    std::vector<int64_t> new_shape_{label_shape_[0]};
    auto const_before_loss = CreateBaseOp(
        graph,
        node,
        "popart_constant",
        {},
        {},
        {{"value", new_shape_},
         {"dims",
          std::vector<int64_t>{static_cast<int64_t>(new_shape_.size())}},
         {"dtype", ONNXDataType::INT64}});

    auto reshape_op =
        CreateBaseOp(graph,
                     node,
                     "popart_reshape",
                     {cast_and_reshape, const_before_loss->outputs[0]},
                     {},
                     {});
    cast_and_reshape = reshape_op->outputs[0];
  }

  auto log = CreateBaseOp(
      graph, node, "popart_log", {softmax_node->outputs[0]}, {}, {});
  bool reshape_back = reduction == 2 && label_shape_.back() == 1;
  final_loss_node =
      CreateBaseOp(graph,
                   node,
                   "popart_nllloss_v2",
                   {log->outputs[0], cast_and_reshape},
                   !(reshape_back || append_identity_loss)
                       ? std::vector<Node *>{GetOutputVarNode("Loss", node)}
                       : std::vector<Node *>{},
                   {
                       {"reduction", reduction},
                       {"ignoreIndex", ignoreIndex},
                       {"inputIsLogProbability", true},
                   });

  if (reshape_back) {
    // reshape output to the shape of input label.
    auto const_after_loss = CreateBaseOp(
        graph,
        node,
        "popart_constant",
        {},
        {},
        {{"value", label_shape_},
         {"dims",
          std::vector<int64_t>{static_cast<int64_t>(label_shape_.size())}},
         {"dtype", ONNXDataType::INT64}});

    final_loss_node = CreateBaseOp(
        graph,
        node,
        "popart_reshape",
        {final_loss_node->outputs[0], const_after_loss->outputs[0]},
        append_identity_loss
            ? std::vector<Node *>{}
            : std::vector<Node *>{GetOutputVarNode("Loss", node)},
        {});
  }
  if (append_identity_loss) {
    final_loss_node = CreateIdentityLossOp(graph,
                                           node,
                                           final_loss_node->outputs,
                                           {GetOutputVarNode("Loss", node)},
                                           2);
  }
  return final_loss_node;
}

}  // namespace
}  // namespace ipu
}  // namespace platform
}  // namespace paddle

REGISTER_HANDLER(identity_loss, identity_loss_handler);
REGISTER_HANDLER(softmax_with_cross_entropy,
                 softmax_with_cross_entropy_handler);
REGISTER_HANDLER(cross_entropy2, cross_entropy2_handler);
