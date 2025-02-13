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

#include "paddle/fluid/framework/ir/ipu/inference_dtype_transfer_pass.h"

#include "paddle/fluid/platform/device/ipu/ipu_backend.h"

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/phi/common/place.h"

namespace paddle {
namespace framework {
namespace ir {

void InferenceDtypeTransferPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter InferenceDtypeTransferPass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  auto* ipu_backend = platform::ipu::IpuBackend::GetInstance();
  auto enable_fp16 = ipu_backend->GetIpuStrategy()->enable_fp16;

  if (enable_fp16) {
    VLOG(10) << "Transfer var to fp16...";
    auto* scope = ipu_backend->GetScope();

    for (auto* node : graph->Nodes()) {
      if (node->IsVar()) {
        auto var_desc = node->Var();
        if (var_desc->GetDataType() == proto::VarType::FP32) {
          // Transfer the dtypes of var_desc
          var_desc->SetDataType(proto::VarType::FP16);
          VLOG(10) << "Transfer the VarDesc of " << var_desc->Name() << " to "
                   << var_desc->GetDataType();

          if (node->inputs.empty() && node->Var()->Persistable() &&
              scope->FindVar(var_desc->Name())) {
            // Transfer the dtypes of weight tensors
            std::vector<float16> fp16_data;
            auto* tensor = scope->FindVar(var_desc->Name())
                               ->GetMutable<framework::LoDTensor>();
            auto* data_ptr = tensor->data<float>();
            auto num_elem = tensor->numel();

            std::transform(data_ptr, data_ptr + num_elem,
                           std::back_inserter(fp16_data),
                           [&](float elem) { return float16(elem); });
            memcpy(reinterpret_cast<void*>(data_ptr), fp16_data.data(),
                   num_elem * sizeof(float16));
            tensor->set_type(
                framework::TransToPhiDataType(proto::VarType::FP16));
          }
        }
      }
      if (node->IsOp()) {
        auto* op_desc = node->Op();
        if (op_desc->Type() == "popart_cast") {
          // Transfer the target dtype of cast Op
          if (BOOST_GET_CONST(std::string, op_desc->GetAttr("to")) == "FLOAT") {
            op_desc->SetAttr("to", std::string("FLOAT16"));
            op_desc->Flush();
          }
        }
        if (op_desc->Type() == "popart_constant") {
          // Transfer the dtype of fill_constant Op
          if (op_desc->GetAttrIfExists<int>("dtype") == 1) {
            op_desc->SetAttr("dtype", 10);
            op_desc->Flush();
          }
        }
      }
    }
    VLOG(10) << "Transfer var to fp16...Done";
  }

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave InferenceDtypeTransferPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inference_dtype_transfer_pass,
              paddle::framework::ir::InferenceDtypeTransferPass);
