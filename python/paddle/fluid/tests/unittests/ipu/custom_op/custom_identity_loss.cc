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

#include "paddle/extension.h"

std::vector<paddle::Tensor> Kernel_Function(const paddle::Tensor& x) { return {x}; }
std::vector<paddle::Tensor> Kernel_Function_Grad(const paddle::Tensor& x) { return {x}; }

std::vector<std::vector<int64_t>> InferShape_IdentityLoss(
    std::vector<int64_t> x_shape,
    const int& reduction) {
  // 0: sum, 1: mean, 2: none
  if (reduction == 2) {
    return {x_shape};
  } else {
    return {{1}};
  }
}

std::vector<paddle::DataType> InferDtype_IdentityLoss(paddle::DataType x_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(identity_loss)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"reduction: int"})
    .SetKernelFn(PD_KERNEL(Kernel_Function))
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape_IdentityLoss))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype_IdentityLoss));

PD_BUILD_GRAD_OP(identity_loss)
    .Inputs({paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(Kernel_Function_Grad));
