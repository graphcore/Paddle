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

// clang-format off

#pragma once

OP_DECL(popart_nllloss_v2, aiGraphcoreOpset.nllloss, SIG_ARG(INT32,popart::ReductionType,reduction) OPT_ARG(INT32,ignoreIndex) ARG(BOOL,inputIsLogProbability) ) // NOLINT
OP_DECL(popart_identity_loss, aiGraphcoreOpset.identityloss, SIG_ARG(INT32,popart::ReductionType,reduction) ) // NOLINT

// clang-format on
