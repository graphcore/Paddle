// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ipu/popart_canonicalization_utils.h"

namespace paddle {
namespace framework {

// This avoids the static initialisation order fiasco,
std::unordered_map<std::string, SymbolHandler> &SymbolHandlers() {
  static std::unordered_map<std::string, SymbolHandler> symbol_handlers;
  return symbol_handlers;
}

bool RegisterHandler(const std::string &symbol, const SymbolHandler &handler) {
  bool new_handler = SymbolHandlers().emplace(symbol, handler).second;
  return new_handler;
}

// Return a pointer to a handler if one is registered for this kind of node or
// an empty std::function otherwise.
SymbolHandler GetHandler(const std::string &kind) {
  auto it = SymbolHandlers().find(kind);
  if (it != SymbolHandlers().end()) {
    return it->second;
  }
  return {};
}

}  // namespace framework
}  // namespace paddle
