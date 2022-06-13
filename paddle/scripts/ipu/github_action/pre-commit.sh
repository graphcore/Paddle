# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PWD must be /checkout
echo "workdir: $PWD"

# add safe.directory for git
git config --global --add safe.directory /checkout

# versions may change in the future
pip cache purge
pip install pip --upgrade
pip install clang-format==13.0.0
pip install cpplint==1.6.0
pip install pylint==2.12.0

# install pre-commit
pre-commit install

echo -e "show PR_CHANGED_FILES"
cat $PR_CHANGED_FILES

# run pre-commit
echo -e "start run pre-commit \n\n"
cat $PR_CHANGED_FILES | xargs pre-commit run --show-diff-on-failure --files
