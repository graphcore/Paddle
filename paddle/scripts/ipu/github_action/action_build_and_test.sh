#!/usr/bin/env bash

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

echo "workdir: $PWD"

mkdir /paddle_build

# install requirements
pip install -r /checkout/python/requirements.txt

# todo use ninja
cmake \
    -DCMAKE_INSTALL_PREFIX:STRING=install \
    -DPYTHON_ABI=conda-python3.7 \
    -DPY_VERSION:STRING=3.7 \
    -DWITH_GPU:STRING=OFF \
    -DWITH_NCCL:STRING=OFF \
    -DWITH_MKL:STRING=OFF \
    -DWITH_TESTING:STRING=ON \
    -DWITH_INFERENCE_API_TEST:SRTING=ON \
    -DWITH_PYTHON:STRING=ON \
    -DON_INFER:STRING=ON \
    -DWITH_IPU:STRING=ON \
    -DPOPLAR_DIR:STRING=/opt/poplar \
    -DPOPART_DIR:STRING=/opt/popart \
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++ \
    -G "Unix Makefiles" \
    -H/checkout \
    -B/paddle_build

cmake --build /paddle_build --config Release --target all -j 25
# cmake --build /paddle_build --config Release --target all -j `grep -c ^processor /proc/cpuinfo`

# ccache
ccache -s

# install paddle
# ls /paddle_build/python/dist/*.whl | xargs pip install

# run gc-monitor
gc-monitor

# check paddle path
export PYTHONPATH=/paddle_build/python:$PYTHONPATH
python -c "import paddle; print(paddle.__file__)"

# create tar to `paddle_wheels`
tar czf /paddle_wheels/paddle_ipu_${GITHUB_SHA}.tar.gz /paddle_build/python/dist/*.whl
echo "create paddle wheel file: /paddle_wheels/paddle_ipu_${GITHUB_SHA}.tar.gz"
ls -lh /paddle_wheels/paddle_ipu_${GITHUB_SHA}.tar.gz

# run unittests
cd /paddle_build/python
# install `pytest-xdist`
pip -V
pip install pytest-xdist
pytest -VV
pytest \
    -o cache_dir=paddle_build/pytest_cache \
    -n=3 \
    --maxfail=3 \
    paddle/fluid/tests/unittests/ipu/ \
    paddle/fluid/tests/unittests/ipu/inference/
