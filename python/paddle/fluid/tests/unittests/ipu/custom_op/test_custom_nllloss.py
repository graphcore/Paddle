#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import paddle.optimizer
import paddle.static
from paddle.utils.cpp_extension import load

paddle.enable_static()

map_np_dtype_to_fluid_dtype = {
    'bool': "bool",
    'int8': "int8",
    'uint8': "uint8",
    "int32": "int32",
    "int64": "int64",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
}


def np_dtype_to_fluid_str(dtype: np.dtype) -> str:
    return map_np_dtype_to_fluid_dtype[dtype.name]


def load_custom_ops():
    # for successful compilation using `paddle.utils.cpp_extension.load`
    # we need a empty paddle custom op which defined in `custom_nop_op.cc`
    # the custom popart pattern is defined in `custom_popart_pattern.cc`
    custom_ops = load(
        name="custom_ops",
        sources=["./custom_nllloss.cc"],
        extra_cxx_cflags=['-DONNX_NAMESPACE=onnx'],
        extra_ldflags=['-lpopfloat'])
    return custom_ops


@unittest.skip("custom op nllloss")
class TestBase(unittest.TestCase):
    def setUp(self):
        self.set_feed()
        self.set_feed_attr()
        self.set_op()

    def set_op(self):
        # setup custom op
        self.op = custom_ops.custom_nll_loss

    def set_feed(self):
        self.feed = {
            "Label": np.random.uniform(
                0, 30400, size=[16, 20]).astype('int32'),
            "X": np.random.rand(16, 20, 30400).astype('float32'),
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [
            np_dtype_to_fluid_str(x.dtype) for x in self.feed.values()
        ]

    def test_base(self):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = 0
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                feeds = []
                for idx in range(len(self.feed_list)):
                    x = paddle.static.data(
                        name=self.feed_list[idx],
                        shape=self.feed_shape[idx],
                        dtype=self.feed_dtype[idx])
                    feeds.append(x)

                out = self.op(feeds[1], feeds[0], 0, "0", False)
                out = paddle.mean(out)
                fetch_list = [out.name]

            place = paddle.IPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            feed_list = self.feed_list
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(num_ipus=1, is_training=False)
            ipu_compiler = compiler.IpuCompiledProgram(
                main_prog, ipu_strategy=ipu_strategy)
            program = ipu_compiler.compile(feed_list, fetch_list)

            res = exe.run(program, self.feed, fetch_list)
            print(res)


@unittest.skip("custom op nllloss")
class TestTraining(TestBase):
    def setUp(self):
        self.set_feed()
        self.set_feed_attr()
        self.set_op()

    def set_op(self):
        # setup custom op
        self.op = custom_ops.custom_nll_loss

    def set_feed(self):
        self.feed = {
            "x":
            np.array([[[1], [3]], [[2], [4]], [[4], [127]]]).astype('int32'),
            "label": np.random.uniform(
                0, 30, size=[3, 2]).astype('int32'),
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [
            np_dtype_to_fluid_str(x.dtype) for x in self.feed.values()
        ]

    def test_base(self):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = 0
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(
                    name=self.feed_list[0],
                    shape=self.feed_shape[0],
                    dtype='int64')
                label = paddle.static.data(
                    name="label", shape=[3, 2], dtype="int32")

                out = paddle.fluid.layers.embedding(
                    x, size=[128, 16], dtype="float32")
                loss = self.op(out, label, 0, "0", False)
                adam = paddle.optimizer.Adam(learning_rate=1e-2)
                adam.minimize(loss)
                fetch_list = [loss.name]

            place = paddle.IPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            feed_list = [x.name, label.name]
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(num_ipus=1, is_training=True)
            ipu_compiler = compiler.IpuCompiledProgram(
                main_prog, ipu_strategy=ipu_strategy)
            program = ipu_compiler.compile(feed_list, fetch_list)

            res = exe.run(program, self.feed, fetch_list)
            print(res)

@unittest.skip("custom op nllloss")
class TestIgnoreIndexNone(TestBase):
    def setUp(self):
        self.set_feed()
        self.set_feed_attr()
        self.set_op()

    def set_op(self):
        # setup custom op
        self.op = custom_ops.custom_nll_loss

    def set_feed(self):
        self.feed = {
            "x":
            np.array([[[1], [3]], [[2], [4]], [[4], [127]]]).astype('int32'),
            "label": np.random.uniform(
                0, 30, size=[3, 2]).astype('int32'),
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [
            np_dtype_to_fluid_str(x.dtype) for x in self.feed.values()
        ]

    def test_base(self):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = 0
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(
                    name=self.feed_list[0],
                    shape=self.feed_shape[0],
                    dtype='int64')
                label = paddle.static.data(
                    name="label", shape=[3, 2], dtype="int32")

                out = paddle.fluid.layers.embedding(
                    x, size=[128, 16], dtype="float32")
                loss = self.op(out, label, 0, "None", False)
                adam = paddle.optimizer.Adam(learning_rate=1e-2)
                adam.minimize(loss)
                fetch_list = [loss.name]

            place = paddle.IPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            feed_list = [x.name, label.name]
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(num_ipus=1, is_training=True)
            ipu_compiler = compiler.IpuCompiledProgram(
                main_prog, ipu_strategy=ipu_strategy)
            program = ipu_compiler.compile(feed_list, fetch_list)

            res = exe.run(program, self.feed, fetch_list)
            print(res)


if __name__ == "__main__":
    custom_ops = load_custom_ops()
    unittest.main()
