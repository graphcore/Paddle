#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestRelu(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [x.dtype for x in self.feed.values()]

    def _test_base(self, run_ipu=True):
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED

        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                # check op name in generated onnx-model/popartIR
                with paddle.static.name_scope('test'):
                    x = paddle.static.data(
                        name=self.feed_list[0],
                        shape=self.feed_shape[0],
                        dtype=self.feed_dtype[0])
                    x = paddle.fluid.layers.relu(x, **self.attrs)

                with paddle.static.name_scope('another_test'):
                    out = paddle.fluid.layers.tanh(x, **self.attrs)

                fetch_list = [out.name]

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()

            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if run_ipu:
                feed_list = self.feed_list
                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.set_graph_config(is_training=self.is_training)
                program = paddle.static.IpuCompiledProgram(
                    main_prog,
                    ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
            else:
                program = main_prog

            result = exe.run(program, feed=self.feed, fetch_list=fetch_list)
            return result[0]

    def run_test_base(self):
        res0 = self._test_base(False)
        res1 = self._test_base(True)

        self.assertTrue(
            np.allclose(
                res0.flatten(), res1.flatten(), atol=self.atol))

        self.assertTrue(res0.shape == res1.shape)

    def test_case0(self):
        self.feed = {
            "x": np.random.uniform(size=[1, 3, 10, 10]).astype('float32'),
        }
        self.attrs = {}
        self.set_feed_attr()
        self.run_test_base()


if __name__ == "__main__":
    unittest.main()
