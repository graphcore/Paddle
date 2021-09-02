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

from __future__ import print_function

import numpy as np
import unittest
import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestAdd(unittest.TestCase):
    def _test_add(self, run_ipu=True):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        np_a = np.random.rand(3, 3, 3).astype(np.float32)
        np_b = np.arange(1, 4).reshape([3]).astype(np.float32)
        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                a = paddle.static.data(
                    name="a",
                    shape=[3, 3, 3],
                    dtype='float32', )
                b = paddle.static.data(
                    name="b",
                    shape=[3],
                    dtype='float32', )
                # out = paddle.fluid.layers.elementwise_add(a, b, axis=-1)
                # out = paddle.fluid.layers.elementwise_add(a, b, axis=0)
                # out = paddle.fluid.layers.elementwise_add(a, b, axis=1)
                out = paddle.fluid.layers.elementwise_add(a, b, axis=2)

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if run_ipu:
                feed_list = [a.name, b.name]
                fetch_list = [out.name]
                ipu_strategy = compiler.get_ipu_strategy()
                ipu_strategy.is_training = False
                program = compiler.IpuCompiler(
                    main_prog, ipu_strategy=ipu_strategy).compile(feed_list,
                                                                  fetch_list)
            else:
                program = main_prog

            result = exe.run(
                program,
                feed={'a': np_a,
                      'b': np_b},
                fetch_list=[out], )
            return result[0]

    def test_add(self):
        ipu_res = self._test_add(True)
        cpu_res = self._test_add(False)

        self.assertTrue(np.allclose(ipu_res, cpu_res, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
