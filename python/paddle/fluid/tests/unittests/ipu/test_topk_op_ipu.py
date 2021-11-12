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
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import paddle.optimizer
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest
import paddle.fluid.contrib.mixed_precision.fp16_utils as fp16_utils

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestTopKOp(IPUOpTest):
    def setUp(self):
        self.set_ops()
        self.set_atol()
        self.set_training()
        self.k = 3
        self.use_K_as_const_variable = False

        self.set_feed()
        self.set_attrs()

    def set_ops(self):
        self.ops = [
            paddle.fluid.layers.topk,
            paddle.topk  # top_k_v2
        ]

    def set_feed(self):
        self.feed_shape = []
        self.feed_shape.append([3, 5])

        self.feed = {}
        self.feed_list = []
        self.feed["in_0"] = np.random.uniform(
            size=self.feed_shape[0]).astype(np.float32)
        self.feed_list.append("in_0")
        if self.use_K_as_const_variable:
            # self.feed["in_1"] = np.array([self.k]).astype("int32")
            # self.feed_list.append("in_1")
            pass
        print("[TestTopKop] feed data:\n%s" % self.feed["in_0"])

    def set_attrs(self):
        self.attrs = {
            # "axis": -1,
            # "sorted": True
        }
        if not self.use_K_as_const_variable:
            self.attrs["k"] = self.k

    def _test_base(self, run_mode=0, op=None, data_feed=None):
        assert (op is not None)
        assert (data_feed is not None)
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = self.SEED
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        dtype = 'float32' if run_mode != self.TEST_IPU_FP16 else 'float16'
        self.feed_fp16 = {"in_0": self.feed["in_0"].astype(np.float16)}
        feed = self.feed_fp16 if run_mode == self.TEST_IPU_FP16 else self.feed

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                with paddle.static.amp.fp16_guard():
                    x = paddle.static.data(
                        name=self.feed_list[0],
                        shape=self.feed_shape[0],
                        dtype=dtype)
                    if not self.use_K_as_const_variable:
                        topk_values, topk_indices = op(x, **self.attrs)
                    else:
                        # !important, popart cannot accept non const tensor
                        # K_t = paddle.static.data(name="in_1", shape=[1], dtype='int32')
                        K_t = fluid.layers.fill_constant(
                            shape=[1], dtype='int32', value=self.k, name="in_2")
                        topk_values, topk_indices = op(x, K_t, **self.attrs)
                    fetch_list = [topk_values.name, topk_indices.name]

            if run_mode != self.TEST_CPU_FP32:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            if run_mode == self.TEST_IPU_FP16:
                fp16_utils.rewrite_program_v2(
                    startup_prog=startup_prog,
                    main_prog=main_prog,
                    amp_lists=self.amp_list)
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if run_mode != self.TEST_CPU_FP32:
                feed_list = self.feed_list
                ipu_strategy = compiler.get_ipu_strategy()
                ipu_strategy.is_training = self.is_training
                program = compiler.IpuCompiler(
                    main_prog,
                    ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
            else:
                program = main_prog

            print("Running inference ...")
            result = exe.run(program, feed=feed, fetch_list=fetch_list)
            print("Complete running infrence.")
            return result

    def test_base(self):
        for op in self.ops:
            res1_topk_values, res1_topk_indices = self._test_base(
                self.TEST_CPU_FP32,
                op=paddle.fluid.layers.topk,
                data_feed=self.feed)
            res0_topk_values, res0_topk_indices = self._test_base(
                self.TEST_IPU_FP32, op=op, data_feed=self.feed)
            res2_topk_values, res2_topk_indices = self._test_base(
                self.TEST_IPU_FP16,
                op=paddle.fluid.layers.topk,
                data_feed=self.feed)

            print("[TestTopKop] IPU fp32 res0 values:\n%s\n" % res0_topk_values)
            print("[TestTopKop] CPU res1 values:\n%s\n" % res1_topk_values)
            print("[TestTopKop] IPU fp16 res2 values:\n%s\n" % res2_topk_values)

            view_type = np.uint32
            print("[TestTopKop] IPU res0 fp32 indices:\n%s\n" %
                  res0_topk_indices.astype(view_type))
            print("[TestTopKop] CPU res1 indices:\n%s\n" % res1_topk_indices)
            print("[TestTopKop] CPU res1 fp16 indices:\n%s\n" %
                  res2_topk_indices)

            self.assertTrue(
                np.allclose(
                    res0_topk_values.flatten(),
                    res1_topk_values.flatten(),
                    atol=self.atol))

            self.assertTrue(
                np.allclose(
                    res0_topk_indices.astype(view_type).flatten(),
                    res1_topk_indices.flatten(),
                    atol=self.atol))
            self.assertTrue(
                np.allclose(
                    res0_topk_indices.astype(view_type).flatten(),
                    res2_topk_indices.flatten(),
                    atol=self.atol))


if __name__ == "__main__":
    unittest.main()
