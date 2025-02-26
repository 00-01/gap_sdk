# Copyright (C) 2020  GreenWaves Technologies, SAS

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
from functools import reduce
from utils.node_id import NodeId

from generation.at_types.at_params import NO_ACTIVATION, gen_active_at_params
from generation.at_types.gen_ctrl import GenCtrl
from generation.code_block import CodeBlock
from generation.generator_decorators import QREC_MULT8, generation_function
from graph.types import (ActivationFusionBase, GlobalAveragePoolParameters,
                         GlobalMaxPoolParameters, GlobalPoolingParameters,
                         GlobalSumPoolParameters)
from utils.largest_factor import balanced_divisors

from ..autotiler_kernel import (AutotilerKernel, gen_include_paths,
                                gen_includes, gen_sources,
                                kernel_include_paths, kernel_includes,
                                kernel_sources)
from ..pow2.global_pool_kernels_generator import gen_globalpool_at_params

LOG = logging.getLogger("nntool." + __name__)


@generation_function("kernels", (GlobalAveragePoolParameters, GlobalMaxPoolParameters, GlobalSumPoolParameters, ActivationFusionBase), qrec_types=(QREC_MULT8, ))
def global_pool_kernels_generator(gen, node, qrec, in_eparams, out_eparams, cname):
    del in_eparams, out_eparams, qrec
    if isinstance(node, ActivationFusionBase):
        cnodes = node.contained_nodes()
        if isinstance(cnodes[0], GlobalPoolingParameters):
            act_q = gen.G.quantization[NodeId(node, cnodes[1])]
            gen.kernels.append(GlobalPoolKernel(
                node.name, cname, cnodes[0], cnodes[1], act_q, at_ver=gen.opts['at_ver'], force_relu=gen.force_relu))
            return True
        return False
    gen.kernels.append(GlobalPoolKernel(node.name, cname, node,
                                        None, at_ver=gen.opts['at_ver'], force_relu=gen.force_relu))
    return True


def gen_cnn_globalpool_sq8(code_block, cname, ctrl, feat, width, height, pooloper, actoper):
    code_block.write('CNN_GlobalPoolAct_SQ8("{}", {}, {}, {}, {}, {}, {});'.format(cname, ctrl,
                                                                                   feat, width,
                                                                                   height, pooloper,
                                                                                   actoper))


@kernel_sources(
    '$(TILER_CNN_KERNEL_PATH_SQ8)/CNN_Pooling_SQ8.c')
@kernel_include_paths(
    '$(TILER_CNN_KERNEL_PATH)',
    '$(TILER_CNN_KERNEL_PATH_SQ8)')
@kernel_includes(
    'CNN_BasicKernels_SQ8.h')
@gen_sources(
    '$(TILER_CNN_GENERATOR_PATH)/CNN_Generator_Util.c',
    '$(TILER_CNN_GENERATOR_PATH_SQ8)/CNN_Generators_SQ8.c')
@gen_include_paths(
    '$(TILER_CNN_GENERATOR_PATH)',
    '$(TILER_CNN_GENERATOR_PATH_SQ8)')
@gen_includes(
    'CNN_Generators_SQ8.h')
class GlobalPoolKernel(AutotilerKernel):
    def __init__(self, node_name, cname, pool_params, act_params, act_q=None, gen_ctrl=None, at_ver=3, force_relu=True):
        if gen_ctrl is None:
            self.gen_ctrl = GenCtrl(None, cname=cname)
        else:
            gen_ctrl.cname = cname
            self.gen_ctrl = gen_ctrl
        if pool_params.ker_in_order and pool_params.ker_in_order[0] == ["h", "w", "c"]:
            self.gen_ctrl.hwc = 1

        if act_params is not None:
            self.at_act_params = gen_active_at_params(
                act_params, force_relu=force_relu,
                asymmetric=act_q.in_qs[0].zero_point != 0)
        else:
            self.at_act_params = NO_ACTIVATION

        self.at_globalpool_params = gen_globalpool_at_params(pool_params)
        in_dim = pool_params.in_dims[0]
        reduce_sz = reduce(lambda x, y: x * y, (sz for idx, sz in enumerate(in_dim.shape)
                                                if idx not in pool_params.axis), 1)
        #self.c = in_dim.size()/reduce_sz
        self.c = reduce_sz
        (self.h, self.w) = balanced_divisors(in_dim.size()/reduce_sz)
        self.cname = cname
        self.node_name = node_name
        self.at_ver = at_ver

    def code(self, code_block=None):
        if code_block is None:
            code_block = CodeBlock()

        code_block.comment("generator for {}", self.node_name)

        if not self.gen_ctrl.is_unmodified:
            self.gen_ctrl.gen_ctrl_decl(code_block)
            gen_ctrl = self.gen_ctrl.ctrl_name
        else:
            gen_ctrl = "0"

        gen_cnn_globalpool_sq8(code_block, self.cname, gen_ctrl, self.c,
                               self.h, self.w,
                               self.at_globalpool_params.GlobalPoolOper,
                               self.at_act_params.ReLUOper)
        return code_block
