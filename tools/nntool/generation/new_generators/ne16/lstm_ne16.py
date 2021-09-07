# Copyright (C) 2021  GreenWaves Technologies, SAS

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

import numpy as np
from generation.at_types.constant_info import ConstantInfo
from generation.at_types.gen_ctrl import GenCtrl
from generation.at_types.tc_arg_info import (GlobalArgInfo, GlobalResetArgInfo,
                                             LocalArgInfo)
from generation.bindings import (CommentBindingList, GArgName, GNodeArgEdge,
                                 GNodeArgNode, NoArg, NodeBindingList)
from generation.generators.bindings.float16.rnn_bindings_generator import \
    num_sequences
from generation.generators.globals.global_names import INFOS
from generation.generators.kernels.autotiler_kernel import NewAutoTilerKernel
from generation.helpers.gen_constant import gen_constant
from generation.new_generators.generator_base import (GeneratorBase,
                                                      paramstype, qrec_options)
from graph.types import LSTMParameters
from quantization.qtype import QType
from utils.numpy_helpers import interleave

LOG = logging.getLogger("nntool." + __name__)

SIGMOID_TABLE = np.array([32768, 33451, 34133, 34813, 35493, 36169, 36843, 37513, 38180, 38841, 39498,
                          40149, 40794, 41432, 42064, 42688, 43304, 43912, 44511, 45102, 45683, 46255,
                          46817, 47369, 47911, 48443, 48964, 49475, 49975, 50464, 50942, 51409, 51865,
                          52311, 52745, 53169, 53581, 53983, 54374, 54755, 55125, 55485, 55834, 56174,
                          56503, 56823, 57133, 57433, 57724, 58007, 58280, 58544, 58800, 59048, 59288,
                          59519, 59743, 59959, 60168, 60370, 60565, 60753, 60935, 61110, 61279, 61441,
                          61599, 61750, 61896, 62036, 62172, 62302, 62428, 62549, 62666, 62778, 62886,
                          62990, 63090, 63186, 63279, 63368, 63454, 63536, 63615, 63691, 63765, 63835,
                          63903, 63968, 64030, 64090, 64148, 64204, 64257, 64308, 64357, 64405, 64450,
                          64494, 64536, 64576, 64614, 64652, 64687, 64721, 64754, 64786, 64816, 64845,
                          64873, 64900, 64926, 64950, 64974, 64997, 65019, 65039, 65060, 65079, 65097,
                          65115, 65132, 65149, 65164, 65179, 65194, 65208, 65221, 65234, 65246, 65258,
                          65269, 65280, 65291, 65301, 65310, 65319, 65328, 65337, 65345, 65352, 65360,
                          65367, 65374, 65381, 65387, 65393, 65399, 65404, 65410, 65415, 65420, 65425,
                          65429, 65433, 65438, 65442, 65445, 65449, 65453, 65456, 65459, 65462, 65465,
                          65468, 65471, 65474, 65476, 65479, 65481, 65483, 65485, 65488, 65489, 65491,
                          65493, 65495, 65497, 65498, 65500, 65501, 65503, 65504, 65505, 65507, 65508,
                          65509, 65510, 65511, 65512, 65513, 65514, 65515, 65516, 65517, 65517, 65518,
                          65519, 65520, 65520, 65521, 65522, 65522, 65523, 65523, 65524, 65524, 65525,
                          65525, 65526, 65526, 65526, 65527, 65527, 65528, 65528, 65528, 65529, 65529,
                          65529, 65529, 65530, 65530, 65530, 65530, 65531, 65531, 65531, 65531, 65531,
                          65532, 65532, 65532, 65532, 65532, 65532, 65533, 65533, 65533, 65533, 65533,
                          65533, 65533, 65533, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534,
                          65534, 65534, 65535], dtype=np.uint16)


@paramstype(LSTMParameters)
@qrec_options(ne16=True)
class LSTMNE16Generator(GeneratorBase):

    @classmethod
    def globals_generator(cls, gen, node, qrec, pnode, fnode) -> bool:
        names = {val: idx for idx, val in enumerate(
            LSTMParameters.INPUT_NAMES)}
        scales = []
        weight_zero = None
        for gate in ['i', 'c', 'f', 'o']:
            for input_tensor in ['i', 'r']:
                scale_name = f'{input_tensor}_2_{gate}_q'
                weight_name = f'{input_tensor}_2_{gate}_w'
                if weight_zero is None:
                    weight_zero = qrec.in_qs[names[weight_name]].zero_point[0]
                else:
                    assert weight_zero == qrec.in_qs[names[weight_name]
                                                     ].zero_point[0]
                w_q = qrec.in_qs[names['r_2_i_w']]
                qscale = qrec.cache[scale_name]
                scales.append(qscale.qbiases)
                scales.append(qscale.qnorms)

        contents = interleave(*scales)

        cname, file_name = gen_constant(gen, pnode, pnode, "scalenorm")
        const_info = ConstantInfo(file_name, QType.Pow2(
            bits=8, q=0, signed=False), contents=contents)
        gen.globals.append(GlobalArgInfo("uint8", cname,
                                         gen.opts['default_global_home_location'],
                                         gen.opts['default_global_exec_location'],
                                         const_info=const_info,
                                         comment=f"{node.name} scales and norms"))
        if node.rnn_states_as_inputs:
            gen.globals.append(GlobalResetArgInfo(
                f"{node.name}_Reset", 'AT_MEM_L2', 'AT_MEM_UNDEF'))

        out_q = qrec.out_qs[0]
        out_scale = qrec.cache["state_out_q"].qbiases[0]
        out_scalen = qrec.cache["state_out_q"].qnorms[0]
        cin_scale = qrec.cache["cell_in_q"].qbiases[0]
        cin_scalen = qrec.cache["cell_in_q"].qnorms[0]
        cout_scale = qrec.cache["cell_out_q"].qbiases[0]
        cout_scalen = qrec.cache["cell_out_q"].qnorms[0]
        out_zeropoint = out_q.zero_point[0]

# define LSTM_NE16_W_ZEROPOINT   0
# define LSTM_NE16_GATE_PRENORM  1
# define LSTM_NE16_CIN_SCALE     (0 + LSTM_NE16_OUT_OFF)
# define LSTM_NE16_CIN_SCALEN    (1 + LSTM_NE16_OUT_OFF)
# define LSTM_NE16_COUT_SCALE    (2 + LSTM_NE16_OUT_OFF)
# define LSTM_NE16_COUT_SCALEN   (3 + LSTM_NE16_OUT_OFF)
# define LSTM_NE16_OUT_SCALE     (4 + LSTM_NE16_OUT_OFF)
# define LSTM_NE16_OUT_SCALEN    (5 + LSTM_NE16_OUT_OFF)
# define LSTM_NE16_OUT_ZEROPOINT (6 + LSTM_NE16_OUT_OFF)

# define LSTM_NE16_INT_A0        (0 + LSTM_NE16_INT_OFF)
# define LSTM_NE16_INT_B0        (1 + LSTM_NE16_INT_OFF)
# define LSTM_NE16_INT_C0        (2 + LSTM_NE16_INT_OFF)

        sigmoid_table = interleave(
            SIGMOID_TABLE & 0xff, SIGMOID_TABLE >> 8).astype(np.int8)
        if out_q.dtype == np.uint8:
            # Maybe get rid of this
            if qrec.cache.get('act_qtype'):
                min_val = qrec.cache['act_qtype'].quantize(-1)
                max_val = qrec.cache['act_qtype'].quantize(1)
            else:
                min_val = max_val = 0
            contents = np.concatenate((
                sigmoid_table,
                np.array([
                    -weight_zero.astype(np.int8),
                    qrec.cache['gate_prenorm'],
                    cin_scale.astype(np.int8),
                    cin_scalen.astype(np.int8),
                    cout_scale.astype(np.int8),
                    cout_scalen.astype(np.int8),
                    out_scale.astype(np.int8),
                    out_scalen.astype(np.int8),
                    out_zeropoint.astype(np.int8),
                    0,
                    0,
                    0,
                    0
                ], dtype=np.int8)))
        else:
            contents = np.concatenate((
                sigmoid_table,
                np.array([
                    -weight_zero.astype(np.int8),
                    qrec.cache['gate_prenorm'],
                    cin_scale.astype(np.int8),
                    cin_scalen.astype(np.int8),
                    cout_scale.astype(np.int8),
                    cout_scalen.astype(np.int8),
                    out_scale.astype(np.int8),
                    out_scalen.astype(np.int8),
                    out_zeropoint.astype(np.uint16) & 0xff,
                    out_zeropoint.astype(np.uint16) >> 8,
                ], dtype=np.int8)))

        comment = (f"WZP: {weight_zero}, Out: {out_scale}/{out_scalen}, Cin: {cin_scale}/{cin_scalen}"
                   f"Cout: {cout_scale}/{cout_scalen}, OZP: {out_zeropoint}")
        cname, file_name = gen_constant(gen, pnode, pnode, INFOS)
        const_info = ConstantInfo(file_name, QType.Pow2(
            bits=8, q=0, signed=True), contents=contents)

        gen.globals.append(GlobalArgInfo("int8", cname,
                                         gen.opts['default_global_home_location'],
                                         gen.opts['default_global_exec_location'],
                                         const_info=const_info,
                                         comment=comment))

        if node.rnn_states_as_inputs:
            gen.globals.append(GlobalResetArgInfo(
                f"{node.name}_Reset", 'AT_MEM_L2', 'AT_MEM_UNDEF'))
        return True

    @classmethod
    def bindings_generator(cls, gen, node, qrec, in_eparams, out_eparams, cname) -> bool:
        names = node.get_name_indexes()

        gen.bindings.append(
            CommentBindingList("Node {} inq {} outq {}", cname,
                               qrec.in_qs[0],
                               qrec.out_qs[0])
        )
        num_seq = num_sequences(node)
        step_idx = node.step_idx
        in_ctype = "char" if qrec.in_qs[0].bits == 8 else "short"
        if num_seq > 1:
            gen.locals.append(LocalArgInfo(
                f"signed {in_ctype}", f"S{step_idx}_CellInternal01"))
            gen.locals.append(LocalArgInfo(
                f"unsigned {in_ctype}", f"S{step_idx}_StateInternal01"))

        if num_seq > 2:
            gen.locals.append(LocalArgInfo(
                f"signed {in_ctype}", f"S{step_idx}_CellInternal02"))
            gen.locals.append(LocalArgInfo(
                f"unsigned {in_ctype}", f"S{step_idx}_StateInternal02"))

        i_state_eparams = in_eparams[names['i_state']]
        c_state_eparams = in_eparams[names['c_state']]
        reset_name = i_state_eparams.creating_node.reset_name if node.rnn_states_as_inputs else "Reset"
        bindings = [
            GNodeArgEdge(c_state_eparams, direction="GNA_INOUT"),
            GNodeArgEdge(i_state_eparams, direction="GNA_INOUT"),
            GNodeArgEdge("S%s_CellInternal01" % step_idx, alias=c_state_eparams,
                         direction="GNA_INOUT") if num_seq > 1 else NoArg(),
            GNodeArgEdge("S%s_StateInternal01" % step_idx, alias=i_state_eparams,
                         direction="GNA_INOUT") if num_seq > 1 else NoArg(),
            GNodeArgEdge("S%s_CellInternal02" % step_idx, alias="S%s_CellInternal01" %
                         step_idx, direction="GNA_INOUT") if num_seq > 2 else NoArg(),
            GNodeArgEdge("S%s_StateInternal02" % step_idx, alias="S%s_CellInternal01" %
                         step_idx, direction="GNA_INOUT") if num_seq > 2 else NoArg(),
            GNodeArgEdge(in_eparams[0]),
            GNodeArgNode(node, "scalenorm")
        ]
        for gate in ['f', 'i', 'c', 'o']:
            for inp_t in ['r', 'i']:
                bindings.append(GNodeArgEdge(
                    in_eparams[names[f'{inp_t}_2_{gate}_w']]))
            bindings.append(GNodeArgEdge(in_eparams[names[f'{gate}_b']]))

        bindings.extend([
            GNodeArgEdge(out_eparams[0], direction="GNA_OUT"),
            GNodeArgNode(node, INFOS),
            GArgName(reset_name)
        ])

        gen.bindings.append(
            NodeBindingList(
                cname,
                *bindings))

        return True

    @classmethod
    def kernel_generator(cls, gen, node, qrec, in_eparams, out_eparams, cname) -> bool:
        del in_eparams, out_eparams
        gen.kernels.append(
            LSTMNE16Kernel(
                node.name, cname, node, qrec)
        )
        return True


class LSTMNE16Kernel(NewAutoTilerKernel):
    CALL_TEMPLATE = """
// generator for {node_name}
LSTM_Stack_NE16("{cname}", {gen_ctrl}, {bias_size}, {feat_size}, {filter_bits},
                 {n_cells}, {k0}, {k1}, {dim_state}, {dim_in}, {always_reset}, {revert});
"""

    def __init__(self, node_name, cname, params, qrec, gen_ctrl=None):
        if gen_ctrl is None:
            gen_ctrl = GenCtrl(None, cname=cname)
        else:
            gen_ctrl.cname = cname

        if params.hard_act:
            gen_ctrl.rnn_use_hardact = 1
            gen_ctrl.gate_prenorm = qrec.cache['i_2_f_q'].pre_normalization

        names = {val: idx for idx, val in enumerate(
            LSTMParameters.INPUT_NAMES)}
        in_qs = qrec.in_qs

        w_bits = None
        for gate in ['f', 'i', 'c', 'o']:
            for inp_t in ['r', 'i']:
                if w_bits is None:
                    w_bits = in_qs[names[f'{inp_t}_2_{gate}_w']].bits
                elif w_bits != in_qs[names[f'{inp_t}_2_{gate}_w']].bits:
                    ValueError(f'bit width of gates differs in {params.name}')

        attrs = {
            'bias_size': in_qs[names['i_b']].bits//8,
            'feat_size': -in_qs[0].bits//8,
            'filter_bits': w_bits,
            'n_cells': params.n_cells,
            'k0': params.n_input_cells,
            'k1': params.n_output_cells,
            'dim_state': params.n_states,
            'dim_in': params.n_inputs,
            'always_reset': 0,
            'revert': 1 if params.revert else 0,
        }

        extra_attrs = {
            'cname': cname,
            'node_name': node_name
        }
        super().__init__(attrs, extra_attrs, gen_ctrl=gen_ctrl)
