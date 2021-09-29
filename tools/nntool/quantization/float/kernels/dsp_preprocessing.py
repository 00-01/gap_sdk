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

import numpy as np
from graph.types import MFCCPreprocessingParameters, RFFT2DPreprocessingParameters
from quantization.kernels.kernel_base import KernelBase, params_type, qrec_type
from quantization.multiplicative.mulbias import (apply_multiplicative_bias,
                                                 apply_zero_offset_bias)
from quantization.new_qrec import QRec
from utils.at_norm import at_norm

# pylint: disable=invalid-name

LOG = logging.getLogger("nntool." + __name__)

@params_type(RFFT2DPreprocessingParameters)
@qrec_type('float')
class Rfft2Dfloat(KernelBase):
    @classmethod
    def execute(cls, params,
                in_tensors,
                qrec: QRec,
                **kwargs):
        in_data = in_tensors[0]
        fft_twiddles = in_tensors[2]
        swap_table = in_tensors[3]
        if params.preemp_factor:
            # need preemp here
            in_data = in_data
        if params.win_fn:
            win_lut = in_tensors[1]
            in_data = in_data * win_lut

        fft_out = np.fft.rfft(in_data)
        return [np.stack((fft_out.real, fft_out.imag))]


@params_type(MFCCPreprocessingParameters)
@qrec_type('float')
class Mfcc2Dfloat(KernelBase):
    @classmethod
    def execute(cls, params,
                in_tensors,
                qrec: QRec,
                **kwargs):
        return in_tensors
