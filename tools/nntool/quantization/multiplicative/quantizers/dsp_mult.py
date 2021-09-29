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

import numpy as np
from graph.types import RFFT2DPreprocessingParameters
from quantization.new_qrec import QRec
from quantization.qtype import QType
from quantization.unified_quantization_handler import params_type, in_qs_constraint

from ..mult_quantization_handler import MultQuantizionHandler

WINDOW_Q = 15
FFT_TWIDDLES_Q = 15

@params_type(RFFT2DPreprocessingParameters)
@in_qs_constraint({'dtype': np.int16})
class RFFTPreprocessingMult(MultQuantizionHandler):
    @classmethod
    def _quantize(cls, params, in_qs, stats, **kwargs):
        force_out_qs, _ = cls.get_mult_opts(**kwargs)
        force_out_q = force_out_qs and force_out_qs[0]
        if force_out_q:
            return None

        win_q = QType(q=WINDOW_Q, dtype=np.int16)
        fft_twiddles_q = QType(q=FFT_TWIDDLES_Q, dtype=np.int16)
        swap_table_q = QType(q=0, dtype=np.int16)
        in_q = in_qs[0]
        in_q = QType.from_min_max_pow2(in_q.min_val, in_q.max_val, dtype=np.int16)
        in_q.q = in_q.q - 1
        if params.is_radix4():
            fft_out_q = in_q.q - 2*(int(np.log(params.n_cfft) / np.log(4)) - 2) - 1
        else:
            fft_out_q = in_q.q - (int(np.log2(params.n_cfft)) - 3) - 1
        out_q = QType(q=fft_out_q, dtype=np.int16)
        rfft_twiddles_q = QType(q=FFT_TWIDDLES_Q, dtype=np.int16)
        return QRec.symmetric(in_qs=[in_q, win_q, fft_twiddles_q, swap_table_q, rfft_twiddles_q], out_qs=[out_q])
