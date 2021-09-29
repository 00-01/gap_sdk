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
from copy import deepcopy
import math

from ..dim import Dim
from .base import SensitiveToOrder, SingleInputAndOutput, Transposable, Parameters, cls_op_name

LOG = logging.getLogger("nntool." + __name__)


@cls_op_name("RFFT2D")
class RFFT2DPreprocessingParameters(Parameters, SingleInputAndOutput, SensitiveToOrder):

    def __init__(self, name: str, n_fft, preemp_factor=0.0, magsquared=True, frame_axis=None, window="hanning", **kwargs):
        super(RFFT2DPreprocessingParameters, self).__init__(name, **kwargs)
        self._win_fn = window
        self._frame_axis = frame_axis
        self._magsquared = magsquared
        self._preemp_factor = preemp_factor
        self._force_rad2 = False
        self._n_fft = n_fft
        self._n_cfft = n_fft // 2

    @property
    def frame_axis(self):
        return self._frame_axis

    @frame_axis.setter
    def frame_axis(self, val):
        self._frame_axis = val

    @property
    def preemp_factor(self):
        return self._preemp_factor

    @preemp_factor.setter
    def preemp_factor(self, val):
        self._preemp_factor = val

    @property
    def n_fft(self):
        return self._n_fft

    @n_fft.setter
    def n_fft(self, val):
        self._n_fft = val

    @property
    def n_cfft(self):
        return self._n_cfft

    @n_cfft.setter
    def n_cfft(self, val):
        self._n_cfft = val

    @property
    def win_fn(self):
        return self._win_fn

    @win_fn.setter
    def win_fn(self, val):
        self._win_fn = val

    @property
    def magsquared(self):
        return self._magsquared

    @magsquared.setter
    def magsquared(self, val):
        self._magsquared = val

    @property
    def force_rad2(self):
        return self._force_rad2

    @force_rad2.setter
    def force_rad2(self, val):
        self._force_rad2 = val

    def is_radix4(self):
        if self.force_rad2:
            return False
        return round(math.log(self.n_cfft, 4)) == math.log(self.n_cfft, 4) and self.n_cfft > 64

    def get_parameter_size(self):
        return self.n_fft

    def get_output_size(self, in_dims):
        out_dim = Dim([2, self.n_fft//2+1])
        return [out_dim]

    @property
    def can_equalize(self):
        return False

    def compute_load(self):
        return 0

    def __str__(self):
        return f"RFFT: NFFT={self.n_fft}, Win={self.win_fn}, Rad4={self.is_radix4()}"


@cls_op_name("MFCC")
class MFCCPreprocessingParameters(Parameters, SingleInputAndOutput, SensitiveToOrder):

    def __init__(self, name: str, conf_dict=None, n_fft=None, magsquared=True, fbank_type=None, n_fbanks=None, fmin=None, fmax=None, log=True, n_dct=None, frame_axis=0, window="hanning", real_samples=True, **kwargs):
        super(MFCCPreprocessingParameters, self).__init__(name, **kwargs)
        self._real_fft = conf_dict.get(
            "real_samples", real_samples) if conf_dict else real_samples
        self._n_fft = conf_dict.get("n_fft", n_fft) if conf_dict else n_fft
        assert self._n_fft
        self._magsquared = conf_dict.get("magsquared", magsquared) if conf_dict else magsquared
        self._fbank_type = conf_dict.get(
            "fbank_type", fbank_type) if conf_dict else fbank_type
        self._n_fbanks = conf_dict.get(
            "n_fbanks", n_fbanks) if conf_dict else n_fbanks
        self._fmin = conf_dict.get("fmin", fmin) if conf_dict else fmin
        self._fmax = conf_dict.get("fmax", fmax) if conf_dict else fmax
        self._log = conf_dict.get("log", log) if conf_dict else log
        self._n_dct = conf_dict.get("n_dct", n_dct) if conf_dict else n_dct
        self._win_fn = conf_dict.get("win_fn", window) if conf_dict else window
        self._frame_axis = frame_axis
        self._force_rad2 = False

    @property
    def frame_axis(self):
        return self._frame_axis

    @frame_axis.setter
    def frame_axis(self, val):
        self._frame_axis = val

    @property
    def real_fft(self):
        return self._real_fft

    @real_fft.setter
    def real_fft(self, val):
        self._real_fft = val

    @property
    def n_fft(self):
        return self._n_fft

    @n_fft.setter
    def n_fft(self, val):
        self._n_fft = val

    @property
    def win_fn(self):
        return self._win_fn

    @win_fn.setter
    def win_fn(self, val):
        self._win_fn = val

    @property
    def fmin(self):
        return self._fmin

    @fmin.setter
    def fmin(self, val):
        self._fmin = val

    @property
    def fmax(self):
        return self._fmax

    @fmax.setter
    def fmax(self, val):
        self._fmax = val

    @property
    def fbank_type(self):
        return self._fbank_type

    @fbank_type.setter
    def fbank_type(self, val):
        self._fbank_type = val

    @property
    def n_fbanks(self):
        return self._n_fbanks

    @n_fbanks.setter
    def n_fbanks(self, val):
        self._n_fbanks = val

    @property
    def log(self):
        return self._log

    @log.setter
    def log(self, val):
        self._log = val

    @property
    def n_dct(self):
        return self._n_dct

    @n_dct.setter
    def n_dct(self, val):
        self._n_dct = val

    @property
    def magsquared(self):
        return self._magsquared

    @magsquared.setter
    def magsquared(self, val):
        self._magsquared = val

    @property
    def force_rad2(self):
        return self._force_rad2

    @force_rad2.setter
    def force_rad2(self, val):
        self._force_rad2 = val

    def is_radix4(self):
        if self.force_rad2:
            return False
        return round(math.log(self.n_fft, 4)) == math.log(self.n_fft, 4)

    def get_parameter_size(self):
        return self.n_fft

    def get_output_size(self, in_dims):
        out_dim = in_dims[0].clone()
        return [out_dim]

    @property
    def can_equalize(self):
        return False

    def compute_load(self):
        return 0

    def __str__(self):
        return f"MFCC: NFFT={self.n_fft}, Win={self.win_fn}, Rad4={self.is_radix4()}, MagSquared={self.magsquared}, Log={self.log}, NDCT={self.n_dct}"
