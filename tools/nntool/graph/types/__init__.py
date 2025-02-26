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

from graph.types.activations import (ActivationParameters,
                                     HSigmoidActivationParameters,
                                     HSwishActivationParameters,
                                     HTanHActivationParameters,
                                     LeakyActivationParameters,
                                     ReluActivationParameters,
                                     SigmoidActivationParameters,
                                     SoftMaxParameters,
                                     TanHActivationParameters)
from graph.types.base import (EdgeParameters, FilterLikeParameters,
                              FilterParameters, InsensitiveToQuantization,
                              MultiplicativeBiasParameters, NNEdge,
                              NodeOptions, Parameters,
                              SameNumberOfDimensionsForInputs,
                              SensitiveToOrder, SingleInputAndOutput)
from graph.types.constant_input import ConstantInputParameters
from graph.types.conv2d import (BatchNormalizationParameters, Conv2DParameters,
                                TransposeConv2DParameters)
from graph.types.dsp_preprocessing import (DSPParameters,
                                           MFCCPreprocessingParameters,
                                           RFFT2DPreprocessingParameters)
from graph.types.expression_fusion import ExpressionFusionParameters
from graph.types.fusions import (ActivationFusion, ActivationFusionBase,
                                 BroadcastableActivationFusion,
                                 ConvFusionParameters, FilterFusionBase,
                                 FusionBase, FusionInputParameters,
                                 FusionOutputParameters,
                                 LinearFusionParameters,
                                 MatMulOpFusionParameters,
                                 MatScaleFusionParameters,
                                 PaddedAddFusionParameters)
from graph.types.global_pooling import (GlobalAveragePoolParameters,
                                        GlobalMaxPoolParameters,
                                        GlobalMinPoolParameters,
                                        GlobalPoolingParameters,
                                        GlobalSumPoolParameters)
from graph.types.image_formatter import ImageFormatParameters
from graph.types.input_output import (InputBaseParameters,
                                      InputOutputParameters, InputParameters,
                                      OutputParameters)
from graph.types.linear import FcParameters
from graph.types.lstm import LSTMParameters
from graph.types.others import (BinaryOpParameters, ConcatParameters,
                                CopyParameters, ExpandParameters,
                                ExpOpParameters, GatherParameters,
                                LogOpParameters, MaxOpParameters,
                                MinOpParameters, NegOpParameters,
                                NoOPParameters, PadParameters, PowOpParameters,
                                QuantizeParameters, ReshapeParameters,
                                ReverseParameters, SplitParameters,
                                SqrtOpParameters, StridedSliceParameters,
                                TransposeParameters, UnaryOpParameters,
                                UnconvertedOpParameters,
                                UnexecutableOpParameters, UnknownOpParameters)
from graph.types.pooling import (AveragePoolParameters, MaxPoolParameters,
                                 PoolingParameters)
from graph.types.resizers import (BilinearResizerParameters,
                                  NearestNeighborResizerParameters,
                                  ResizerParameters)
from graph.types.rnn import GRUParameters, RNNBaseParameters, RNNParameters
from graph.types.ssd import NMSParameters, SSDDetectorParameters
from graph.types.tensor_arithmetic import (Broadcastable, MatMulOpParameters,
                                           MatMulTransposedParameters,
                                           MatrixAddParameters,
                                           MatrixBroadcastedLinearOpParameters,
                                           MatrixDivParameters,
                                           MatrixMulParameters,
                                           MatrixSubParameters)
