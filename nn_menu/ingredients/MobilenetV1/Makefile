# Copyright (C) 2017 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

ifndef GAP_SDK_HOME
  $(error Source sourceme in gap_sdk first)
endif
MODEL_PREFIX = mobilenetv1
include common.mk

IMAGE=$(CURDIR)/images/goldfish.ppm

io=host

QUANT_BITS=8
BUILD_DIR=BUILD
MODEL_SQ8=1

$(info Building GAP8 mode with $(QUANT_BITS) bit quantization)

NNTOOL_SCRIPT=model/nntool_script
MODEL_SUFFIX = _SQ8BIT

#LOAD A TFLITE QUANTIZED GRAPH
NNTOOL_EXTRA_FLAGS= -q

include common/model_decl.mk
TRAINED_TFLITE_MODEL=model/mobilenet_v1_1.0_224_quant.tflite

# Here we set the memory allocation for the generated kernels
# REMEMBER THAT THE L1 MEMORY ALLOCATION MUST INCLUDE SPACE
# FOR ALLOCATED STACKS!
# Here we set the memory allocation for the generated kernels
# REMEMBER THAT THE L1 MEMORY ALLOCATION MUST INCLUDE SPACE
# FOR ALLOCATED STACKS!
# FC stack size:
MAIN_STACK_SIZE=1024
# Cluster PE0 stack size:
CLUSTER_STACK_SIZE=4096
# Cluster PE1-PE7 stack size:
CLUSTER_SLAVE_STACK_SIZE=1024
TOTAL_STACK_SIZE=$(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* 7)
ifeq '$(TARGET_CHIP_FAMILY)' 'GAP9'
  FREQ_CL?=50
  FREQ_FC?=50
  MODEL_L1_MEMORY=$(shell expr 125000 \- $(TOTAL_STACK_SIZE))
  MODEL_L2_MEMORY=1300000
  MODEL_L3_MEMORY=8388608
else
  FREQ_CL?=175
  FREQ_FC?=250
  MODEL_L1_MEMORY=$(shell expr 60000 \- $(TOTAL_STACK_SIZE))
  MODEL_L2_MEMORY=300000
  MODEL_L3_MEMORY=8388608
endif
# hram - HyperBus RAM
# qspiram - Quad SPI RAM
MODEL_L3_EXEC=hram
# hflash - HyperBus Flash
# qpsiflash - Quad SPI Flash
MODEL_L3_CONST=hflash

pulpChip = GAP
PULP_APP = mobilenetv1

APP = mobilenetv1
APP_SRCS += mobilenetv1.c $(MODEL_GEN_C) $(MODEL_COMMON_SRCS) $(CNN_LIB)

APP_CFLAGS += -O3 -mno-memcpy -fno-tree-loop-distribute-patterns 
APP_CFLAGS += -I. -I$(MODEL_COMMON_INC) -I$(TILER_EMU_INC) -I$(TILER_INC) $(CNN_LIB_INCLUDE) -I$(MODEL_BUILD)
APP_CFLAGS += -DPERF -DAT_MODEL_PREFIX=$(MODEL_PREFIX) $(MODEL_SIZE_CFLAGS)
APP_CFLAGS += -DSTACK_SIZE=$(CLUSTER_STACK_SIZE) -DSLAVE_STACK_SIZE=$(CLUSTER_SLAVE_STACK_SIZE)
APP_CFLAGS += -DAT_IMAGE=$(IMAGE)
APP_CFLAGS += -DFREQ_CL=$(FREQ_CL) -DFREQ_FC=$(FREQ_FC)

READFS_FILES=$(abspath $(MODEL_TENSORS))
PLPBRIDGE_FLAGS += -f

# all depends on the model
all:: model

clean:: clean_model

include common/model_rules.mk
$(info APP_SRCS... $(APP_SRCS))
$(info APP_CFLAGS... $(APP_CFLAGS))
include $(RULES_DIR)/pmsis_rules.mk

