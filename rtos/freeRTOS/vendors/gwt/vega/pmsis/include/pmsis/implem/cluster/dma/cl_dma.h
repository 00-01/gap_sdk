/*
 * Copyright (c) 2021, GreenWaves Technologies, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * o Redistributions of source code must retain the above copyright notice, this list
 *   of conditions and the following disclaimer.
 *
 * o Redistributions in binary form must reproduce the above copyright notice, this
 *   list of conditions and the following disclaimer in the documentation and/or
 *   other materials provided with the distribution.
 *
 * o Neither the name of GreenWaves Technologies, Inc. nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if (FEATURE_CLUSTER == 1)

#pragma once

#include "pmsis.h"
#include "pmsis/implem/hal/hal.h"

/*******************************************************************************
 * Definitions
 ******************************************************************************/

struct pi_cl_dma_cmd_s
{
    uint32_t tid;
    uint32_t cmd;
    uint32_t size;
    uint32_t stride;
    uint32_t length;
    uint32_t loc;
    uint32_t ext;
    struct pi_cl_dma_cmd_s *next;
};

/*******************************************************************************
 * Driver data
 *****************************************************************************/

extern pi_cl_dma_cmd_t *fifo_first;
extern pi_cl_dma_cmd_t *fifo_last;

/*******************************************************************************
 * Function declaration
 ******************************************************************************/

static inline void __pi_cl_dma_mutex_lock()
{
    hal_cl_eu_mutex_lock((uint32_t) CL_DMA_MUTEX_ID);
}

static inline void __pi_cl_dma_mutex_unlock()
{
    hal_cl_eu_mutex_unlock((uint32_t) CL_DMA_MUTEX_ID);
}

/*******************************************************************************
 * Internal functions
 ******************************************************************************/

static inline void __pi_cl_dma_1d_copy(uint8_t device_id,
                                       uint32_t ext, uint32_t loc, uint16_t len,
                                       uint8_t dir, uint8_t merge, pi_cl_dma_cmd_t *cmd)
{
    int32_t tid = -1;
    __pi_cl_dma_mutex_lock();
    if (!merge)
    {
        tid = hal_cl_dma_tid_get(device_id);
    }
    cmd->tid = tid;
    cmd->length = 0;            /* Used to check if 2D transfer. */
    uint32_t dma_cmd = hal_cl_dma_cmd_make(len, dir, CL_DMA_INC, CL_DMA_IS_1D_EXT,
                                           CL_DMA_ELE_ENA, CL_DMA_ILE_DIS, CL_DMA_BLE_ENA,
                                           CL_DMA_IS_1D_TCDM);
    hal_compiler_barrier();
    hal_cl_dma_1d_transfer_push(device_id, dma_cmd, loc, ext);
    __pi_cl_dma_mutex_unlock();
}

static inline void __pi_cl_dma_2d_copy(uint8_t device_id,
                                       uint32_t ext, uint32_t loc, uint32_t len,
                                       uint32_t stride, uint32_t length,
                                       uint8_t dir, uint8_t merge, pi_cl_dma_cmd_t *cmd)
{
    int32_t tid = -1;
    __pi_cl_dma_mutex_lock();
    if (!merge)
    {
        tid = hal_cl_dma_tid_get(device_id);
    }
    cmd->tid = tid;
    cmd->stride = stride;
    cmd->length = length;
    uint32_t tcdm_2d = (uint32_t) CL_DMA_IS_1D_TCDM;
    uint32_t dma_cmd = hal_cl_dma_cmd_make(len, dir, CL_DMA_INC, CL_DMA_IS_2D_EXT,
                                           CL_DMA_ELE_ENA, CL_DMA_ILE_DIS, CL_DMA_BLE_ENA,
                                           tcdm_2d);
    hal_compiler_barrier();
    hal_cl_dma_2d_transfer_push(device_id, dma_cmd, loc, ext, stride, length, tcdm_2d);
    __pi_cl_dma_mutex_unlock();
}

static inline void __pi_cl_dma_wait(uint8_t device_id, pi_cl_dma_cmd_t *cmd)
{
    __pi_cl_dma_mutex_lock();
    while (hal_cl_dma_status_get(device_id) & (1 << cmd->tid))
    {
        __pi_cl_dma_mutex_unlock();
        hal_cl_eu_event_mask_wait_and_clear(1 << CL_IRQ_DMA0);
        __pi_cl_dma_mutex_lock();
    }
    hal_cl_dma_tid_free(device_id, cmd->tid);
    __pi_cl_dma_mutex_unlock();
}

static inline void __pi_cl_dma_flush(uint8_t device_id)
{
    __pi_cl_dma_mutex_lock();
    while (hal_cl_dma_status_get(device_id) & 0xFFFF)
    {
        __pi_cl_dma_mutex_unlock();
        hal_cl_eu_event_mask_wait_and_clear(1 << CL_IRQ_DMA0);
        __pi_cl_dma_mutex_lock();
    }
    hal_cl_dma_status_set(device_id, -1); /* -1 = 0xFFFFFFFFu */
    __pi_cl_dma_mutex_unlock();
}

/*******************************************************************************
 * Function implementation
 ******************************************************************************/

static inline void pi_cl_dma_cmd(uint32_t ext, uint32_t loc, uint32_t size,
                                 pi_cl_dma_dir_e dir, pi_cl_dma_cmd_t *cmd)
{
    __pi_cl_dma_1d_copy(CL_DMA_ID(0), ext, loc, size, dir, 0, cmd);
}

static inline void pi_cl_dma_cmd_2d(uint32_t ext, uint32_t loc, uint32_t size,
                                    uint32_t stride, uint32_t length,
                                    pi_cl_dma_dir_e dir, pi_cl_dma_cmd_t *cmd)
{
    __pi_cl_dma_2d_copy(CL_DMA_ID(0), ext, loc, size, stride, length, dir, 0, cmd);
}

static inline void pi_cl_dma_cmd_wait(pi_cl_dma_cmd_t *cmd)
{
    __pi_cl_dma_wait(CL_DMA_ID(0), cmd);
}

static inline void pi_cl_dma_flush()
{
    __pi_cl_dma_flush(CL_DMA_ID(0));
}

static inline void pi_cl_dma_memcpy(pi_cl_dma_copy_t *copy)
{
    __pi_cl_dma_1d_copy(CL_DMA_ID(0),
                        copy->ext, copy->loc, copy->size,
                        copy->dir, copy->merge, (pi_cl_dma_cmd_t *) copy);
}

static inline void pi_cl_dma_memcpy_2d(pi_cl_dma_copy_2d_t *copy)
{
    __pi_cl_dma_2d_copy(CL_DMA_ID(0),
                        copy->ext, copy->loc, copy->size,
                        copy->stride, copy->length,
                        copy->dir, copy->merge, (pi_cl_dma_cmd_t *) copy);
}

static inline void pi_cl_dma_wait(void *copy)
{
    __pi_cl_dma_wait(CL_DMA_ID(0), (pi_cl_dma_cmd_t *) copy);
}

#endif  /* FEATURE_CLUSTER */
