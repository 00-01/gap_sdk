/*
 * Copyright (C) 2021 GreenWaves Technologies
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpointer-sign"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <stdio.h>
#include <math.h>
#include "RNN_BasicKernels_NE16.h"
#include "stage_desc.h"

static int CoreCountDynamic = 1;
static int ActiveCore = gap_ncore();
// #define DEBUG_NE16 0


void dump_u8(const char *name, int len, const unsigned char *out)
{
        pi_cl_team_critical_enter();
        printf("%s[%d] = { ", name, len);
        for (int i = 0; i < len; i++)
                printf("%x%s", out[i], (i == len - 1 ? "}\n" : ", "));
        pi_cl_team_critical_exit();
}

void dump_u16(const char *name, int len, const unsigned short *out)
{
        pi_cl_team_critical_enter();
        printf("%s[%d] = { ", name, len);
        for (int i = 0; i < len; i++)
                printf("%u%s", out[i], (i == len - 1 ? "}\n" : ", "));
        pi_cl_team_critical_exit();
}

void dump_16(const char *name, int len, const short *out)
{
        pi_cl_team_critical_enter();
        printf("%s[%d] = { ", name, len);
        for (int i = 0; i < len; i++)
                printf("%d%s", out[i], (i == len - 1 ? "}\n" : ", "));
        pi_cl_team_critical_exit();
}

void dump_i32(const char *name, int len, const int *out)
{
        pi_cl_team_critical_enter();
        printf("%s[%d] = { ", name, len);
        for (int i = 0; i < len; i++)
                printf("%d%s", out[i], (i == len - 1 ? "}\n" : ", "));
        pi_cl_team_critical_exit();
}

static inline unsigned int __attribute__((always_inline)) ChunkSize(unsigned int X, int IncludeCC)

{
        unsigned int NCore;
        unsigned int Log2Core;
        unsigned int Chunk;

        if (CoreCountDynamic)
                NCore = ActiveCore;
        else
                NCore = gap_ncore();
        if (IncludeCC)
        {
                NCore += 1;
                Chunk = X / NCore;
        }
        else
        {
                Log2Core = gap_fl1(NCore);
                Chunk = (X >> Log2Core) + ((X & (NCore - 1)) != 0);
        }
        return Chunk;
}


static inline void ParSetup(int Nout, int IncCC, int * First, int * Last)
{
        int CoreId = gap_coreid();
        int ChunkCell = ChunkSize(Nout, 1);
        *First = CoreId * ChunkCell;
        *Last = Nout;
}

#define B_CLR(x, bits) ((x) & (~((1 << (bits)) - 1)))

static inline void Copy(void *__restrict__ To, void *__restrict__ From, unsigned int Size, unsigned int CoreId)

{
        unsigned int Chunk = ChunkSize(Size, 0), First = Chunk * CoreId, Last = Min(First + Chunk, Size);
        unsigned int Iter = Max(0, Last - First);

        int *pFrom = (int *)(From + First), *pTo = (int *)(To + First);
        for (int i = 0; i < Iter / 8; i++)
        {
                int V0 = pFrom[2 * i], V1 = pFrom[2 * i + 1];
                pTo[2 * i] = V0;
                pTo[2 * i + 1] = V1;
        }
        if (Iter & 0x4)
                *((int *)(To + First + B_CLR(Iter, 3))) = *((int *)(From + First + B_CLR(Iter, 3)));
        if (Iter & 0x2)
                *((short int *)(To + First + B_CLR(Iter, 2))) = *((short int *)(From + First + B_CLR(Iter, 2)));
        if (Iter & 0x1)
                *((signed char *)(To + First + Iter - 1)) = *((signed char *)(From + First + Iter - 1));
}

static inline void ZeroBody(void *__restrict__ To, unsigned int Cnt)
{
        int *pTo = (int *)To;
        for (int i = 0; i < Cnt / 8; i++)
        {
                pTo[2 * i] = 0;
                pTo[2 * i + 1] = 0;
        }
        if (Cnt & 0x4)
                *((int *)(To + B_CLR(Cnt, 3))) = 0;
        if (Cnt & 0x2)
                *((short int *)(To + B_CLR(Cnt, 2))) = 0;
        if (Cnt & 0x1)
                *((signed char *)(To + Cnt - 1)) = 0;
}

static inline void Zero(void *__restrict__ To, unsigned int Size, unsigned int CoreId)
{
        int Trace = 0;
        unsigned int Chunk = ChunkSize(Size, 0), First = Chunk * CoreId, Last = Min(First + Chunk, Size);
        if (Trace)
                printf("zero %d %d %d\n", CoreId, First, Last);

        int Iter = Max(0, Last - First);
        ZeroBody(To + First, Iter);
}

static inline void ZeroState8Body(void *__restrict__ To, unsigned int Cnt)
{
        int *pTo = (int *)To;
        for (int i = 0; i < Cnt / 8; i++)
        {
                pTo[2 * i] = 0x80808080;
                pTo[2 * i + 1] = 0x80808080;
        }
        if (Cnt & 0x4)
                *((int *)(To + B_CLR(Cnt, 3))) = 0x80808080;
        if (Cnt & 0x2)
                *((short int *)(To + B_CLR(Cnt, 2))) = 0x8080;
        if (Cnt & 0x1)
                *((signed char *)(To + Cnt - 1)) = 0x80;
}

static inline void ZeroState8(unsigned char *__restrict__ To, unsigned int Size, unsigned int CoreId)

{
        int Trace = 0;
        unsigned int Chunk = ChunkSize(Size, 0), First = Chunk * CoreId, Last = Min(First + Chunk, Size);
        if (Trace)
                printf("zero %d %d %d\n", CoreId, First, Last);

        int Iter = Max(0, Last - First);
        ZeroState8Body(To + First, Iter);
}

static inline void ZeroState16Body(void *__restrict__ To, unsigned int Cnt, unsigned int ZeroPoint)
{
        unsigned int ZeroPointInt = ZeroPoint << 16 | ZeroPoint;
        unsigned int *pTo = (int *)To;
        for (int i = 0; i < Cnt / 4; i++)
        {
                pTo[2 * i] = ZeroPointInt;
                pTo[2 * i + 1] = ZeroPointInt;
        }
        if (Cnt & 0x2)
                *((unsigned int *)(To + B_CLR(Cnt, 2))) = ZeroPointInt;
        if (Cnt & 0x1)
                *((unsigned short int *)(To + Cnt - 1)) = ZeroPoint;
}

static inline void ZeroState16(short int *__restrict__ To, unsigned int Size, unsigned int CoreId, unsigned int ZeroPoint)

{
        int Trace = 1;
        unsigned int Chunk = ChunkSize(Size, 0), First = Chunk * CoreId, Last = Min(First + Chunk, Size);

        int Iter = Max(0, Last - First);

        if (Trace)
                printf("zero %d %d %d %x\n", CoreId, First, Last, ZeroPoint);
        ZeroState16Body(To + First, Iter, ZeroPoint);

}

static inline void SetNE16_InPointer(void *InPointer)
{
        NE16_WRITE_REG(NE16_REG_INFEAT_PTR, (int)InPointer);
#if defined(DEBUG_NE16) || defined(DEBUG_NE16_PNTR)
        printf("InPointer:\t%x\n", InPointer);
#endif
}
static inline void SetNE16_OutPointer(void *OutPointer)
{
        NE16_WRITE_REG(NE16_REG_OUTFEAT_PTR, (int)OutPointer);
#if defined(DEBUG_NE16) || defined(DEBUG_NE16_PNTR)
        printf("OutPointer:\t%x\n", OutPointer);
#endif
}
static inline void SetNE16_WeightsPointer(void *WeightsPointer)
{
        NE16_WRITE_REG(NE16_REG_WEIGHTS_PTR, (int)WeightsPointer);
#ifdef DEBUG_NE16
        printf("WeightsPointer:\t%x\n", WeightsPointer);
#endif
}
static inline void SetNE16_BiasPointer(void *BiasPointer)
{
        NE16_WRITE_REG(NE16_REG_SCALE_BIAS_PTR, (int)BiasPointer);
#ifdef DEBUG_NE16
        printf("BiasPointer:\t%x\n", BiasPointer);
#endif
}
static inline void SetNE16_ScalePointer(void *ScalePointer)
{
        NE16_WRITE_REG(NE16_REG_SCALE_PTR, (int)ScalePointer);
#ifdef DEBUG_NE16
        printf("ScalePointer:\t%x\n", ScalePointer);
#endif
}
static inline void SetNE16_ScaleNPointer(void *ScaleNPointer)
{
        NE16_WRITE_REG(NE16_REG_SCALE_SHIFT_PTR, (int)ScaleNPointer);
#ifdef DEBUG_NE16
        printf("ScaleNPointer:\t%x\n", ScaleNPointer);
#endif
}

static inline void SetNE16_Strides(unsigned short In_D0,
                                   unsigned short In_D1,
                                   unsigned short In_D2,

                                   unsigned short Out_D0,
                                   unsigned short Out_D1,
                                   unsigned short Out_D2,

                                   unsigned short Weights_D0,
                                   unsigned short Weights_D1,
                                   unsigned short Weights_D2)
{
#ifdef DEBUG_NE16
        printf("InStrides: %d - %d - %d OutStrides: %d - %d - %d WeightsStrides: %d - %d - %d\n", In_D0, In_D1, In_D2, Out_D0, Out_D1, Out_D2, Weights_D0, Weights_D1, Weights_D2);
#endif
        NE16_WRITE_REG(NE16_REG_INFEAT_D0_STRIDE, In_D0);
        NE16_WRITE_REG(NE16_REG_INFEAT_D1_STRIDE, In_D1);
        NE16_WRITE_REG(NE16_REG_INFEAT_D2_STRIDE, In_D2);
        NE16_WRITE_REG(NE16_REG_OUTFEAT_D0_STRIDE, Out_D0);
        NE16_WRITE_REG(NE16_REG_OUTFEAT_D1_STRIDE, Out_D1);
        NE16_WRITE_REG(NE16_REG_OUTFEAT_D2_STRIDE, Out_D2);
        NE16_WRITE_REG(NE16_REG_WEIGHTS_D0_STRIDE, Weights_D0);
        NE16_WRITE_REG(NE16_REG_WEIGHTS_D1_STRIDE, Weights_D1);
        NE16_WRITE_REG(NE16_REG_WEIGHTS_D2_STRIDE, Weights_D2);
}

static inline void SetNE16_Dim(unsigned short Nb_KI,
                               unsigned short Nb_KO,
                               unsigned short Nb_WO,
                               unsigned short Nb_HO)
{
#ifdef DEBUG_NE16
        printf("Nb_KI:\t%d\tNb_KO:\t%d\tNb_HO:\t%d\tNb_WO:\t%d\n", Nb_KI, Nb_KO, Nb_WO, Nb_HO);
#endif
        NE16_WRITE_REG(NE16_REG_NB_KO_KI, ((Nb_KI & NE16_MASK_NB_KI) << NE16_SHIFT_NB_KI) |     /**< The Ki remainder must be the remainder of Ki / 16 if such remainder is not zero, otherwise must be 16 */
                                              ((Nb_KO & NE16_MASK_NB_KO) << NE16_SHIFT_NB_KO)); /**< The number of Ko subtiles is Ko / 32, plus 1 if Ko / 32 has non-zero remainder */
        NE16_WRITE_REG(NE16_REG_NB_HO_WO, ((Nb_WO & NE16_MASK_NB_WO) << NE16_SHIFT_NB_WO) |     /**< The number of Wo subtitles is Wo / 3, plus 1 if Wo / 3 has non-zero remainder */
                                              ((Nb_HO & NE16_MASK_NB_HO) << NE16_SHIFT_NB_HO)); /**< The number of Ho subtitles is Ho / 3, plus 1 if Ho / 3 has non-zero remainder */
}

static inline void SetNE16_Reminders(unsigned short Rem_WI,
                                     unsigned short Rem_HI,
                                     unsigned short Rem_KI,
                                     unsigned short Rem_KO,
                                     unsigned short Rem_WO,
                                     unsigned short Rem_HO)
{
#ifdef DEBUG_NE16
        printf("Rem_KI:\t%d\tRem_KO:\t%d\tRem_HI:\t%d\tRem_WI:\t%d\tRem_HO:\t%d\tRem_WO:\t%d\n", Rem_KI, Rem_KO, Rem_HI, Rem_WI, Rem_HO, Rem_WO);
#endif
        NE16_WRITE_REG(NE16_REG_REM_KO_KI, ((Rem_KI & NE16_MASK_REM_KI) << NE16_SHIFT_REM_KI) |     /**< The Ki remainder must be the remainder of Ki / 16 if such remainder is not zero, otherwise must be 16 */
                                               ((Rem_KO & NE16_MASK_REM_KO) << NE16_SHIFT_REM_KO)); /**< The Ko remainder must be the remainder of Ko / 32 if such remainder is not zero, otherwise must be 32 */
        NE16_WRITE_REG(NE16_REG_REM_HO_WO, ((Rem_WO & NE16_MASK_REM_WO) << NE16_SHIFT_REM_WO) |     /**< The Wo remainder must be the remainder of Wo / 3 */
                                               ((Rem_HO & NE16_MASK_REM_HO) << NE16_SHIFT_REM_HO)); /**< The Ho remainder must be the remainder of Ho / 3 */
        NE16_WRITE_REG(NE16_REG_REM_HI_WI, ((Rem_WI & NE16_MASK_REM_WI) << NE16_SHIFT_REM_WI) |     /**< */
                                               ((Rem_HI & NE16_MASK_REM_HI) << NE16_SHIFT_REM_HI)); /**< */
}

static inline void SetNE16_ConfigPad(v4s Pad, short int PadVal)
{
        NE16_WRITE_REG(NE16_REG_PADDING, ((PadVal & NE16_MASK_PADDING_VALUE) << NE16_SHIFT_PADDING_VALUE) |     /**< Only if the NE16 is set to 3x3 mode. Explicit padding forces the value of part of the input set.  */
                                             ((Pad[0] & NE16_MASK_PADDING_LEFT) << NE16_SHIFT_PADDING_LEFT) |   /**< It can be from 0 to 2 in each direction (left/right/top/bottom) */
                                             ((Pad[1] & NE16_MASK_PADDING_RIGHT) << NE16_SHIFT_PADDING_RIGHT) | /**< The padding value can be from 0 to 255 in basic mode and from 0 to 65535 in mode16 */
                                             ((Pad[2] & NE16_MASK_PADDING_TOP) << NE16_SHIFT_PADDING_TOP) |
                                             ((Pad[3] & NE16_MASK_PADDING_BOTTOM) << NE16_SHIFT_PADDING_BOTTOM));
#ifdef DEBUG_NE16
        printf("Padding: {%d, %d, %d, %d}, Val: %d\n", Pad[0], Pad[1], Pad[2], Pad[3], PadVal);
#endif
}

static inline void SetNE16_ConfigFMask(v4s FilterMask)
{
        NE16_WRITE_REG(NE16_REG_FILTER_MASK, ((FilterMask[0] & NE16_MASK_FILTER_MASK_LEFT) << NE16_SHIFT_FILTER_MASK_LEFT) |       /**< Only if the NE16 is set to 3x3 mode. */
                                                 ((FilterMask[1] & NE16_MASK_FILTER_MASK_RIGHT) << NE16_SHIFT_FILTER_MASK_RIGHT) | /**< Filter masking forces the value of part of the weights in the spatial direction */
                                                 ((FilterMask[2] & NE16_MASK_FILTER_MASK_TOP) << NE16_SHIFT_FILTER_MASK_TOP) |     /**< It can be from 0 to 1 in each direction (left/right/top/bottom) */
                                                 ((FilterMask[3] & NE16_MASK_FILTER_MASK_BOTTOM) << NE16_SHIFT_FILTER_MASK_BOTTOM));
#ifdef DEBUG_NE16
        printf("FMask: {%d, %d, %d, %d}\n", FilterMask[0], FilterMask[1], FilterMask[2], FilterMask[3]);
#endif
}

static inline void SetNE16_WOffset(int W_Offset)
{
        NE16_WRITE_REG(NE16_REG_WEIGHT_OFFSET, W_Offset);
#ifdef DEBUG_NE16
        printf("W_Offset: %d\n", W_Offset);
#endif
}

static inline void PrintNE16_GenConfig(unsigned int Cfg)
{
#ifdef DEBUG_NE16
        int Qw = ((Cfg >> NE16_SHIFT_WBITS_M1) & NE16_MASK_WBITS_M1) + 1;
        int Mode16 = ((Cfg >> NE16_SHIFT_MODE16) & NE16_MASK_MODE16);
        int StreamoutMode = ((Cfg >> NE16_SHIFT_OUTQUANT) & NE16_MASK_OUTQUANT);
        int FilterMode = ((Cfg >> NE16_SHIFT_FILTER_MODE) & NE16_MASK_FILTER_MODE);
        int LinearMode = ((Cfg >> NE16_SHIFT_LINEAR_MODE) & NE16_MASK_LINEAR_MODE);
        int StridedMode = ((Cfg >> NE16_SHIFT_STRIDED_MODE) & NE16_MASK_STRIDED_MODE);
        int NormBits = ((Cfg >> NE16_SHIFT_NORM_BITS) & NE16_MASK_NORM_BITS);
        int Streamin = ((Cfg >> NE16_SHIFT_STREAMIN) & NE16_MASK_STREAMIN);
        int WOffsetCfg = ((Cfg >> NE16_SHIFT_WEIGHT_OFFSET_CFG) & NE16_MASK_WEIGHT_OFFSET_CFG);
        int QuantRightShift = ((Cfg >> NE16_SHIFT_QUANT_RIGHT_SHIFT) & NE16_MASK_QUANT_RIGHT_SHIFT);
        int QuantBits = ((Cfg >> NE16_SHIFT_QUANT_BITS) & NE16_MASK_QUANT_BITS);
        int QuantNoRect = ((Cfg >> NE16_SHIFT_QUANT_NORECT) & NE16_MASK_QUANT_NORECT);
        int NormShift = ((Cfg >> NE16_SHIFT_NORM_SHIFT) & NE16_MASK_NORM_SHIFT);
        int NormBias = ((Cfg >> NE16_SHIFT_NORM_BIAS) & NE16_MASK_NORM_BIAS);
        printf("General config: %d\n\tQw: %d Mode16: %d StreamoutMode: %d FilterMode: %d LinearMode: %d\n\tStridedMode: %d NormBits: %d Streamin: %d WOffsetCfg: %d\n\tQuantRightShift: %d QuantBits: %d QuantNoRect: %d NormShift: %d\n\tNormBias %d\n", Cfg, Qw, Mode16, StreamoutMode, FilterMode, LinearMode, StridedMode, NormBits, Streamin, WOffsetCfg, QuantRightShift, QuantBits, QuantNoRect, NormShift, NormBias);
#endif
}

static inline void SetNE16_GenConfig(unsigned int Cfg)
{
        NE16_WRITE_REG(NE16_REG_CONFIG, 0);
        NE16_WRITE_REG(NE16_REG_CONFIG, Cfg);
#ifdef DEBUG_NE16
        PrintNE16_GenConfig(Cfg);
#endif
}


static inline void CalcNumIn(int cfg, int Nin, int *Rem_KI, int *Nb_KI)
{
        // int KiTileBig = Nin / 256;
        // int KiTileBigRem = KiTileBig % 256;
        // int KiTileSmall = (KiTileBigRem?KiTileBigRem/16 : 0);
        // if (KiTileSmall % 16) printf("WARNING!!! - Nin must be divisible by 16");
        // *Rem_KI = KiTileSmall;
        // *Nb_KI = KiTileBig + (KiTileSmall ? 1 : 0);
        int divisor = ((cfg >> NE16_SHIFT_MODE16) & NE16_MASK_MODE16 ? 512 : 256);
        *Rem_KI = ((Nin % divisor) / 16) == 0 ? 16 : (Nin % divisor) / 16;
        *Nb_KI = Nin / divisor + (Nin % divisor ? 1 : 0);
}

static inline int GetJobId()
{
        volatile int job_id;
        // printf("cfg: %x\n", cfg);
        // acquire job
        NE16_BARRIER_ACQUIRE(job_id);
        return job_id;
}

static inline void SetupNE16Job(int Cfg, void *pIn, void *pOut, void *pWeights, void *pBias, int OutBytes, int NumIn, int NumOut, int NumTileOut, int RemOut, int NumWOut, int NumHOut, int Qw, void *pScale, void *pScaleN, int WOff)
{

        int Rem_KI = 0, Nb_KI = 0;
        CalcNumIn(Cfg, NumIn, &Rem_KI, &Nb_KI);
        // printf("Nin %d, Rem_KI %d, Nb_KI %d\n", DimIn, Rem_KI, Nb_KI);

        if (pBias)
        {
                Cfg |= (NE16_MASK_NORM_BIAS << NE16_SHIFT_NORM_BIAS);
        }
        else
        {
                Cfg &= ~(NE16_MASK_NORM_BIAS << NE16_SHIFT_NORM_BIAS);
        }

        // load configuration for the layer - input only
        SetNE16_InPointer(pIn);
        SetNE16_OutPointer(pOut);
        SetNE16_WeightsPointer(pWeights);
        SetNE16_BiasPointer(pBias);
        SetNE16_ScalePointer(pScale);
        SetNE16_ScaleNPointer(pScaleN);
        SetNE16_Strides(16, 0, 0,                                           // In_D0, In_D1 - unused, In_D2 - unused
                        OutBytes * 8, OutBytes * NumOut, OutBytes * NumOut, // Out_D0, Out_D1 - unused, Out_D2 - unused
                        // OutBytes * 8, 0, 0,                                   // Out_D0, Out_D1 - unused, Out_D2 - unused
                        NumIn * 2 / 16, Qw * NumIn * 2 / 16, Qw * NumIn * 2); // Weights_D0, Weights_D1, Weights_D2
        SetNE16_Reminders(0, 0, Rem_KI, RemOut, 1, 1);
        SetNE16_Dim(Nb_KI, NumTileOut, 1, 1);
        SetNE16_WOffset(WOff);
        SetNE16_ConfigPad((v4s){0, 0, 0, 0}, 0);
        SetNE16_ConfigFMask((v4s){0, 0, 0, 0});
        SetNE16_GenConfig(Cfg);
}

static inline void TriggerNE16Job(int Cfg, void *pIn, void *pOut, void *pWeights, void *pBias, int OutBytes, int NumIn, int NumOut, int NumTileOut, int RemOut, int NumWOut, int NumHOut, int Qw, void *pScale, void *pScaleN, int WOff)
{
        GetJobId();
        SetupNE16Job(Cfg, pIn, pOut, pWeights, pBias, OutBytes, NumIn, NumOut, NumTileOut, RemOut, NumWOut, NumHOut, Qw, pScale, pScaleN, WOff);
        // commit and trigger NE16 computation
        NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);
}

void RNN_ParKerB32_Hard_NE16(KerRNN_NE16_T *Arg)

{
        /*	Sequences
                In:	DimIn!=0, Hout==0
                InOut:	DimIn!=0, Hout!=0
                None:	DimIn==0, Hout==0
                Out:	DimIn==0, Hout!=0

                Infos: Forget: HSigmoid:       Scale, ScaleN, A0, B0, C0, ActScale, ActScaleN          7

                if (PerChannelQuant) Infos group for each output elemt (Nout) else one group for all out
        */
        int Trace = 1;
        if (Trace)
                NE16_WRITE_REG(NE16_SPECIAL_TRACE_REG, 3);
        unsigned char *__restrict__ StateInOut = Arg->StateInOut;
        unsigned char *__restrict__ Xin = Arg->Xin;
        unsigned char *__restrict__ State = Arg->State;
        unsigned short int DimState = Arg->DimState;
        unsigned short int DimIn = Arg->DimIn;
        unsigned short int DimStateInt = Arg->DimStateInt;
        unsigned short int DimInInt = Arg->DimInInt;
        unsigned char *__restrict__ Hout = Arg->Hout;
        unsigned short int Nout = Arg->Nout;
        signed char *__restrict__ Infos = Arg->Infos;
        int *volatile __restrict__ OutBuff = (int *)Arg->OutBuff;
        int TileOff = Arg->TileOffset;

        unsigned char OutZeroPoint = *((unsigned char *)&Infos[RNN_NE16_OUT_ZERO_POINT]);

        unsigned int CoreId = gap_coreid();
        unsigned int ChunkCell = ChunkSize(Nout, 1);
        unsigned int First = CoreId * ChunkCell;
        unsigned int Last = Min(First + ChunkCell, Nout);

        if (CoreId != 8)
        {
                if (Arg->FirstOut)
                {
                        if (Arg->FirstCell && Arg->Reset)
                        {
                                if (Trace)
                                        printf("%d zero state\n", CoreId);
                                ZeroState8(State, DimState, CoreId);
                        }
                        else
                        {
                                if (Trace)
                                        printf("%d copy state\n", CoreId);
                                Copy(State, StateInOut, DimState, CoreId);
                        }
                        gap_waitbarrier(0);
                }
                if (Xin)
                {
                        Copy(State + DimStateInt, Xin, DimIn, CoreId);
                        gap_waitbarrier(0);
                }
        }
        else
        {
                if (Trace)
                        printf("NE16: FirstCell %d FirstOut %d Reset %d DimState %d DimIn %d DimStateInt %d DimInInt %d Nout %d\n",
                               Arg->FirstCell, Arg->FirstOut, Arg->Reset, DimState, DimIn, DimStateInt, DimInInt, Nout);
                Last = Nout;
                // Execute NE16 job
                unsigned char *__restrict__ Scale = Arg->ScaleNorm;
                unsigned char *__restrict__ ScaleN = &Arg->ScaleNorm[2 * Nout];
                unsigned char *__restrict__ Wf = Arg->Wf;
                unsigned char *__restrict__ Wfi = Arg->Wfi;
                char FilterDataSizeBits = Arg->FilterDataSizeBits;
                int *__restrict__ Bf = Arg->Bf;
                int *__restrict__ Bfi = &Arg->Bf[Nout];
                printf("Input weights/bias=%x/%x State weights/bias=%x/%x\n", Wfi, Bfi, Wf, Bf);

                int Nb_KI, Rem_KI;
                int Nb_KO = Nout / 32 + (Nout % 32 ? 1 : 0);
                int Rem_KO = Nout % 32 ? Nout % 32 : 32; // Check different wrt simulator

                unsigned int cfg = Arg->Default_NE16_Job_Cfg;

                int QuantBitsFlag = (cfg >> NE16_SHIFT_QUANT_BITS) & NE16_MASK_QUANT_BITS;

                if (Arg->FirstOut)
                {
                        if (Arg->FirstCell)
                        {
                                if (Trace)
                                        printf("%d zero state pad %d %d\n", CoreId, DimState, DimStateInt);
                                ZeroBody(&State[DimState], DimStateInt-DimState);
                        }
                        gap_waitbarrier(0);
                }
                if (Xin)
                {
                        if (Trace)
                                printf("%d zero Xin pad %d %d\n", CoreId, DimIn, DimInInt);
                        ZeroBody(&State[DimStateInt + DimIn], DimInInt-DimIn);
                        gap_waitbarrier(0);
                }

                // bias always on when in 8 bit mode set by generator

                NE16_SETPRIORITY_NE16(); // priority to NE16 w.r.t. cores, DMA
                if (Xin)
                {
                        if (Trace)
                        {
                                dump_u8("inp_scale", Nout, &Scale[Nout]);
                                dump_u8("inp_scale_n", Nout, &ScaleN[Nout]);
                        }
                        // switch off streamin
                        cfg &= ~(NE16_MASK_STREAMIN << NE16_SHIFT_STREAMIN);
                        TriggerNE16Job(cfg, &State[DimStateInt], OutBuff, Wfi, Bfi, 4, DimInInt, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, &Scale[Nout], &ScaleN[Nout], Infos[RNN_NE16_W_ZERO_POINT]);
                }

                if (Trace)
                {
                        dump_u8("state_scale", Nout, Scale);
                        dump_u8("state_scale_n", Nout, ScaleN);
                }
                // switch on streamin
                cfg |= (NE16_MASK_STREAMIN << NE16_SHIFT_STREAMIN);
                TriggerNE16Job(cfg, State, OutBuff, Wf, Bf, 4, DimStateInt, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, ScaleN, Infos[RNN_NE16_W_ZERO_POINT]);

                // wait for end of computation
                NE16_BARRIER();

                // set priority to core side
                NE16_SETPRIORITY_CORE();
        }
        gap_waitbarrier(0);
        for (int o = First; o < Last; o++)
        {
                // Already in output scale - just max min tahn
                unsigned char Of = ((unsigned char)Max(Infos[RNN_NE16_F_A0], Min(Infos[RNN_NE16_F_B0], OutBuff[o]))) + OutZeroPoint;

                if (StateInOut)
                        StateInOut[TileOff + o] = Of;
                if (Hout)
                        Hout[o] = Of;
        }
        if (Trace)
        {
                gap_waitbarrier(0);
                if (CoreId == 8)
                {
                        if (StateInOut)
                                dump_u8("state_out", Nout, &StateInOut[TileOff]);
                        if (Hout)
                                dump_u8("hout_out", Nout, Hout);
                }
        }
        gap_waitbarrier(0);
}

void RNN_ParKerB32_NE16(KerRNN_NE16_T *Arg)

{
        /*	Sequences
                In:	DimIn!=0, Hout==0
                InOut:	DimIn!=0, Hout!=0
                None:	DimIn==0, Hout==0
                Out:	DimIn==0, Hout!=0

                Infos: Forget: HSigmoid:       Scale, ScaleN, A0, B0, C0, ActScale, ActScaleN          7

                if (PerChannelQuant) Infos group for each output elemt (Nout) else one group for all out
        */
        int Trace = 1;
        if (Trace)
                NE16_WRITE_REG(NE16_SPECIAL_TRACE_REG, 3);
        unsigned char *__restrict__ StateInOut = Arg->StateInOut;
        unsigned char *__restrict__ Xin = Arg->Xin;
        unsigned char *__restrict__ State = Arg->State;
        unsigned short int DimState = Arg->DimState;
        unsigned short int DimIn = Arg->DimIn;
        unsigned short int DimStateInt = Arg->DimStateInt;
        unsigned short int DimInInt = Arg->DimInInt;
        unsigned char *__restrict__ Hout = Arg->Hout;
        unsigned short int Nout = Arg->Nout;
        signed char *__restrict__ Infos = Arg->Infos;
        int *volatile __restrict__ OutBuff = (int *)Arg->OutBuff;
        int TileOff = Arg->TileOffset;

        unsigned int CoreId = gap_coreid();
        unsigned int ChunkCell = ChunkSize(Nout, 1);
        unsigned int First = CoreId * ChunkCell;
        unsigned int Last = Min(First + ChunkCell, Nout);

        unsigned char OutZeroPoint = *((unsigned char *)&Infos[RNN_NE16_OUT_ZERO_POINT]);

        if (CoreId != 8)
        {
                if (Arg->FirstOut)
                {
                        if (Arg->FirstCell && Arg->Reset)
                        {
                                if (Trace)
                                        printf("%d zero state\n", CoreId);
                                ZeroState8(State, DimState, CoreId);
                        }
                        else
                        {
                                if (Trace)
                                        printf("%d copy state\n", CoreId);
                                Copy(State, StateInOut, DimState, CoreId);
                        }
                        gap_waitbarrier(0);
                }
                if (Xin)
                {
                        Copy(State + DimStateInt, Xin, DimIn, CoreId);
                        gap_waitbarrier(0);
                }
        }
        else
        {
                if (Trace)
                        printf("NE16: FirstCell %d FirstOut %d Reset %d DimState %d DimIn %d DimStateInt %d DimInInt %d Nout %d\n",
                               Arg->FirstCell, Arg->FirstOut, Arg->Reset, DimState, DimIn, DimStateInt, DimInInt, Nout);
                Last = Nout;
                // Execute NE16 job
                unsigned char *__restrict__ Scale = Arg->ScaleNorm;
                unsigned char *__restrict__ ScaleN = &Arg->ScaleNorm[2 * Nout];
                unsigned char *__restrict__ Wf = Arg->Wf;
                unsigned char *__restrict__ Wfi = Arg->Wfi;
                char FilterDataSizeBits = Arg->FilterDataSizeBits;
                int *__restrict__ Bf = Arg->Bf;
                int *__restrict__ Bfi = &Arg->Bf[Nout];
                printf("Input weights/bias=%x/%x State weights/bias=%x/%x\n", Wfi, Bfi, Wf, Bf);

                int Nb_KI, Rem_KI;
                int Nb_KO = Nout / 32 + (Nout % 32 ? 1 : 0);
                int Rem_KO = Nout % 32 ? Nout % 32 : 32; // Check different wrt simulator

                unsigned int cfg = Arg->Default_NE16_Job_Cfg;

                int QuantBitsFlag = (cfg >> NE16_SHIFT_QUANT_BITS) & NE16_MASK_QUANT_BITS;

                if (Arg->FirstOut)
                {
                        if (Arg->FirstCell)
                        {
                                if (Trace)
                                        printf("%d zero state pad %d %d\n", CoreId, DimState, DimStateInt);
                                ZeroBody(&State[DimState], DimStateInt-DimState);
                        }
                        gap_waitbarrier(0);
                }
                if (Xin)
                {
                        if (Trace)
                                printf("%d zero Xin pad %d %d\n", CoreId, DimIn, DimInInt);
                        ZeroBody(&State[DimStateInt + DimIn], DimInInt-DimIn);
                        gap_waitbarrier(0);
                }

                // bias always on when in 8 bit mode set by generator

                NE16_SETPRIORITY_NE16(); // priority to NE16 w.r.t. cores, DMA
                if (Xin)
                {
                        if (Trace)
                        {
                                dump_u8("inp_scale", Nout, &Scale[Nout]);
                                dump_u8("inp_scale_n", Nout, &ScaleN[Nout]);
                        }
                        // switch off streamin
                        cfg &= ~(NE16_MASK_STREAMIN << NE16_SHIFT_STREAMIN);
                        TriggerNE16Job(cfg, &State[DimStateInt], OutBuff, Wfi, Bfi, 4, DimInInt, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, &Scale[Nout], &ScaleN[Nout], Infos[RNN_NE16_W_ZERO_POINT]);
                }

                if (Trace)
                {
                        dump_u8("state_scale", Nout, Scale);
                        dump_u8("state_scale_n", Nout, ScaleN);
                }
                // switch on streamin
                cfg |= (NE16_MASK_STREAMIN << NE16_SHIFT_STREAMIN);
                TriggerNE16Job(cfg, State, OutBuff, Wf, Bf, 4, DimStateInt, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, ScaleN, Infos[RNN_NE16_W_ZERO_POINT]);

                // wait for end of computation
                NE16_BARRIER();

                // set priority to core side
                NE16_SETPRIORITY_CORE();
        }
        gap_waitbarrier(0);
        for (int o = First; o < Last; o++)
        {
                /* Scale to Output scale*/
                unsigned char Of = ((unsigned char)gap_clip(AT_SCALE(Tanh(OutBuff[o]), ((unsigned char *)Infos)[RNN_NE16_OUT_SCALE], ((unsigned char *)Infos)[RNN_NE16_OUT_SCALEN]), 7)) + OutZeroPoint;

                if (StateInOut)
                        StateInOut[TileOff + o] = Of;
                if (Hout)
                        Hout[o] = Of;
        }
        if (Trace)
        {
                gap_waitbarrier(0);
                if (CoreId == 8)
                {
                        if (StateInOut)
                                dump_u8("state_out", Nout, &StateInOut[TileOff]);
                        if (Hout)
                                dump_u8("hout_out", Nout, Hout);
                }
        }
        gap_waitbarrier(0);
}

inline static void NE16_Manual_SubTile_Linear_Start(unsigned int Cfg, void *InPointer, int NumIn, int Nb_KI, int WOff, int Qw)
{
        SetNE16_InPointer(InPointer);
        SetNE16_Dim(Nb_KI, 1, 1, 1);
        SetNE16_ConfigPad((v4s){0, 0, 0, 0}, 0);
        SetNE16_ConfigFMask((v4s){0, 0, 0, 0});
        SetNE16_GenConfig(Cfg);
        SetNE16_WOffset(WOff);
        SetNE16_Strides(16, 0, 0,    // In_D0, In_D1 - unused, In_D2 - unused
                        4 * 8, 0, 0, // Out_D0, Out_D1 - unused, Out_D2 - unused
                        // OutBytes * 8, 0, 0,                              // Out_D0, Out_D1 - unused, Out_D2 - unused
                        NumIn * 2 / 16, Qw * NumIn * 2 / 16,
                        Qw * NumIn * 2); // Weights_D0, Weights_D1, Weights_D2
}

inline static void NE16_Manual_SubTile_Linear_iter(void *OutPointer, void *WeightsPointer, int Rem_KI, int Rem_KO)
{
        SetNE16_OutPointer(OutPointer);
        SetNE16_WeightsPointer(WeightsPointer);
        SetNE16_Reminders(0, 0, Rem_KI, Rem_KO, 0, 0);
}

inline static void NE16_Manual_SubTile_Linear_Setup(
    unsigned int Cfg,
    int NumIn,
    int Qw,
    int *Nb_KI,
    int *Rem_KI,
    int *FiltSize)
{
        CalcNumIn(Cfg, NumIn, Rem_KI, Nb_KI);
        *FiltSize = Qw * (NumIn >> 3);
}

inline static void NE16_Manual_SubTile_Linear_Body(
    unsigned int Cfg,
    void *InPointer,
    int *OutPointer,
    unsigned char *WeightsPointer,
    int NumIn,
    int Nb_KI,
    int Rem_KI,
    int Nb_KO,
    int Rem_KO,
    int WOff,
    int Qw,
    int FiltSize,
    int IsInput,
    int JobId,
    pStageDesc_t pStageDesc)
{
        int Trace = 0;
        for (int Ko = 0; Ko < Nb_KO; Ko++)
        {
                int Off = Ko * 32;
                int IsLastKo = (Ko == (Nb_KO - 1));
                int NumOutC = (IsLastKo ? Rem_KO : 32);
                if (JobId == -1)
                        JobId = GetJobId();
                if (pStageDesc)
                        StageDescIterJobAquired(pStageDesc);
                if (Trace)
                {
                        printf(
                                "  State START_JOB\n    input=%d\n    k_out_major=%d\n    off=%d\n    is_last_ko=%d\n    job_id=%d\n",
                                IsInput, Ko, Off, IsLastKo, JobId);
                }
                JobId = -1;
                // Initialize both shadow register sets
                if (Ko < 2)
                {
                        NE16_Manual_SubTile_Linear_Start(Cfg, InPointer, NumIn, Nb_KI, WOff, Qw);
                }
                NE16_Manual_SubTile_Linear_iter(&OutPointer[Off], &WeightsPointer[Off * FiltSize], Rem_KI, NumOutC);
                NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);
        }
}

inline static void NE16_Manual_SubTile_Linear(
    unsigned int Cfg,
    void *InPointer,
    int *OutPointer,
    unsigned char *WeightsPointer,
    int NumIn,
    int Nb_KO,
    int Rem_KO,
    int WOff,
    int Qw,
    int IsInput)
{
        int Nb_KI;
        int Rem_KI;
        int FiltSize;
        NE16_Manual_SubTile_Linear_Setup(Cfg, NumIn, Qw, &Nb_KI, &Rem_KI, &FiltSize);
        NE16_Manual_SubTile_Linear_Body(Cfg, InPointer, OutPointer, WeightsPointer, NumIn, Nb_KI, Rem_KI,
                                 Nb_KO, Rem_KO, WOff, Qw, FiltSize, IsInput, -1, 0);
}

void RNN_ParKerB32_NE16fp(KerRNN_NE16fp_T *Arg)
{
        /*	Sequences
                In:	DimIn!=0, Hout==0
                InOut:	DimIn!=0, Hout!=0
                None:	DimIn==0, Hout==0
                Out:	DimIn==0, Hout!=0

                Infos: Forget: HSigmoid:       Scale, ScaleN, A0, B0, C0, ActScale, ActScaleN          7

                if (PerChannelQuant) Infos group for each output elemt (Nout) else one group for all out
        */
        int Trace = 1;
        if (Trace)
                NE16_WRITE_REG(NE16_SPECIAL_TRACE_REG, 3);
        unsigned short *__restrict__ StateInOut = Arg->StateInOut;
        unsigned short *__restrict__ Xin = Arg->Xin;
        unsigned short *__restrict__ State = Arg->State;

        unsigned short int DimState = Arg->DimState;
        unsigned short int DimIn = Arg->DimIn;
        unsigned short int DimStateInt = Arg->DimStateInt;
        unsigned short int DimInInt = Arg->DimInInt;

        unsigned short *__restrict__ Hout = Arg->Hout;
        unsigned short int Nout = Arg->Nout;
        signed char *__restrict__ Infos = Arg->Infos;
        int *volatile __restrict__ OutBuff = (int *)Arg->OutBuff;
        int TileOff = Arg->TileOffset;

        unsigned int CoreId = gap_coreid();
        unsigned int ChunkCell;
        unsigned int First;
        unsigned int Last;

        unsigned char *__restrict__ Scale = Arg->ScaleNorm;
        unsigned char *__restrict__ ScaleN = &Arg->ScaleNorm[2 * Nout];
        int *__restrict__ Bf = Arg->Bf;
        int *__restrict__ Bfi = &Arg->Bf[Nout];
        unsigned short OutZeroPoint = *((unsigned short *)&Infos[RNN_NE16_OUT_ZERO_POINT]);

        if (CoreId != 8)
        {
                if (Arg->FirstOut)
                {
                        if (Arg->FirstCell && Arg->Reset)
                        {
                                ZeroState16(State, DimState, CoreId, OutZeroPoint);
                        }
                        else
                        {
                                Copy(State, StateInOut, DimState * 2, CoreId);
                        }
                        gap_waitbarrier(0);
                }
                if (Xin)
                {
                        Copy(State + DimStateInt, Xin, DimIn * 2, CoreId);
                        Copy(&OutBuff[Nout], Bfi, Nout * 4, CoreId);
                        gap_waitbarrier(0);
                }

                Copy(OutBuff, Bf, Nout * 4, CoreId);
                gap_waitbarrier(0);
                ChunkCell = ChunkSize(Nout, 1);
                First = CoreId * ChunkCell;
                Last = Min(First + ChunkCell, Nout);
        }
        else
        {
                if (Trace)
                        printf("NE16: FirstCell %d FirstOut %d Reset %d DimState %d DimIn %d DimStateInt %d DimInInt %d Nout %d\n",
                               Arg->FirstCell, Arg->FirstOut, Arg->Reset, DimState, DimIn, DimStateInt, DimInInt, Nout);
                Last = Nout;
                // Execute NE16 job
                char FilterDataSizeBits = Arg->FilterDataSizeBits;
                unsigned char *__restrict__ Wf = Arg->Wf;
                unsigned char *__restrict__ Wfi = Arg->Wfi;

                // Setup output size
                int Nb_KO = Nout / 32 + (Nout % 32 ? 1 : 0);
                int Rem_KO = Nout % 32 ? Nout % 32 : 32;

                unsigned int cfg = Arg->Default_NE16_Job_Cfg;

                // Zero state padding
                if (Arg->FirstOut)
                {
                        if (Arg->FirstCell)
                        {
                                if (Trace)
                                        printf("%d zero state pad %d %d\n", CoreId, DimState, DimStateInt);
                                ZeroBody(&State[DimState], (DimStateInt-DimState)*2);
                        }
                        gap_waitbarrier(0);
                }
                // Zero input padding
                if (Xin)
                {
                        if (Trace)
                                printf("%d zero Xin pad %d %d\n", CoreId, DimIn, DimInInt);

                        ZeroBody(&State[DimStateInt + DimIn], (DimInInt-DimIn)*2);
                        gap_waitbarrier(0);
                }

                // switch on streamin
                int FiltSize;
                NE16_SETPRIORITY_NE16(); // priority to NE16 w.r.t. cores, DMA
                // printf("Nin %d, Rem_KI %d, Nb_KI %d\n", DimIn, Rem_KI, Nb_KI);

                // load configuration for the layer - input only
                if (Xin)
                {
                        NE16_Manual_SubTile_Linear(
                                cfg, &State[DimStateInt], &OutBuff[Nout], Wfi, DimInInt,
                                Nb_KO, Rem_KO, Infos[RNN_NE16_W_ZERO_POINT], FilterDataSizeBits, 1);
                }

                ChunkCell = ChunkSize(Nout, 1);
                First = CoreId * ChunkCell;
                Last = Nout;

                gap_waitbarrier(0);

                NE16_Manual_SubTile_Linear(
                        cfg, State, OutBuff, Wf, DimStateInt,
                        Nb_KO, Rem_KO, Infos[RNN_NE16_W_ZERO_POINT], FilterDataSizeBits, 0);

                // wait for end of computation
                NE16_BARRIER();

                // set priority to core side
                NE16_SETPRIORITY_CORE();
                if (Trace)
                {
                        dump_u8("inp_scale", Nout, &Scale[Nout]);
                        dump_u8("inp_scale_n", Nout, &ScaleN[Nout]);
                        dump_u8("state_scale", Nout, Scale);
                        dump_u8("state_scale_n", Nout, ScaleN);
                }
        }
        int NoOutScale = (Infos[RNN_NE16_OUT_SCALE] == 1 && Infos[RNN_NE16_OUT_SCALEN] == 0);
        gap_waitbarrier(0);
        if (NoOutScale)
        {
                for (int o = First; o < Last; o++)
                {
                        int InputOut = AT_NORM(AT_NORM(OutBuff[o + Nout], Infos[RNN_NE16_I_PRE_N]) * *((unsigned char *)&Scale[Nout + o]), ScaleN[Nout + o]);
                        int StateOut = AT_NORM(AT_NORM(OutBuff[o], Infos[RNN_NE16_R_PRE_N]) * *((unsigned char *)&Scale[o]), ScaleN[o]);
                        unsigned short Of = ((unsigned short)gap_clip(
                                                Tanh(gap_clip(InputOut + StateOut, 15)),
                                                15)) +
                                            OutZeroPoint;

                        if (StateInOut)
                                StateInOut[TileOff + o] = Of;
                        if (Hout)
                                Hout[o] = Of;
                }
        }
        else
        {
                for (int o = First; o < Last; o++)
                {
                        int InputOut = AT_NORM(AT_NORM(OutBuff[o + Nout], Infos[RNN_NE16_I_PRE_N]) * *((unsigned char *)&Scale[Nout + o]), ScaleN[Nout + o]);
                        int StateOut = AT_NORM(AT_NORM(OutBuff[o], Infos[RNN_NE16_R_PRE_N]) * *((unsigned char *)&Scale[o]), ScaleN[o]);
                        unsigned short Of = ((unsigned short)gap_clip(
                                                AT_NORM(Tanh(gap_clip(InputOut + StateOut, 15)) * *((unsigned char *)&Infos[RNN_NE16_OUT_SCALE]), Infos[RNN_NE16_OUT_SCALEN]),
                                                15)) +
                                            OutZeroPoint;

                        if (StateInOut)
                                StateInOut[TileOff + o] = Of;
                        if (Hout)
                                Hout[o] = Of;
                }
        }
        gap_waitbarrier(0);
}

static inline void LSTM_Queue_Jobs_UInt8(
    unsigned int CoreId,
    unsigned int cfg,
    volatile int *__restrict__ OutBuff1,
    volatile int *__restrict__ OutBuff2,
    volatile int *__restrict__ OutBuff3,
    unsigned char *__restrict__ Wf,
    unsigned char *__restrict__ Wi,
    unsigned char *__restrict__ Wg,
    unsigned char *__restrict__ Wo,
    unsigned int *__restrict__ Bf,
    unsigned int *__restrict__ Bi,
    unsigned int *__restrict__ Bg,
    unsigned int *__restrict__ Bo,
    unsigned char *__restrict__ Wfi,
    unsigned char *__restrict__ Wii,
    unsigned char *__restrict__ Wgi,
    unsigned char *__restrict__ Woi,
    unsigned char *__restrict__ State,
    unsigned char *__restrict__ Scale,
    signed char *__restrict__ Infos,
    int FilterDataSizeBits,
    int Nout,
    int Nb_KO,
    int Rem_KO,
    int DimIn,
    int DimState,
    int DimInInt,
    int DimStateInt,
    int Trace)
{
        NE16_SETPRIORITY_NE16(); // priority to NE16 w.r.t. cores, DMA
        if (Trace)
                NE16_WRITE_REG(NE16_SPECIAL_TRACE_REG, 3);

        // I
        // I input
        int id_i = GetJobId();
        cfg &= ~(NE16_MASK_STREAMIN << NE16_SHIFT_STREAMIN);
        SetupNE16Job(cfg, &State[DimState+DimStateInt], OutBuff1, Wii, &Bi[Nout], 4, DimInInt, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_W_ZEROPOINT]);
        if (Trace)
                printf("Master Queue i INPUT %d %d\n", CoreId, id_i);
        NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);
        // NE16_BARRIER();
        // dump_i32("streamout_accum", Nout, OutBuff1);

        // I state
        Scale += 2 * Nout;
        id_i = GetJobId();
        cfg |= (NE16_MASK_STREAMIN << NE16_SHIFT_STREAMIN);
        SetupNE16Job(cfg, &State[DimState], OutBuff1, Wi, Bi, 4, DimStateInt, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_W_ZEROPOINT]);
        if (Trace)
                printf("Master Queue i STATE %d %d\n", CoreId, id_i);
        NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);

        // G input - I input done when passed
        Scale += 2 * Nout;
        int id_g = GetJobId();

        // G

        cfg &= ~(NE16_MASK_STREAMIN << NE16_SHIFT_STREAMIN);
        SetupNE16Job(cfg, &State[DimState+DimStateInt], OutBuff2, Wgi, &Bg[Nout], 4, DimInInt, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_W_ZEROPOINT]);
        if (Trace)
                printf("Master Queue g INPUT %d %d\n", CoreId, id_g);
        NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);

        // G state - I input and start done when passed
        Scale += 2 * Nout;
        id_g = GetJobId();
        if (Trace)
                printf("Master Done i %d\n", CoreId);
        gap_waitbarrier(0);

        cfg |= (NE16_MASK_STREAMIN << NE16_SHIFT_STREAMIN);
        SetupNE16Job(cfg, &State[DimState], OutBuff2, Wg, Bg, 4, DimStateInt, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_W_ZEROPOINT]);
        if (Trace)
                printf("Master Queue g STATE %d %d\n", CoreId, id_g);
        NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);

        // F

        Scale += 2 * Nout;
        int id_f = GetJobId();
        cfg &= ~(NE16_MASK_STREAMIN << NE16_SHIFT_STREAMIN);
        SetupNE16Job(cfg, &State[DimState+DimStateInt], OutBuff3, Wfi, &Bf[Nout], 4, DimInInt, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_W_ZEROPOINT]);
        if (Trace)
                printf("Master Queue f INPUT %d %d\n", CoreId, id_f);
        NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);

        Scale += 2 * Nout;
        id_f = GetJobId();
        if (Trace)
                printf("Master Done g %d\n", CoreId);
        gap_waitbarrier(0);

        cfg |= (NE16_MASK_STREAMIN << NE16_SHIFT_STREAMIN);
        SetupNE16Job(cfg, &State[DimState], OutBuff3, Wf, Bf, 4, DimStateInt, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_W_ZEROPOINT]);
        if (Trace)
                printf("Master Queue f STATE %d %d\n", CoreId, id_f);
        NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);

        // O

        Scale += 2 * Nout;
        int id_o = GetJobId();
        cfg &= ~(NE16_MASK_STREAMIN << NE16_SHIFT_STREAMIN);
        SetupNE16Job(cfg, &State[DimState+DimStateInt], OutBuff2, Woi, &Bo[Nout], 4, DimInInt, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_W_ZEROPOINT]);
        if (Trace)
                printf("Master Queue o INPUT %d %d\n", CoreId, id_o);
        NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);

        Scale += 2 * Nout;
        id_o = GetJobId();
        if (Trace)
                printf("Master Done f %d\n", CoreId);
        gap_waitbarrier(0);

        cfg |= (NE16_MASK_STREAMIN << NE16_SHIFT_STREAMIN);
        SetupNE16Job(cfg, &State[DimState], OutBuff2, Wo, Bo, 4, DimStateInt, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_W_ZEROPOINT]);
        if (Trace)
                printf("Master Queue o STATE %d %d\n", CoreId, id_o);
        NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);
}

void LSTM_ParKerB32_NE16(KerLSTM_NE16_T *Arg)
{
        /*	Sequences
	 	In:	DimIn!=0, Hout==0
		InOut:	DimIn!=0, Hout!=0
		None:	DimIn==0, Hout==0
		Out:	DimIn==0, Hout!=0

		Infos:
			2 + 2 + 2 + 2 + 6 + 7  LSTM_INT group contains 3 shorts and 1 byte

		In total: 21	LSTM_CELL_INFOS
		if (PerChannelQuant) Infos group for each output elemt (Nout) else one group for all out
	*/
        int Trace = 0;
        int SingleCore = 0;

        unsigned char *__restrict__ StateInOut = Arg->StateInOut;
        unsigned char *__restrict__ State = Arg->State;
        unsigned char *__restrict__ Xin = Arg->Xin;
        unsigned short int DimState = Arg->DimState;
        unsigned short int DimIn = Arg->DimIn;
        unsigned short int DimStateInt = Arg->DimStateInt;
        unsigned short int DimInInt = Arg->DimInInt;
        unsigned char *__restrict__ Hout = Arg->Hout;
        unsigned short int Nout = Arg->Nout;
        signed char *__restrict__ Infos = Arg->Infos;
        int TileOff = Arg->TileOffset;
        int *__restrict__ OutBuff1 = (int *)Arg->OutBuff1;
        int *__restrict__ OutBuff2 = (int *)Arg->OutBuff2;
        int *__restrict__ OutBuff3 = (int *)Arg->OutBuff3;

        unsigned int CoreId = gap_coreid();
        unsigned int ChunkCell = ChunkSize(Nout, 0);
        unsigned int First = CoreId * ChunkCell;
        unsigned int Last = Min(First + ChunkCell, Nout);
        if (Trace)
                printf("Entry %d\n", CoreId);
        if (Trace && SingleCore) {
                if (CoreId == 0) {
                        First = 0;
                        Last = Nout;
                } else {
                        First = Nout;
                }
        }
        if (CoreId != 8)
        {
                if (Trace)
                        printf("core %d\n", CoreId);
                if (Arg->FirstOut)
                {
                        if (Arg->FirstCell && Arg->Reset)
                        {
                                unsigned int StateChunk = ChunkSize(DimState, 0);
                                unsigned int StateFirst = CoreId * StateChunk;
                                unsigned int StateLast = Min(StateFirst + StateChunk, DimState);
                                int Iter = Max(0, StateLast - StateFirst);
                                // Zero cell in - cell in is signed
                                ZeroBody(State + StateFirst, Iter);
                                // Zero h state - h state is unsigned
                                ZeroState8Body(State + DimState + StateFirst, Iter);
                        }
                        else
                        {
                                // c_state then h_state to leave zeros at end
                                Copy(State, StateInOut, DimState*2, CoreId);
                        }
                        if (Trace)
                                printf("%d wait copy/zero state\n", CoreId);
                        gap_waitbarrier(0);
                }
                if (Xin)
                {
                        // copy input past c_state and h_state
                        Copy(State+DimState+DimStateInt, Xin, DimIn, CoreId);
                        if (Trace)
                                printf("%d wait copy input\n", CoreId);
                        gap_waitbarrier(0);
                }

                if (Trace)
                        printf("%d wait start postprocess\n", CoreId);
                gap_waitbarrier(0);
                if (Trace)
                        printf("Start postprocess i %d\n", CoreId);
                for (int o = First; o < Last; o++)
                {
                        /* Oi = HSigmoid(Scaled(Oi)) */
                        OutBuff1[o] = SigmoidLUT(OutBuff1[o], (unsigned short *)&Infos[0]);
                }

                // if (Trace)
                //         printf("Done i %d\n", CoreId);
                gap_waitbarrier(0);
                if (Trace && SingleCore && CoreId == 0) {
                        dump_i32("i_gate_after_act", Nout, OutBuff1);
                }
                if (Trace)
                        printf("Start postprocess g %d\n", CoreId);

                if (Trace && SingleCore && CoreId == 0)
                        printf("%s[%d] = { ", "cstate_c_i", Nout);

                for (int o = First; o < Last; o++)
                {
                        /* Og = HTanh(Scaled(Og)) */
                        /* Half of cell calculation i gate * g gate */
                        int Og = TanhLUT(OutBuff2[o], (unsigned short *)&Infos[0]);
                        OutBuff1[o] = AT_NORM(OutBuff1[o] * Og, 15);
                        if (Trace && SingleCore && CoreId == 0)
                                printf("%d%s", OutBuff1[o], (o == Last - 1 ? "}\n" : ", "));
                }

                // if (Trace)
                //         printf("Done g %d\n", CoreId);
                gap_waitbarrier(0);
                if (Trace && SingleCore && CoreId == 0) {
                        dump_i32("i_gate_times_g_gate", Nout, OutBuff1);
                }
                if (Trace)
                        printf("Start postprocess f %d\n", CoreId);

                if (Trace && SingleCore && CoreId == 0) {
                        printf("CSTATE -- Scale %d ScaleN %d\n", ((unsigned char *)Infos)[LSTM_NE16_CIN_SCALE], ((unsigned char *)Infos)[LSTM_NE16_CIN_SCALEN]);
                        printf("%s[%d] = { ", "cstate_cbar_f_pre_post", Nout);
                }
                for (int o = First; o < Last; o++)
                {
                        /* Of = HSigmoid(Scaled(Of)) */
                        int X1 = SigmoidLUT(OutBuff3[o], (unsigned short *)&Infos[0]);
                        /* Q15 * Cbar scale -> Q15 */
                        if (Trace && SingleCore && CoreId == 0)
                                printf("%d/%d/", X1, ((signed char)State[TileOff + o]));
                        X1 = ((signed char)State[TileOff + o]) * X1;
                        if (Trace && SingleCore && CoreId == 0)
                                printf("%d/", X1);
                        X1 = AT_SCALE(X1, ((unsigned char *)Infos)[LSTM_NE16_CIN_SCALE], ((unsigned char *)Infos)[LSTM_NE16_CIN_SCALEN]);
                        if (Trace && SingleCore && CoreId == 0)
                                printf("%d%s", X1, (o == Last - 1 ? "}\n" : ", "));
                        // Finish compute of X1 in Q15
                        X1 += OutBuff1[o];
                        if (StateInOut)
                                *((signed char *)&StateInOut[TileOff + o]) = (signed char) gap_clip(AT_SCALE(X1, ((unsigned char *)Infos)[LSTM_NE16_COUT_SCALE], ((unsigned char *)Infos)[LSTM_NE16_COUT_SCALEN]), 7);
                        /* Q15 -> Q12 -> Q15 */
                        OutBuff1[o] = TanhLUT(AT_NORM(X1, 3), (unsigned short *)&Infos[0]);
                }

                if (Trace)
                        printf("Done f %d\n", CoreId);
                gap_waitbarrier(0);
                if (Trace)
                        printf("Start postprocess o %d\n", CoreId);
                if (Trace && SingleCore && CoreId == 0)
                        printf("%s[%d] = { ", "output_before_scale", Nout);
                for (int o = First; o < Last; o++)
                {
                        /* Oo = HSigmoid(Scaled(Oo)) */
                        int Oo = SigmoidLUT(OutBuff2[o], (unsigned short *)&Infos[0]);
                        Oo = AT_NORM(Oo * OutBuff1[o], 15);
                        if (Trace && SingleCore && CoreId == 0)
                                printf("%d%s", Oo, (o == Last - 1 ? "}\n" : ", "));
                        unsigned char X2 = (unsigned char)(
                                gap_clipu(
                                        AT_SCALE(
                                                Oo,
                                                ((unsigned char *)Infos)[LSTM_NE16_OUT_SCALE],
                                                ((unsigned char *)Infos)[LSTM_NE16_OUT_SCALEN]
                                        ) + (unsigned char)Infos[LSTM_NE16_OUT_ZEROPOINT],
                                        8));
                        if (StateInOut)
                                StateInOut[TileOff + DimState + o] = X2;
                        if (Hout)
                                Hout[o] = X2;
                }
                if (Trace)
                        printf("Done o %d\n", CoreId);
        }
        else
        {
                if (Trace)
                        printf("Master core %d DimState %d DimIn %d Nout %d\n", CoreId, DimState, DimIn, Nout);

                int Nb_KI, Rem_KI;
                int Nb_KO = Nout / 32 + (Nout % 32 ? 1 : 0);
                int Rem_KO = Nout % 32 ? Nout % 32 : 32; // Check different wrt simulator
                char FilterDataSizeBits = Arg->FilterDataSizeBits;
                signed char *Scale = Arg->ScaleNorm;

                unsigned int cfg = Arg->Default_NE16_Job_Cfg;

                if (Arg->FirstOut)
                {
                        if (Arg->FirstCell)
                        {
                                ZeroBody(&State[DimState*2], DimStateInt-DimState);
                        }
                        if (Trace)
                                printf("%d wait zero state pad\n", CoreId);
                        gap_waitbarrier(0);
                }
                if (Xin)
                {
                        ZeroBody(&State[DimStateInt+DimState+DimIn], DimInInt-DimIn);
                        if (Trace)
                                printf("%d wait zero input pad\n", CoreId);
                        gap_waitbarrier(0);
                }

                LSTM_Queue_Jobs_UInt8(
                        CoreId, cfg, OutBuff1, OutBuff2, OutBuff3,
                        Arg->Wf, Arg->Wi, Arg->Wg, Arg->Wo,
                        Arg->Bf, Arg->Bi, Arg->Bg, Arg->Bo,
                        Arg->Wfi, Arg->Wii, Arg->Wgi, Arg->Woi,
                        State, Scale, Infos, FilterDataSizeBits, Nout,
                        Nb_KO, Rem_KO, DimIn, DimState, DimInInt, DimStateInt,
                        Trace);

                NE16_BARRIER();

                if (Trace)
                        printf("Master Done o %d %d\n", CoreId);
                gap_waitbarrier(0);
                // set priority to core side
                NE16_SETPRIORITY_CORE();
        }
        if (Trace)
                printf("Final wait %d\n", CoreId);
        gap_waitbarrier(0);
}

static inline void LSTM_Queue_Jobs_UInt16(
    unsigned int CoreId,
    unsigned int cfg,
    volatile int *__restrict__ OutBuff1,
    volatile int *__restrict__ OutBuff2,
    volatile int *__restrict__ OutBuff3,
    unsigned char *__restrict__ Wf,
    unsigned char *__restrict__ Wi,
    unsigned char *__restrict__ Wg,
    unsigned char *__restrict__ Wo,
    unsigned char *__restrict__ Wfi,
    unsigned char *__restrict__ Wii,
    unsigned char *__restrict__ Wgi,
    unsigned char *__restrict__ Woi,
    unsigned short *__restrict__ State,
    signed char *__restrict__ Infos,
    int FilterDataSizeBits,
    int Nout,
    int Nb_KO,
    int Rem_KO,
    int DimIn,
    int DimState,
    int DimInInt,
    int DimStateInt,
    pStageDesc_t pStageDesc,
    int Trace)
{
        if (Trace)
                NE16_WRITE_REG(NE16_SPECIAL_TRACE_REG, 3);

        // I
        // I input

        int Inp_Nb_KI, Inp_Rem_KI, Inp_Fs, State_Nb_KI, State_Rem_KI, State_Fs;
        NE16_Manual_SubTile_Linear_Setup(cfg, DimInInt, FilterDataSizeBits, &Inp_Nb_KI, &Inp_Rem_KI, &Inp_Fs);
        NE16_Manual_SubTile_Linear_Setup(cfg, DimStateInt, FilterDataSizeBits, &State_Nb_KI, &State_Rem_KI, &State_Fs);

        gap_waitbarrier(0); // Extra sync for cores to setup OutBuff1
        NE16_SETPRIORITY_NE16(); // priority to NE16 w.r.t. cores, DMA
        if (Trace)
                printf("Master Queue i INPUT %d\n", CoreId);
        NE16_Manual_SubTile_Linear_Body(
                cfg, &State[DimState+DimStateInt], &OutBuff1[Nout], Wii, DimInInt,
                Inp_Nb_KI, Inp_Rem_KI, Nb_KO, Rem_KO, Infos[LSTM_NE16_W_ZEROPOINT],
                FilterDataSizeBits, Inp_Fs, 1, -1, pStageDesc);
        // dump_i32("streamout_accum", Nout, OutBuff1);

        // I state
        if (Trace)
                printf("Master Queue i STATE %d\n", CoreId);
        NE16_Manual_SubTile_Linear_Body(
                cfg, &State[DimState], OutBuff1, Wi, DimStateInt,
                State_Nb_KI, State_Rem_KI, Nb_KO, Rem_KO, Infos[LSTM_NE16_W_ZEROPOINT],
                FilterDataSizeBits, State_Fs, 0, -1, pStageDesc);

        // G
        // G input

        // Extra sync for cores to setup OutBuff2 & 3
        gap_waitbarrier(0); // 2

        if (Trace)
                printf("Master Queue g INPUT %d\n", CoreId);
        NE16_Manual_SubTile_Linear_Body(
                cfg, &State[DimState+DimStateInt], &OutBuff2[Nout], Wgi, DimInInt,
                Inp_Nb_KI, Inp_Rem_KI, Nb_KO, Rem_KO, Infos[LSTM_NE16_W_ZEROPOINT],
                FilterDataSizeBits, Inp_Fs, 1, -1, pStageDesc);

        // G state - I input and state done when passed
        int JobId = GetJobId();
        if (Trace)
                printf("Master Done i %d\n", CoreId);
        // I DONE
        if (Trace)
                printf("Master Queue g STATE %d\n", CoreId);
        NE16_Manual_SubTile_Linear_Body(
                cfg, &State[DimState], OutBuff2, Wg, DimStateInt,
                State_Nb_KI, State_Rem_KI, Nb_KO, Rem_KO, Infos[LSTM_NE16_W_ZEROPOINT],
                FilterDataSizeBits, State_Fs, 0, JobId, pStageDesc);


        // F
        if (Trace)
                printf("Master Queue f INPUT %d\n", CoreId);
        NE16_Manual_SubTile_Linear_Body(
                cfg, &State[DimState+DimStateInt], &OutBuff3[Nout], Wfi, DimInInt,
                Inp_Nb_KI, Inp_Rem_KI, Nb_KO, Rem_KO, Infos[LSTM_NE16_W_ZEROPOINT],
                FilterDataSizeBits, Inp_Fs, 1, -1, pStageDesc);

        JobId = GetJobId();
        if (Trace)
                printf("Master Done g %d\n", CoreId);
        // G DONE
        if (Trace)
                printf("Master Queue f STATE %d\n", CoreId);
        NE16_Manual_SubTile_Linear_Body(
                cfg, &State[DimState], OutBuff3, Wf, DimStateInt,
                State_Nb_KI, State_Rem_KI, Nb_KO, Rem_KO, Infos[LSTM_NE16_W_ZEROPOINT],
                FilterDataSizeBits, State_Fs, 0, JobId, pStageDesc);


        // O
        // Extra sync for cores to setup OutBuff2
        gap_waitbarrier(0); // 3
        if (Trace)
                printf("Master Queue o INPUT %d\n", CoreId);
        NE16_Manual_SubTile_Linear_Body(
                cfg, &State[DimState+DimStateInt], &OutBuff2[Nout], Woi, DimInInt,
                Inp_Nb_KI, Inp_Rem_KI, Nb_KO, Rem_KO, Infos[LSTM_NE16_W_ZEROPOINT],
                FilterDataSizeBits, Inp_Fs, 1, -1, pStageDesc);

        JobId = GetJobId();
        if (Trace)
                printf("Master Done f %d\n", CoreId);
        // F DONE
        if (Trace)
                printf("Master Queue o STATE %d\n", CoreId);
        NE16_Manual_SubTile_Linear_Body(
                cfg, &State[DimState], OutBuff2, Wo, DimStateInt,
                State_Nb_KI, State_Rem_KI, Nb_KO, Rem_KO, Infos[LSTM_NE16_W_ZEROPOINT],
                FilterDataSizeBits, State_Fs, 0, JobId, pStageDesc);

}

void LSTM_ParKerB32_NE16fp(KerLSTM_NE16fp_T *Arg)
{
        /*	Sequences
	 	In:	DimIn!=0, Hout==0
		InOut:	DimIn!=0, Hout!=0
		None:	DimIn==0, Hout==0
		Out:	DimIn==0, Hout!=0

		Infos:
			2 + 2 + 2 + 2 + 6 + 7  LSTM_INT group contains 3 shorts and 1 byte

		In total: 21	LSTM_CELL_INFOS
		if (PerChannelQuant) Infos group for each output elemt (Nout) else one group for all out
	*/
        int Trace = 0;
        int SingleCore = 0;

        unsigned short *__restrict__ StateInOut = Arg->StateInOut;
        unsigned short *__restrict__ State = Arg->State;
        unsigned short *__restrict__ Xin = Arg->Xin;
        unsigned short int DimState = Arg->DimState;
        unsigned short int DimIn = Arg->DimIn;
        unsigned short int DimStateInt = Arg->DimStateInt;
        unsigned short int DimInInt = Arg->DimInInt;
        unsigned short *__restrict__ Hout = Arg->Hout;
        unsigned short int Nout = Arg->Nout;
        signed char *__restrict__ Infos = Arg->Infos;
        int TileOff = Arg->TileOffset;
        int *__restrict__ OutBuff1 = (int *)Arg->OutBuff1;
        int *__restrict__ OutBuff2 = (int *)Arg->OutBuff2;
        int *__restrict__ OutBuff3 = (int *)Arg->OutBuff3;

        unsigned int CoreId = gap_coreid();
        unsigned int ChunkCell = ChunkSize(Nout, 0);
        unsigned int First = CoreId * ChunkCell;
        unsigned int Last = Min(First + ChunkCell, Nout);
        if (Trace) {
                printf("Entry %d\n", CoreId);
                if (CoreId == 8) {
                        dump_16("cstate_in", DimState, StateInOut);
                        dump_u16("hstate_in", DimState, &StateInOut[DimState]);
                }
                gap_waitbarrier(0);
                if (SingleCore) {
                        if (CoreId == 0) {
                                First = 0;
                                Last = Nout;
                        } else {
                                First = Nout;
                        }
                }
        }

        if (CoreId != 8)
        {
                unsigned char * Scalei = Arg->ScaleNorm;
                unsigned char * Scaler = &Arg->ScaleNorm[2*Nout];
                int GatePrenorm = Infos[LSTM_NE16_GATE_PRENORM];
                if (Trace)
                        printf("core %d FirstOut %d FirstCell %d Reset %d GatePrenorm %d\n", CoreId, Arg->FirstOut, Arg->FirstCell, Arg->Reset, GatePrenorm);
                if (Arg->FirstOut)
                {
                        if (Arg->FirstCell && Arg->Reset)
                        {
                                unsigned int StateChunk = ChunkSize(DimState, 0);
                                unsigned int StateFirst = CoreId * StateChunk;
                                unsigned int StateLast = Min(StateFirst + StateChunk, DimState);
                                int Iter = Max(0, StateLast - StateFirst);
                                // Zero cell in - cell in is signed
                                ZeroBody(State + StateFirst, Iter * 2);
                                // Zero h state - h state is unsigned
                                ZeroState16Body(State + DimState + StateFirst, Iter, *((unsigned short *)&Infos[LSTM_NE16_OUT_ZEROPOINT]));
                                if (Trace)
                                        printf("%d wait zero state\n", CoreId);
                        }
                        else
                        {
                                // c_state then h_state to leave zeros at end
                                Copy(State, StateInOut, DimState * 4, CoreId);
                                if (Trace)
                                        printf("%d wait copy state\n", CoreId);
                        }
                        gap_waitbarrier(0);
                }
                if (Xin)
                {
                        // copy input past c_state and h_state
                        Copy(State+DimState+DimStateInt, Xin, DimIn * 2, CoreId);
                        if (Trace)
                                printf("%d wait copy input\n", CoreId);
                        gap_waitbarrier(0);
                }

                for (int o = First; o < Last; o++) {
                        OutBuff1[o] = Arg->Bi[o];
                        OutBuff1[o+Nout] = Arg->Bi[o+Nout];
                }
                gap_waitbarrier(0); // OutBuff1 Loaded

                for (int o = First; o < Last; o++) {
                        OutBuff2[o] = Arg->Bg[o];
                        OutBuff2[o+Nout] = Arg->Bg[o+Nout];
                        OutBuff3[o] = Arg->Bf[o];
                        OutBuff3[o+Nout] = Arg->Bf[o+Nout];
                }
                gap_waitbarrier(0); // OutBuff2 & 3 Loaded

                if (Trace)
                        printf("%d wait start postprocess\n", CoreId);
                StageDescTestCompleted(Arg->pStageDesc, 1);
                if (Trace)
                        printf("Start postprocess i %d\n", CoreId);
                for (int o = First; o < Last; o++)
                {
                        /* Oi = HSigmoid(Scaled(Oi)) */
                        int Oi = AT_SCALE(gap_roundnorm_reg(OutBuff1[o], GatePrenorm), Scaler[o], Scaler[o+Nout]);
                        Oi += AT_SCALE(gap_roundnorm_reg(OutBuff1[o+Nout], GatePrenorm), Scalei[o], Scalei[o+Nout]);
                        OutBuff1[o] = SigmoidLUT(Oi, (unsigned short *)&Infos[0]);
                }

                // if (Trace)
                //         printf("Done i %d\n", CoreId);
                StageDescTestCompleted(Arg->pStageDesc, 2);
                Scaler += Nout * 4;
                Scalei += Nout * 4;
                if (Trace && SingleCore && CoreId == 0) {
                        dump_i32("i_gate_after_act", Nout, OutBuff1);
                }
                if (Trace)
                        printf("Start postprocess g %d\n", CoreId);

                if (Trace && SingleCore && CoreId == 0)
                        printf("%s[%d] = { ", "c_gate_after_act", Nout);

                for (int o = First; o < Last; o++)
                {
                        /* Og = HTanh(Scaled(Og)) */
                        /* Half of cell calculation i gate * g gate */
                        int Og = AT_SCALE(gap_roundnorm_reg(OutBuff2[o], GatePrenorm), Scaler[o], Scaler[o+Nout]);
                        Og += AT_SCALE(gap_roundnorm_reg(OutBuff2[o+Nout], GatePrenorm), Scalei[o], Scalei[o+Nout]);
                        Og = TanhLUT(Og, (unsigned short *)&Infos[0]);
                        if (Trace && SingleCore && CoreId == 0)
                                printf("%d%s", Og, (o == Last - 1 ? "}\n" : ", "));
                        OutBuff1[o] = AT_NORM(OutBuff1[o] * Og, 15);
                        OutBuff2[o] = Arg->Bo[o];
                        OutBuff2[o+Nout] = Arg->Bo[o+Nout];
                }

                gap_waitbarrier(0); // OutBuff2 loaded
                StageDescTestCompleted(Arg->pStageDesc, 3);
                Scaler += Nout * 4;
                Scalei += Nout * 4;
                if (Trace)
                        printf("Done g %d\n", CoreId);
                if (Trace && SingleCore && CoreId == 0) {
                        dump_i32("i_gate_times_g_gate", Nout, OutBuff1);
                }
                if (Trace)
                        printf("Start postprocess f %d\n", CoreId);

                if (Trace && SingleCore && CoreId == 0) {
                        printf("CSTATE -- Scale %d ScaleN %d\n", ((unsigned char *)Infos)[LSTM_NE16_CIN_SCALE], ((unsigned char *)Infos)[LSTM_NE16_CIN_SCALEN]);
                        printf("%s[%d] = { ", "cstate_cbar_f_pre_post", Nout);
                }
                for (int o = First; o < Last; o++)
                {
                        /* Of = HSigmoid(Scaled(Of)) */
                        int X1 = AT_SCALE(gap_roundnorm_reg(OutBuff3[o],GatePrenorm), Scaler[o], Scaler[o+Nout]);
                        X1 += AT_SCALE(gap_roundnorm_reg(OutBuff3[o+Nout],GatePrenorm), Scalei[o], Scalei[o+Nout]);
                        X1 = SigmoidLUT(X1, (unsigned short *)&Infos[0]);
                        /* Q15 * Cbar scale -> Q15 */
                        if (Trace && SingleCore && CoreId == 0)
                                printf("%d/%d/", X1, ((signed short)State[TileOff + o]));
                        X1 = ((short)State[TileOff + o]) * X1;
                        if (Trace && SingleCore && CoreId == 0)
                                printf("%d/", X1);
                        X1 = AT_SCALE(AT_NORM(X1, 8), ((unsigned char *)Infos)[LSTM_NE16_CIN_SCALE], ((unsigned char *)Infos)[LSTM_NE16_CIN_SCALEN]);
                        if (Trace && SingleCore && CoreId == 0)
                                printf("%d%s", X1, (o == Last - 1 ? "}\n" : ", "));
                        // Finish compute of X1 in Q15
                        X1 += OutBuff1[o];
                        if (StateInOut)
                                *((short *)&StateInOut[TileOff + o]) = (short) gap_clip(AT_SCALE(X1, ((unsigned char *)Infos)[LSTM_NE16_COUT_SCALE], ((unsigned char *)Infos)[LSTM_NE16_COUT_SCALEN]), 15);
                        /* Q15 -> Q12 -> Q15 */
                        OutBuff1[o] = TanhLUT(AT_NORM(X1, 3), (unsigned short *)&Infos[0]);
                }

                if (Trace)
                        printf("Done f %d\n", CoreId);
                gap_waitbarrier(0);
                Scaler += Nout * 4;
                Scalei += Nout * 4;
                if (Trace)
                        printf("Start postprocess o %d\n", CoreId);
                if (Trace && SingleCore && CoreId == 0)
                        printf("%s[%d] %d = { ", "o_gate_after_act", Nout, TileOff);
                for (int o = First; o < Last; o++)
                {
                        /* Oo = HSigmoid(Scaled(Oo)) */
                        int Oo = AT_SCALE(gap_roundnorm_reg(OutBuff2[o],GatePrenorm), Scaler[o], Scaler[o+Nout]);
                        Oo += AT_SCALE(gap_roundnorm_reg(OutBuff2[o+Nout],GatePrenorm), Scalei[o], Scalei[o+Nout]);
                        Oo = SigmoidLUT(Oo, (unsigned short *)&Infos[0]);
                        if (Trace && SingleCore && CoreId == 0)
                                printf("%d%s", Oo, (o == Last - 1 ? "}\n" : ", "));
                        Oo = AT_NORM(Oo * OutBuff1[o], 15);
                        unsigned short X2 = (unsigned short)(
                                gap_clipu(
                                        AT_SCALE(
                                                Oo,
                                                ((unsigned char *)Infos)[LSTM_NE16_OUT_SCALE],
                                                ((unsigned char *)Infos)[LSTM_NE16_OUT_SCALEN]
                                        ) + *((unsigned short *)&Infos[LSTM_NE16_OUT_ZEROPOINT]),
                                        16));
                        if (StateInOut)
                                StateInOut[TileOff + DimState + o] = X2;
                        if (Hout)
                                Hout[o] = X2;
                }
                if (Trace)
                        printf("Done o %d\n", CoreId);
        }
        else
        {
                if (Trace)
                        printf("Master core %d DimStateInt %d DimInInt %d Nout %d\n", CoreId, DimStateInt, DimInInt, Nout);

                int Nb_KI, Rem_KI;
                int Nb_KO = Nout / 32 + (Nout % 32 ? 1 : 0);
                int Rem_KO = Nout % 32 ? Nout % 32 : 32; // Check different wrt simulator
                char FilterDataSizeBits = Arg->FilterDataSizeBits;
                signed char *Scale = Arg->ScaleNorm;

                unsigned int cfg = Arg->Default_NE16_Job_Cfg;


                if (Arg->FirstOut)
                {
                        if (Arg->FirstCell)
                        {       if (Trace)
                                        printf("First out / First Cell\n");
                                StageDescAlloc(Arg->pStageDesc, Nb_KO*2);
                                ZeroBody(&State[DimState*2], (DimStateInt-DimState)*2);
                        } else {
                               StageDescReset(Arg->pStageDesc, Nb_KO*2);
                        }
                        if (Trace)
                                printf("%d wait zero state pad\n", CoreId);
                        gap_waitbarrier(0);
                } else {
                        StageDescReset(Arg->pStageDesc, Nb_KO*2);
                }
                if (Xin)
                {
                        ZeroBody(&State[DimStateInt+DimState+DimIn], (DimInInt-DimIn)*2);
                        if (Trace)
                                printf("%d wait zero input pad\n", CoreId);
                        gap_waitbarrier(0);
                }

                LSTM_Queue_Jobs_UInt16(
                        CoreId, cfg, OutBuff1, OutBuff2, OutBuff3,
                        Arg->Wf, Arg->Wi, Arg->Wg, Arg->Wo,
                        Arg->Wfi, Arg->Wii, Arg->Wgi, Arg->Woi,
                        State, Infos, FilterDataSizeBits, Nout,
                        Nb_KO, Rem_KO, DimIn, DimState, DimInInt, DimStateInt,
                        Arg->pStageDesc, Trace);

                NE16_BARRIER();

                if (Trace)
                        printf("Master Done o %d %d\n", CoreId);
                NE16_SETPRIORITY_CORE();
                // set priority to core side and signal all cores
                gap_waitbarrier(0);
                if (Arg->LastCell && Arg->LastOut)
                        StageDescFree(Arg->pStageDesc);
        }
        if (Trace)
                printf("Final wait %d\n", CoreId);
        gap_waitbarrier(0);
        if (Trace) {
                if (CoreId == 8) {
                        dump_16("cstate_out", DimState, StateInOut);
                        dump_u16("hstate_out", DimState, &StateInOut[DimState]);
                }
                gap_waitbarrier(0);
        }
}

// void LSTM_ParKerB32_Hard_NE16(KerLSTM_NE16_T *Arg)

// {
//         /*	Sequences
// 	 	In:	DimIn!=0, Hout==0
// 		InOut:	DimIn!=0, Hout!=0
// 		None:	DimIn==0, Hout==0
// 		Out:	DimIn==0, Hout!=0

// 		Infos:
// 			2 + 2 + 2 + 2 + 6 + 7  LSTM_INT group contains 3 shorts and 1 byte

// 		In total: 21	LSTM_CELL_INFOS
// 		if (PerChannelQuant) Infos group for each output elemt (Nout) else one group for all out
// 	   */
//         int Trace = 0;
//         unsigned char *__restrict__ StateInOut = Arg->StateInOut;
//         unsigned char *__restrict__ State = Arg->State;
//         unsigned char *__restrict__ Xin = Arg->Xin;
//         unsigned short int DimState = Arg->DimState;
//         unsigned short int DimIn = Arg->DimIn;
//         unsigned char *__restrict__ Hout = Arg->Hout;
//         unsigned short int Nout = Arg->Nout;
//         signed char *__restrict__ Infos = Arg->Infos;
//         int TileOff = Arg->TileOffset;
//         int *__restrict__ OutBuff1 = (int *)Arg->OutBuff1;
//         int *__restrict__ OutBuff2 = (int *)Arg->OutBuff2;
//         int *__restrict__ OutBuff3 = (int *)Arg->OutBuff3;

//         unsigned int Nin = DimState + DimIn;
//         unsigned int NS = Nin;
//         unsigned int CoreId = gap_coreid();
//         unsigned int ChunkCell = ChunkSize(Nout, 0);
//         unsigned int First = CoreId * ChunkCell;
//         unsigned int Last = Min(First + ChunkCell, Nout);
//         if (Trace)
//                 printf("Entry %d\n", CoreId);

//         if (CoreId != 8)
//         {
//                 if (Trace)
//                         printf("core %d\n", CoreId);
//                 if (Arg->FirstOut)
//                 {
//                         if (Trace)
//                                 printf("Init state %d\n", CoreId);
//                         if (Arg->FirstCell && Arg->Reset)
//                                 Zero(State, 2 * DimState, CoreId);
//                         else
//                                 Copy(State, StateInOut, 2 * DimState, CoreId);
//                         gap_waitbarrier(0);
//                 }

//                 if (Trace)
//                         printf("Done copy %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 if (Trace)
//                         printf("Start i %d\n", CoreId);
//                 for (int o = First; o < Last; o++)
//                 {
//                         /* Oi = HSigmoid(Scaled(Oi)) */
//                         OutBuff1[o] = AT_NORM(AT_CLIP_POS(OutBuff1[o] + *(short *)&Infos[LSTM_NE16_INT_B0], *((short *)&Infos[LSTM_NE16_INT_A0])) * *((short *)&Infos[LSTM_NE16_INT_C0]), ((unsigned char *)Infos)[LSTM_NE16_INT_Q]);
//                 }

//                 if (Trace)
//                         printf("Done i %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 if (Trace)
//                         printf("Start g %d\n", CoreId);
//                 int one = 1 << ((unsigned char *)Infos)[LSTM_NE16_INT_Q];
//                 for (int o = First; o < Last; o++)
//                 {
//                         /* Og = HTanh(Scaled(Og)) */
//                         int Og = Max(-one, Min(one, OutBuff2[o]));
//                         // Og = AT_CLIP(Og, One);
//                         // Compute Oi * Og
//                         OutBuff1[o] *= Og;
//                 }

//                 if (Trace)
//                         printf("Done g %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 if (Trace)
//                         printf("Start f %d\n", CoreId);
//                 one = one << Infos[LSTM_NE16_INT_Q];
//                 for (int o = First; o < Last; o++)
//                 {
//                         /* Of = HSigmoid(Scaled(Of)) */
//                         int Of = AT_NORM(AT_CLIP_POS(OutBuff3[o] + *(short *)&Infos[LSTM_NE16_INT_B0], *((short *)&Infos[LSTM_NE16_INT_A0])) * *((short *)&Infos[LSTM_NE16_INT_C0]), ((unsigned char *)Infos)[LSTM_NE16_INT_Q]);
//                         int X1 = AT_SCALE(State[TileOff + o], ((unsigned char *)Infos)[LSTM_NE16_CIN_SCALE], ((unsigned char *)Infos)[LSTM_NE16_CIN_SCALEN]);
//                         // Finish compute of X1
//                         X1 = Of * X1 + OutBuff1[o];
//                         if (StateInOut)
//                                 StateInOut[TileOff + o] = gap_clipu(AT_SCALE(X1, ((unsigned char *)Infos)[LSTM_NE16_COUT_SCALE], ((unsigned char *)Infos)[LSTM_NE16_COUT_SCALEN]), 8);
//                         X1 = Max(-one, Min(one, X1));
//                         OutBuff1[o] = X1;
//                 }

//                 if (Trace)
//                         printf("Done f %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 if (Trace)
//                         printf("Start o %d\n", CoreId);
//                 for (int o = First; o < Last; o++)
//                 {
//                         /* Oo = HSigmoid(Scaled(Oo)) */
//                         int Oo = AT_NORM(AT_CLIP_POS(OutBuff2[o] + *((short *)&Infos[LSTM_NE16_INT_B0]), *((short *)&Infos[LSTM_NE16_INT_A0])) * *((short *)&Infos[LSTM_NE16_INT_C0]), ((unsigned char *)Infos)[LSTM_NE16_INT_Q]);
//                         int X2 = gap_clipu(AT_SCALE(Oo * OutBuff1[o], ((unsigned char *)Infos)[LSTM_NE16_OUT_SCALE], ((unsigned char *)Infos)[LSTM_NE16_OUT_SCALEN]), 8);
//                         if (StateInOut)
//                                 StateInOut[TileOff + DimState + o] = X2;
//                         if (Hout)
//                                 Hout[o] = X2;
//                 }
//                 if (Trace)
//                         printf("Done o %d\n", CoreId);
//         }
//         else
//         {
//                 if (Trace)
//                         printf("Master core %d\n", CoreId);

//                 unsigned char *__restrict__ Wf = Arg->Wf;
//                 int *__restrict__ Bf = Arg->Bf;
//                 unsigned char *__restrict__ Wi = Arg->Wi;
//                 int *__restrict__ Bi = Arg->Bi;
//                 unsigned char *__restrict__ Wg = Arg->Wg;
//                 int *__restrict__ Bg = Arg->Bg;
//                 unsigned char *__restrict__ Wo = Arg->Wo;
//                 int *__restrict__ Bo = Arg->Bo;
//                 int Nb_KI, Rem_KI;
//                 int Nb_KO = Nout / 32 + (Nout % 32 ? 1 : 0);
//                 int Rem_KO = Nout % 32 ? Nout % 32 : 32; // Check different wrt simulator
//                 char FilterDataSizeBits = Arg->FilterDataSizeBits;
//                 unsigned char *Scale = Arg->ScaleNorm;

//                 unsigned int cfg = Arg->Default_NE16_Job_Cfg;

//                 int id_i = GetJobId();
//                 SetupNE16Job(cfg, State, OutBuff1, Wi, Bi, 1, Nin, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_I_ZEROPOINT]);
//                 if (Arg->FirstOut)
//                 {
//                         if (Trace)
//                                 printf("Master wait init state %d\n", CoreId);
//                         gap_waitbarrier(0);
//                 }
//                 if (Xin)
//                 {
//                         if (Trace)
//                                 printf("Master wait init input %d\n", CoreId);
//                         gap_waitbarrier(0);
//                 }

//                 NE16_SETPRIORITY_NE16(); // priority to NE16 w.r.t. cores, DMA
//                 if (Trace)
//                         printf("Master Queue i %d %d\n", CoreId, id_i);
//                 NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);

//                 int id_g = GetJobId();
//                 SetupNE16Job(cfg, State, OutBuff2, Wg, Bg, 1, Nin, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_G_ZEROPOINT]);
//                 if (Trace)
//                         printf("Master Queue g %d %d\n", CoreId, id_g);
//                 NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);

//                 int id_f = GetJobId();
//                 if (Trace)
//                         printf("Master Done i %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 SetupNE16Job(cfg, State, OutBuff3, Wf, Bf, 1, Nin, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_F_ZEROPOINT]);

//                 if (Trace)
//                         printf("Master Queue f %d %d\n", CoreId, id_f);
//                 NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);
//                 int id_o = GetJobId();
//                 if (Trace)
//                         printf("Master Done g %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 SetupNE16Job(cfg, State, OutBuff2, Wo, Bo, 1, Nin, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_O_ZEROPOINT]);

//                 if (Trace)
//                         printf("Master Queue o %d %d\n", CoreId, id_o);
//                 NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);

//                 // Wait for <= 1 job
//                 while (1)
//                 {
//                         int status = NE16_READ_STATUS();
//                         int job = NE16_READ_RUNNING_JOB();
//                         if (Trace)
//                                 printf("Status %d job %d\n", status, job);
//                         if (status == 0 || job != id_f)
//                                 break;
//                         NE16_BARRIER_NOSTATUS();
//                 }
//                 if (Trace)
//                         printf("Master Done f %d %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 // wait for end of computation
//                 NE16_BARRIER();

//                 if (Trace)
//                         printf("Master Done o %d %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 // set priority to core side
//                 NE16_SETPRIORITY_CORE();
//         }
//         if (Trace)
//                 printf("Final wait %d\n", CoreId);
//         gap_waitbarrier(0);
// }

// void LSTM_ParKerB32_Hard_SameInStateScale_NE16(KerLSTM_NE16_T *Arg)

// {
//         /*	Sequences
// 	 	In:	DimIn!=0, Hout==0
// 		InOut:	DimIn!=0, Hout!=0
// 		None:	DimIn==0, Hout==0
// 		Out:	DimIn==0, Hout!=0

// 		Infos:
// 			2 + 2 + 2 + 2 + 6 + 7  LSTM_INT group contains 3 shorts and 1 byte

// 		In total: 21	LSTM_CELL_INFOS
// 		if (PerChannelQuant) Infos group for each output elemt (Nout) else one group for all out
// 	*/
//         int Trace = 0;
//         unsigned char *__restrict__ StateInOut = Arg->StateInOut;
//         unsigned char *__restrict__ State = Arg->State;
//         unsigned char *__restrict__ Xin = Arg->Xin;
//         unsigned short int DimState = Arg->DimState;
//         unsigned short int DimIn = Arg->DimIn;
//         unsigned char *__restrict__ Hout = Arg->Hout;
//         unsigned short int Nout = Arg->Nout;
//         signed char *__restrict__ Infos = Arg->Infos;
//         int TileOff = Arg->TileOffset;
//         int *__restrict__ OutBuff1 = (int *)Arg->OutBuff1;
//         int *__restrict__ OutBuff2 = (int *)Arg->OutBuff2;
//         int *__restrict__ OutBuff3 = (int *)Arg->OutBuff3;

//         unsigned int Nin = DimState + DimIn;
//         unsigned int NS = Nin;
//         unsigned int CoreId = gap_coreid();
//         unsigned int ChunkCell = ChunkSize(Nout, 0);
//         unsigned int First = CoreId * ChunkCell;
//         unsigned int Last = Min(First + ChunkCell, Nout);
//         if (Trace)
//                 printf("Entry %d\n", CoreId);

//         if (CoreId != 8)
//         {
//                 if (Trace)
//                         printf("core %d\n", CoreId);
//                 if (Arg->FirstOut)
//                 {
//                         if (Trace)
//                                 printf("Init state %d\n", CoreId);
//                         if (Arg->FirstCell && Arg->Reset)
//                                 Zero(State, 2 * DimState, CoreId);
//                         else
//                                 Copy(State, StateInOut, 2 * DimState, CoreId);
//                         gap_waitbarrier(0);
//                 }
//                 if (Xin)
//                 {
//                         if (Trace)
//                                 printf("Copy Xin %d\n", CoreId);
//                         Copy(State + 2 * DimState, Xin, DimIn, CoreId);
//                         gap_waitbarrier(0);
//                 }

//                 if (Trace)
//                         printf("Done copy %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 if (Trace)
//                         printf("Start i %d\n", CoreId);
//                 for (int o = First; o < Last; o++)
//                 {
//                         /* Oi = HSigmoid(Scaled(Oi)) */
//                         OutBuff1[o] = AT_NORM(AT_CLIP_POS(OutBuff1[o] + *(short *)&Infos[LSTM_NE16_INT_B0], *((short *)&Infos[LSTM_NE16_INT_A0])) * *((short *)&Infos[LSTM_NE16_INT_C0]), ((unsigned char *)Infos)[LSTM_NE16_INT_Q]);
//                 }

//                 if (Trace)
//                         printf("Done i %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 if (Trace)
//                         printf("Start g %d\n", CoreId);
//                 int one = 1 << ((unsigned char *)Infos)[LSTM_NE16_INT_Q];
//                 for (int o = First; o < Last; o++)
//                 {
//                         /* Og = HTanh(Scaled(Og)) */
//                         int Og = Max(-one, Min(one, OutBuff2[o]));
//                         // Og = AT_CLIP(Og, One);
//                         // Compute Oi * Og
//                         OutBuff1[o] *= Og;
//                 }

//                 if (Trace)
//                         printf("Done g %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 if (Trace)
//                         printf("Start f %d\n", CoreId);
//                 one = one << Infos[LSTM_NE16_INT_Q];
//                 for (int o = First; o < Last; o++)
//                 {
//                         /* Of = HSigmoid(Scaled(Of)) */
//                         int Of = AT_NORM(AT_CLIP_POS(OutBuff3[o] + *(short *)&Infos[LSTM_NE16_INT_B0], *((short *)&Infos[LSTM_NE16_INT_A0])) * *((short *)&Infos[LSTM_NE16_INT_C0]), ((unsigned char *)Infos)[LSTM_NE16_INT_Q]);
//                         int X1 = AT_SCALE(State[TileOff + o], ((unsigned char *)Infos)[LSTM_NE16_CIN_SCALE], ((unsigned char *)Infos)[LSTM_NE16_CIN_SCALEN]);
//                         // Finish compute of X1
//                         X1 = Of * X1 + OutBuff1[o];
//                         if (StateInOut)
//                                 StateInOut[TileOff + o] = gap_clipu(AT_SCALE(X1, ((unsigned char *)Infos)[LSTM_NE16_COUT_SCALE], ((unsigned char *)Infos)[LSTM_NE16_COUT_SCALEN]), 8);
//                         X1 = Max(-one, Min(one, X1));
//                         OutBuff1[o] = X1;
//                 }

//                 if (Trace)
//                         printf("Done f %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 if (Trace)
//                         printf("Start o %d\n", CoreId);
//                 for (int o = First; o < Last; o++)
//                 {
//                         /* Oo = HSigmoid(Scaled(Oo)) */
//                         int Oo = AT_NORM(AT_CLIP_POS(OutBuff2[o] + *((short *)&Infos[LSTM_NE16_INT_B0]), *((short *)&Infos[LSTM_NE16_INT_A0])) * *((short *)&Infos[LSTM_NE16_INT_C0]), ((unsigned char *)Infos)[LSTM_NE16_INT_Q]);
//                         int X2 = gap_clipu(AT_SCALE(Oo * OutBuff1[o], ((unsigned char *)Infos)[LSTM_NE16_OUT_SCALE], ((unsigned char *)Infos)[LSTM_NE16_OUT_SCALEN]), 8);
//                         if (StateInOut)
//                                 StateInOut[TileOff + DimState + o] = X2;
//                         if (Hout)
//                                 Hout[o] = X2;
//                 }
//                 if (Trace)
//                         printf("Done o %d\n", CoreId);
//         }
//         else
//         {
//                 if (Trace)
//                         printf("Master core %d\n", CoreId);
//                 if (!Xin)
//                         Nin -= DimIn;

//                 unsigned char *__restrict__ Wf = Arg->Wf;
//                 int *__restrict__ Bf = Arg->Bf;
//                 unsigned char *__restrict__ Wi = Arg->Wi;
//                 int *__restrict__ Bi = Arg->Bi;
//                 unsigned char *__restrict__ Wg = Arg->Wg;
//                 int *__restrict__ Bg = Arg->Bg;
//                 unsigned char *__restrict__ Wo = Arg->Wo;
//                 int *__restrict__ Bo = Arg->Bo;
//                 int Nb_KI, Rem_KI;
//                 int Nb_KO = Nout / 32 + (Nout % 32 ? 1 : 0);
//                 int Rem_KO = Nout % 32 ? Nout % 32 : 32; // Check different wrt simulator
//                 char FilterDataSizeBits = Arg->FilterDataSizeBits;
//                 unsigned char *Scale = Arg->ScaleNorm;

//                 unsigned int cfg = Arg->Default_NE16_Job_Cfg;

//                 int id_i = GetJobId();
//                 SetupNE16Job(cfg, State, OutBuff1, Wi, Bi, 1, Nin, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_I_ZEROPOINT]);
//                 if (Arg->FirstOut)
//                 {
//                         if (Trace)
//                                 printf("Master wait init state %d\n", CoreId);
//                         gap_waitbarrier(0);
//                 }
//                 if (Xin)
//                 {
//                         if (Trace)
//                                 printf("Master wait init input %d\n", CoreId);
//                         gap_waitbarrier(0);
//                 }

//                 NE16_SETPRIORITY_NE16(); // priority to NE16 w.r.t. cores, DMA
//                 if (Trace)
//                         printf("Master Queue i %d %d\n", CoreId, id_i);
//                 NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);

//                 int id_g = GetJobId();
//                 SetupNE16Job(cfg, State, OutBuff2, Wg, Bg, 1, Nin, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_G_ZEROPOINT]);
//                 if (Trace)
//                         printf("Master Queue g %d %d\n", CoreId, id_g);
//                 NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);

//                 int id_f = GetJobId();
//                 if (Trace)
//                         printf("Master Done i %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 SetupNE16Job(cfg, State, OutBuff3, Wf, Bf, 1, Nin, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_F_ZEROPOINT]);

//                 if (Trace)
//                         printf("Master Queue f %d %d\n", CoreId, id_f);
//                 NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);
//                 int id_o = GetJobId();
//                 if (Trace)
//                         printf("Master Done g %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 SetupNE16Job(cfg, State, OutBuff2, Wo, Bo, 1, Nin, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_O_ZEROPOINT]);

//                 if (Trace)
//                         printf("Master Queue o %d %d\n", CoreId, id_o);
//                 NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);

//                 // Wait for <= 1 job
//                 while (1)
//                 {
//                         int status = NE16_READ_STATUS();
//                         int job = NE16_READ_RUNNING_JOB();
//                         if (Trace)
//                                 printf("Status %d job %d\n", status, job);
//                         if (status == 0 || job != id_f)
//                                 break;
//                         NE16_BARRIER_NOSTATUS();
//                 }
//                 if (Trace)
//                         printf("Master Done f %d %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 // wait for end of computation
//                 NE16_BARRIER();

//                 if (Trace)
//                         printf("Master Done o %d %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 // set priority to core side
//                 NE16_SETPRIORITY_CORE();
//         }
//         if (Trace)
//                 printf("Final wait %d\n", CoreId);
//         gap_waitbarrier(0);
// }

// void LSTM_ParKerB32_SameInStateScale_NE16(KerLSTM_NE16_T *Arg)

// {
//         /*	Sequences
// 	 	In:	DimIn!=0, Hout==0
// 		InOut:	DimIn!=0, Hout!=0
// 		None:	DimIn==0, Hout==0
// 		Out:	DimIn==0, Hout!=0

// 		Infos:
// 			2 + 2 + 2 + 2 + 6 + 7  LSTM_INT group contains 3 shorts and 1 byte

// 		In total: 21	LSTM_CELL_INFOS
// 		if (PerChannelQuant) Infos group for each output elemt (Nout) else one group for all out
// 	*/
//         int Trace = 0;
//         unsigned char *__restrict__ StateInOut = Arg->StateInOut;
//         unsigned char *__restrict__ State = Arg->State;
//         unsigned char *__restrict__ Xin = Arg->Xin;
//         unsigned short int DimState = Arg->DimState;
//         unsigned short int DimIn = Arg->DimIn;
//         unsigned char *__restrict__ Hout = Arg->Hout;
//         unsigned short int Nout = Arg->Nout;
//         signed char *__restrict__ Infos = Arg->Infos;
//         int TileOff = Arg->TileOffset;
//         int *__restrict__ OutBuff1 = (int *)Arg->OutBuff1;
//         int *__restrict__ OutBuff2 = (int *)Arg->OutBuff2;
//         int *__restrict__ OutBuff3 = (int *)Arg->OutBuff3;

//         unsigned int Nin = DimState + DimIn;
//         unsigned int NS = Nin;
//         unsigned int CoreId = gap_coreid();
//         unsigned int ChunkCell = ChunkSize(Nout, 0);
//         unsigned int First = CoreId * ChunkCell;
//         unsigned int Last = Min(First + ChunkCell, Nout);
//         if (Trace)
//                 printf("Entry %d\n", CoreId);

//         if (CoreId != 8)
//         {
//                 if (Trace)
//                         printf("core %d\n", CoreId);
//                 if (Arg->FirstOut)
//                 {
//                         if (Trace)
//                                 printf("Init state %d\n", CoreId);
//                         if (Arg->FirstCell && Arg->Reset)
//                                 Zero(State, 2 * DimState, CoreId);
//                         else
//                                 Copy(State, StateInOut, 2 * DimState, CoreId);
//                         gap_waitbarrier(0);
//                 }
//                 if (Xin)
//                 {
//                         if (Trace)
//                                 printf("Copy Xin %d\n", CoreId);
//                         Copy(State + 2 * DimState, Xin, DimIn, CoreId);
//                         gap_waitbarrier(0);
//                 }

//                 if (Trace)
//                         printf("Done copy %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 if (Trace)
//                         printf("Start i %d\n", CoreId);
//                 for (int o = First; o < Last; o++)
//                 {
//                         /* Oi = HSigmoid(Scaled(Oi)) */
//                         OutBuff1[o] = Sigmoid(gap_clip(OutBuff1[o], 15));
//                 }

//                 if (Trace)
//                         printf("Done i %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 if (Trace)
//                         printf("Start g %d\n", CoreId);

//                 for (int o = First; o < Last; o++)
//                 {
//                         /* Og = HTanh(Scaled(Og)) */
//                         OutBuff1[o] *= Tanh(gap_clip(OutBuff2[o], 15));
//                 }

//                 if (Trace)
//                         printf("Done g %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 if (Trace)
//                         printf("Start f %d\n", CoreId);

//                 for (int o = First; o < Last; o++)
//                 {
//                         /* Of = HSigmoid(Scaled(Of)) */
//                         int Of = Sigmoid(gap_clip(OutBuff3[o], 15));
//                         int X1 = AT_SCALE(State[TileOff + o], ((unsigned char *)Infos)[LSTM_NE16_CIN_SCALE], ((unsigned char *)Infos)[LSTM_NE16_CIN_SCALEN]);
//                         // Finish compute of X1
//                         X1 = Of * X1 + OutBuff1[o];
//                         if (StateInOut)
//                                 StateInOut[TileOff + o] = gap_clipu(AT_SCALE(X1, ((unsigned char *)Infos)[LSTM_NE16_COUT_SCALE], ((unsigned char *)Infos)[LSTM_NE16_COUT_SCALEN]), 8);
//                         OutBuff1[o] = Tanh(gap_clip(X1, 15));
//                 }

//                 if (Trace)
//                         printf("Done f %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 if (Trace)
//                         printf("Start o %d\n", CoreId);
//                 for (int o = First; o < Last; o++)
//                 {
//                         /* Oo = HSigmoid(Scaled(Oo)) */
//                         int Oo = Sigmoid(gap_clip(OutBuff2[o], 15));
//                         int X2 = gap_clipu(AT_SCALE(Oo * OutBuff1[o], ((unsigned char *)Infos)[LSTM_NE16_OUT_SCALE], ((unsigned char *)Infos)[LSTM_NE16_OUT_SCALEN]), 8);
//                         if (StateInOut)
//                                 StateInOut[TileOff + DimState + o] = X2;
//                         if (Hout)
//                                 Hout[o] = X2;
//                 }
//                 if (Trace)
//                         printf("Done o %d\n", CoreId);
//         }
//         else
//         {
//                 if (Trace)
//                         printf("Master core %d\n", CoreId);
//                 if (!Xin)
//                         Nin -= DimIn;

//                 unsigned char *__restrict__ Wf = Arg->Wf;
//                 int *__restrict__ Bf = Arg->Bf;
//                 unsigned char *__restrict__ Wi = Arg->Wi;
//                 int *__restrict__ Bi = Arg->Bi;
//                 unsigned char *__restrict__ Wg = Arg->Wg;
//                 int *__restrict__ Bg = Arg->Bg;
//                 unsigned char *__restrict__ Wo = Arg->Wo;
//                 int *__restrict__ Bo = Arg->Bo;
//                 int Nb_KI, Rem_KI;
//                 int Nb_KO = Nout / 32 + (Nout % 32 ? 1 : 0);
//                 int Rem_KO = Nout % 32 ? Nout % 32 : 32; // Check different wrt simulator
//                 char FilterDataSizeBits = Arg->FilterDataSizeBits;
//                 signed char *Scale = Arg->ScaleNorm;

//                 unsigned int cfg = Arg->Default_NE16_Job_Cfg;

//                 int id_i = GetJobId();
//                 SetupNE16Job(cfg, State, OutBuff1, Wi, Bi, 1, Nin, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_I_ZEROPOINT]);
//                 Scale += 2 * Nout;
//                 if (Arg->FirstOut)
//                 {
//                         if (Trace)
//                                 printf("Master wait init state %d\n", CoreId);
//                         gap_waitbarrier(0);
//                 }
//                 if (Xin)
//                 {
//                         if (Trace)
//                                 printf("Master wait init input %d\n", CoreId);
//                         gap_waitbarrier(0);
//                 }

//                 NE16_SETPRIORITY_NE16(); // priority to NE16 w.r.t. cores, DMA
//                 if (Trace)
//                         printf("Master Queue i %d %d\n", CoreId, id_i);
//                 NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);

//                 int id_g = GetJobId();
//                 SetupNE16Job(cfg, State, OutBuff2, Wg, Bg, 1, Nin, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_G_ZEROPOINT]);
//                 Scale += 2 * Nout;
//                 if (Trace)
//                         printf("Master Queue g %d %d\n", CoreId, id_g);
//                 NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);

//                 int id_f = GetJobId();
//                 if (Trace)
//                         printf("Master Done i %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 SetupNE16Job(cfg, State, OutBuff3, Wf, Bf, 1, Nin, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_F_ZEROPOINT]);
//                 Scale += 2 * Nout;

//                 if (Trace)
//                         printf("Master Queue f %d %d\n", CoreId, id_f);
//                 NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);
//                 int id_o = GetJobId();
//                 if (Trace)
//                         printf("Master Done g %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 SetupNE16Job(cfg, State, OutBuff2, Wo, Bo, 1, Nin, Nout, Nb_KO, Rem_KO, 1, 1, FilterDataSizeBits, Scale, &Scale[Nout], Infos[LSTM_NE16_O_ZEROPOINT]);

//                 if (Trace)
//                         printf("Master Queue o %d %d\n", CoreId, id_o);
//                 NE16_WRITE_CMD(NE16_COMMIT_AND_TRIGGER, NE16_TRIGGER_CMD);

//                 // Wait for <= 1 job
//                 while (1)
//                 {
//                         int status = NE16_READ_STATUS();
//                         int job = NE16_READ_RUNNING_JOB();
//                         if (Trace)
//                                 printf("Status %d job %d\n", status, job);
//                         if (status == 0 || job != id_f)
//                                 break;
//                         NE16_BARRIER_NOSTATUS();
//                 }
//                 if (Trace)
//                         printf("Master Done f %d %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 // wait for end of computation
//                 NE16_BARRIER();

//                 if (Trace)
//                         printf("Master Done o %d %d\n", CoreId);
//                 gap_waitbarrier(0);
//                 // set priority to core side
//                 NE16_SETPRIORITY_CORE();
//         }
//         if (Trace)
//                 printf("Final wait %d\n", CoreId);
//         gap_waitbarrier(0);
// }
