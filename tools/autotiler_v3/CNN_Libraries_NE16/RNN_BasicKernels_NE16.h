/*
 * Copyright (C) 2018 GreenWaves Technologies
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

#ifndef __RNN_BASICKERNELS_NE16__
#define __RNN_BASICKERNELS_NE16__
#include "Gap.h"
#include "hal_ne16.h"
#include "../CNN_Libraries/CNN_Defines.h"
#include "../CNN_Libraries_SQ8/CNN_BasicKernels_SQ8.h"
#include "./CNN_BasicKernels_NE16.h"
#include "./stage_desc.h"


/******************************************************************************************************************
         Recursive NN (RNN, LSTM, GRU)
******************************************************************************************************************/
#define RNN_NE16_SIGMOID_TABLE      256 * 2

#define RNN_NE16_F_INF              9
#define RNN_NE16_F_OFF              0
#define RNN_NE16_F_A0               (0 + RNN_NE16_F_OFF)
#define RNN_NE16_F_B0               (1 + RNN_NE16_F_OFF)
#define RNN_NE16_W_ZERO_POINT       (2 + RNN_NE16_F_OFF)
#define RNN_NE16_OUT_ZERO_POINT     (3 + RNN_NE16_F_OFF)
#define RNN_NE16_OUT_SCALE          (5 + RNN_NE16_F_OFF)
#define RNN_NE16_OUT_SCALEN         (6 + RNN_NE16_F_OFF)
#define RNN_NE16_I_PRE_N            (7 + RNN_NE16_F_OFF)
#define RNN_NE16_R_PRE_N            (8 + RNN_NE16_F_OFF)

#define RNN_NE16_CELL_INFOS (RNN_NE16_F_INF + RNN_NE16_F_OFF)

typedef struct {
        unsigned char *__restrict__ StateInOut;	/**< Pointer to In/Out state, Dim=DimState   */
        unsigned char *__restrict__ State;	/**< Pointer to to a copy of state with extra space for in (DimState+DimIn)   */
        unsigned char *__restrict__ Xin;	/**< Pointer to In state, Dim=DimIn */
        unsigned char *__restrict__ ScaleNorm;  /**< Pointer to ScaleNorm - If separate In/State scaling then 4 x State otherwise 2 x State */
        void *__restrict__ OutBuff;
        unsigned short int DimState;		/**< State dimension */
        unsigned short int DimIn;		/**< Input dimension */
        unsigned short int DimStateInt;		/**< State dimension - padded % 16 */
        unsigned short int DimInInt;		/**< Input dimension - padded % 16 */
        unsigned char *__restrict__ Wf;		/**< Pointer to Forget gate (whether to erase cell) weights, Dim=[DimState+DimIn,DimState] or [DimState,DimState] */
        unsigned char *__restrict__ Wfi;	/**< Pointer to Forget gate (whether to erase cell) weights, Dim=[DimIn,DimState] Not used if same in / out scale*/
        int * __restrict__ Bf;			/**< Pointer to Forget gate bias */
        unsigned char *__restrict__ Hout;		/**< Pointer to Hout in case sequence must be exposed, null otherwise */
        unsigned short int Nout;		/**< Number of output produced in StateInOut */
        signed char *__restrict__ Infos;	/**< Infos vector for scaling */
        char FirstOut;				/**< 1 if first out of one cell to eval */
        char FirstCell;				/**< 1 if first cell of a group of cell */
        char FilterDataSizeBits;		/**< Qw */
        int Default_NE16_Job_Cfg;	        /**< NE16 job config */
        char Reset;				/**< 1 if RNN State has to be reset */
        int TileOffset;				/**< Buffer Offset related to the current Tile index */
} KerRNN_NE16_T;

typedef struct {
        unsigned short *__restrict__ StateInOut;	/**< Pointer to In/Out state, Dim=DimState   */
        unsigned short *__restrict__ State;	/**< Pointer to to a copy of state with extra space for in (DimState+DimIn)   */
        unsigned short *__restrict__ Xin;	/**< Pointer to In state, Dim=DimIn */
        unsigned char *__restrict__ ScaleNorm;  /**< Pointer to ScaleNorm - If separate In/State scaling then 4 x State otherwise 2 x State */
        void *__restrict__ OutBuff;
        unsigned short int DimState;		/**< State dimension */
        unsigned short int DimIn;		/**< Input dimension */
        unsigned short int DimStateInt;		/**< State dimension - padded % 16 */
        unsigned short int DimInInt;		/**< Input dimension - padded % 16 */
        unsigned char *__restrict__ Wf;		/**< Pointer to Forget gate (whether to erase cell) weights, Dim=[DimState+DimIn,DimState] or [DimState,DimState] */
        unsigned char *__restrict__ Wfi;	/**< Pointer to Forget gate (whether to erase cell) weights, Dim=[DimIn,DimState] Not used if same in / out scale*/
        int * __restrict__ Bf;			/**< Pointer to Forget gate bias */
        unsigned short *__restrict__ Hout;	/**< Pointer to Hout in case sequence must be exposed, null otherwise */
        unsigned short int Nout;		/**< Number of output produced in StateInOut */
        signed char *__restrict__ Infos;	/**< Infos vector for scaling */
        char FirstCell;				/**< 1 if first cell of a group of cell */
        char FirstOut;				/**< 1 if first out of one cell to eval */
        char FilterDataSizeBits;		/**< Qw */
        int Default_NE16_Job_Cfg;	        /**< NE16 job config */
        char Reset;				/**< 1 if RNN State has to be reset */
        int TileOffset;				/**< Buffer Offset related to the current Tile index */
} KerRNN_NE16fp_T;

#define LSTM_NE16_W_INF         2
#define LSTM_NE16_W_OFF         RNN_NE16_SIGMOID_TABLE
#define LSTM_NE16_W_ZEROPOINT   (0 + LSTM_NE16_W_OFF)
#define LSTM_NE16_GATE_PRENORM  (1 + LSTM_NE16_W_OFF)

#define LSTM_NE16_OUT_INF      8 // LSTM_NE16_OUT_ZEROPOINT is 2 bytes
#define LSTM_NE16_OUT_OFF      (LSTM_NE16_W_OFF+LSTM_NE16_W_INF)
#define LSTM_NE16_CIN_SCALE     (0 + LSTM_NE16_OUT_OFF)
#define LSTM_NE16_CIN_SCALEN    (1 + LSTM_NE16_OUT_OFF)
#define LSTM_NE16_COUT_SCALE    (2 + LSTM_NE16_OUT_OFF)
#define LSTM_NE16_COUT_SCALEN   (3 + LSTM_NE16_OUT_OFF)
#define LSTM_NE16_OUT_SCALE     (4 + LSTM_NE16_OUT_OFF)
#define LSTM_NE16_OUT_SCALEN    (5 + LSTM_NE16_OUT_OFF)
#define LSTM_NE16_OUT_ZEROPOINT (6 + LSTM_NE16_OUT_OFF)

#define LSTM_NE16_INT_INF       3
#define LSTM_NE16_INT_OFF       (LSTM_NE16_OUT_OFF+LSTM_NE16_COUT_INF)
#define LSTM_NE16_INT_A0        (0 + LSTM_NE16_INT_OFF)
#define LSTM_NE16_INT_B0        (1 + LSTM_NE16_INT_OFF)
#define LSTM_NE16_INT_C0        (2 + LSTM_NE16_INT_OFF)

#define LSTM_NE16_CELL_INFOS         (LSTM_NE16_W_INF+LSTM_NE16_OUT_INF+LSTM_NE16_INT_INF+RNN_NE16_SIGMOID_TABLE)
#define LSTM_NE16_CELL_INFOS_SHORT   (LSTM_NE16_W_INF+LSTM_NE16_OUT_INF+RNN_NE16_SIGMOID_TABLE)

typedef struct {
        unsigned char *__restrict__ StateInOut;	/**< Pointer to In/Out state, Dim=DimState   */
        unsigned char *__restrict__ State;	/**< Pointer to to a copy of state with extra space for in (DimState+DimStateInt+DimInInt)   */
        unsigned char *__restrict__ Xin;		/**< Pointer to In state, Dim=DimIn */
        unsigned char *__restrict__ ScaleNorm;    /**< Pointer to Scale and Norm - If separate In/State scaling then 2 x State * 8 otherwise State * 8 */
        int *__restrict__ OutBuff1;
        int *__restrict__ OutBuff2;
        int *__restrict__ OutBuff3;
        unsigned short int DimState;		/**< State dimension */
        unsigned short int DimIn;		/**< Input dimension */
        unsigned short int DimStateInt;		/**< State dimension - padded % 16 */
        unsigned short int DimInInt;		/**< Input dimension - padded % 16 */
        unsigned char *__restrict__ Wf;		/**< Pointer to Forget gate (whether to erase cell) weights, Dim=[DimState+DimIn,DimState] */
        unsigned char *__restrict__ Wfi;		/**< Pointer to Forget gate (whether to erase cell) weights, Dim=[DimState+DimIn,DimState] */
        int * __restrict__ Bf;			/**< Pointer to Forget gate bias */
        unsigned char *__restrict__ Wi;		/**< Pointer to Input gate (whether to write cell) weights, Dim=[DimState+DimIn,DimState] */
        unsigned char *__restrict__ Wii;		/**< Pointer to Input gate (whether to write cell) weights, Dim=[DimState+DimIn,DimState] */
        int * __restrict__ Bi;			/**< Pointer to Input gate bias */
        unsigned char *__restrict__ Wg;		/**< Pointer to Gate gate (how much to write to cell) weights, Dim=[DimState+DimIn,DimState] */
        unsigned char *__restrict__ Wgi;		/**< Pointer to Gate gate (how much to write to cell) weights, Dim=[DimState+DimIn,DimState] */
        int * __restrict__ Bg;			/**< Pointer to Gate gate bias */
        unsigned char *__restrict__ Wo;		/**< Pointer to Output gate (how much to reveal cell) weights, Dim=[DimState+DimIn,DimState] */
        unsigned char *__restrict__ Woi;		/**< Pointer to Output gate (how much to reveal cell) weights, Dim=[DimState+DimIn,DimState] */
        int * __restrict__ Bo;			/**< Pointer to Output gate bias */
        unsigned char *__restrict__ Hout;		/**< Pointer to Hout in case sequence must be exposed, null otherwise */
        unsigned short int Nout;		/**< Number of output produced in StateInOut and CInOut */
        signed char *__restrict__ Infos;	/**< Infos vector for scaling */
        char FirstOut;				/**< 1 if first out of one cell to eval */
        char FirstCell;				/**< 1 if first cell of a group of cell */
        char LastOut;				/**< 1 if last out of one cell to eval */
        char LastCell;				/**< 1 if last cell of a group of cell */
        char FilterDataSizeBits;		/**< Qw */
        int Default_NE16_Job_Cfg;	        /**< NE16 job config */
        char Reset;				/**< If 1 LSTM State is reset */
        int TileOffset;				/**< Buffer Offset related to the current Tile index */
        pStageDesc_t pStageDesc;
} KerLSTM_NE16_T;

typedef struct {
        unsigned short *__restrict__ StateInOut;	/**< Pointer to In/Out state, Dim=DimState   */
        unsigned short *__restrict__ State;	/**< Pointer to to a copy of state with extra space for in (DimState+DimStateInt+DimInInt)   */
        unsigned short *__restrict__ Xin;		/**< Pointer to In state, Dim=DimIn */
        unsigned char *__restrict__ ScaleNorm;    /**< Pointer to Scale and Norm - If separate In/State scaling then 2 x State * 8 otherwise State * 8 */
        int *__restrict__ OutBuff1;
        int *__restrict__ OutBuff2;
        int *__restrict__ OutBuff3;
        unsigned short int DimState;		/**< State dimension */
        unsigned short int DimIn;		/**< Input dimension */
        unsigned short int DimStateInt;		/**< State dimension - padded % 16 */
        unsigned short int DimInInt;		/**< Input dimension - padded % 16 */
        unsigned char *__restrict__ Wf;		/**< Pointer to Forget gate (whether to erase cell) weights, Dim=[DimState+DimIn,DimState] */
        unsigned char *__restrict__ Wfi;		/**< Pointer to Forget gate (whether to erase cell) weights, Dim=[DimState+DimIn,DimState] */
        int * __restrict__ Bf;			/**< Pointer to Forget gate bias */
        unsigned char *__restrict__ Wi;		/**< Pointer to Input gate (whether to write cell) weights, Dim=[DimState+DimIn,DimState] */
        unsigned char *__restrict__ Wii;		/**< Pointer to Input gate (whether to write cell) weights, Dim=[DimState+DimIn,DimState] */
        int * __restrict__ Bi;			/**< Pointer to Input gate bias */
        unsigned char *__restrict__ Wg;		/**< Pointer to Gate gate (how much to write to cell) weights, Dim=[DimState+DimIn,DimState] */
        unsigned char *__restrict__ Wgi;		/**< Pointer to Gate gate (how much to write to cell) weights, Dim=[DimState+DimIn,DimState] */
        int * __restrict__ Bg;			/**< Pointer to Gate gate bias */
        unsigned char *__restrict__ Wo;		/**< Pointer to Output gate (how much to reveal cell) weights, Dim=[DimState+DimIn,DimState] */
        unsigned char *__restrict__ Woi;		/**< Pointer to Output gate (how much to reveal cell) weights, Dim=[DimState+DimIn,DimState] */
        int * __restrict__ Bo;			/**< Pointer to Output gate bias */
        unsigned short *__restrict__ Hout;		/**< Pointer to Hout in case sequence must be exposed, null otherwise */
        unsigned short int Nout;		/**< Number of output produced in StateInOut and CInOut */
        signed char *__restrict__ Infos;	/**< Infos vector for scaling */
        char FirstOut;				/**< 1 if first out of one cell to eval */
        char FirstCell;				/**< 1 if first cell of a group of cell */
        char LastOut;				/**< 1 if last out of one cell to eval */
        char LastCell;				/**< 1 if last cell of a group of cell */
        char FilterDataSizeBits;		/**< Qw */
        int Default_NE16_Job_Cfg;	        /**< NE16 job config */
        char Reset;				/**< If 1 LSTM State is reset */
        int TileOffset;				/**< Buffer Offset related to the current Tile index */
        pStageDesc_t pStageDesc;
} KerLSTM_NE16fp_T;



#define GRU_R_INF              4
#define GRU_R_OFF              0
#define GRU_R_INT_SCALE        0
#define GRU_R_INT_SCALEN       1
#define GRU_R_IN_SCALE         2
#define GRU_R_IN_SCALEN        3

#define GRU_Z_INF              4
#define GRU_Z_OFF              (GRU_R_OFF+GRU_R_INF)
#define GRU_Z_INT_SCALE        (0 + GRU_Z_OFF)
#define GRU_Z_INT_SCALEN       (1 + GRU_Z_OFF)
#define GRU_Z_IN_SCALE         (2 + GRU_Z_OFF)
#define GRU_Z_IN_SCALEN        (3 + GRU_Z_OFF)

#define GRU_HT_INF              2
#define GRU_HT_OFF              (GRU_Z_OFF+GRU_Z_INF)
#define GRU_HT_IN_SCALE         (0 + GRU_HT_OFF)
#define GRU_HT_IN_SCALEN        (1 + GRU_HT_OFF)

#define GRU_H_INF				2
#define GRU_H_OFF				(GRU_HT_OFF+GRU_HT_INF)
#define GRU_H_INT_SCALE			(0 + GRU_H_OFF)
#define GRU_H_INT_SCALEN		(1 + GRU_H_OFF)

#define GRU_INT_INF             7
#define GRU_INT_OFF             (GRU_H_OFF+GRU_H_INF)
#define GRU_INT_A0              (0 + GRU_INT_OFF)
#define GRU_INT_B0              (2 + GRU_INT_OFF)
#define GRU_INT_C0              (4 + GRU_INT_OFF)
#define GRU_INT_Q               (6 + GRU_INT_OFF)

#define GRU_CELL_INFOS (GRU_R_INF+GRU_Z_INF+GRU_HT_INF+GRU_H_INF+GRU_INT_INF)

typedef struct {
        signed char *__restrict__ StateInOut;	/**< Pointer to In/Out state, Dim=DimState   */
        signed char *__restrict__ Xin;		/**< Pointer to In state, Dim=DimIn */
        signed char *__restrict__ State;	/**< Pointer to to a copy of state with extra space for in (DimState+DimIn)   */
        unsigned short int DimState;		/**< State dimension */
        unsigned short int DimIn;		/**< Input dimension */
        signed char *__restrict__ Wr;		/**< Pointer to R gate weights, Dim=[DimState+DimIn,DimState] */
        void * __restrict__ Br;			/**< Pointer to R gate bias */
        signed char *__restrict__ Wz;		/**< Pointer to Z gate weights, Dim=[DimState+DimIn,DimState] */
        void * __restrict__ Bz;			/**< Pointer to Z gate bias */
        signed char *__restrict__ Wh;		/**< Pointer to H gate weights, Dim=[DimState+DimIn,DimState] */
        void * __restrict__ Bwh;		/**< Pointer to H gate bias vs Inputs */
        void * __restrict__ Brh;		/**< Pointer to H gate bias vs States */
        signed char *__restrict__ Hout;		/**< Pointer to Hout in case sequence must be exposed, null otherwise */
        unsigned short int Nout;		/**< Number of output produced in StateInOut */
        unsigned short int OutBase;		/**< Index of first output produced in StateInOut */
        signed char *__restrict__ Infos;	/**< Infos vector for scaling */
        char FirstOut;				/**< 1 if first out of one cell to eval */
        char FirstCell;				/**< 1 if first cell of a group of cell */
        char Reset;				/**< If 1 GRU State is reset */
        int TileOffset;				/**< Buffer Offset related to the current Tile index */
} KerGRU_NE16_T;

void RNN_ParKerB32_Hard_SameInStateScale_NE16(KerRNN_NE16_T *Arg);
void RNN_ParKerB32_Hard_NE16(KerRNN_NE16_T *Arg);
void RNN_ParKerB32_SameInStateScale_NE16(KerRNN_NE16_T *Arg);
void RNN_ParKerB32_NE16(KerRNN_NE16_T *Arg);
void RNN_ParKerB32_NE16fp(KerRNN_NE16fp_T *Arg);

void LSTM_ParKerB32_Hard_SameInStateScale_NE16(KerLSTM_NE16_T *Arg);
void LSTM_ParKerB32_SameInStateScale_NE16(KerLSTM_NE16_T *Arg);
void LSTM_ParKerB32_NE16(KerLSTM_NE16_T *Arg);
void LSTM_ParKerB32_NE16fp(KerLSTM_NE16fp_T *Arg);

#endif