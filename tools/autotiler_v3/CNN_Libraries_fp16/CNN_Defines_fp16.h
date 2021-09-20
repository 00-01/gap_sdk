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

#ifndef __CNN_DEFINED_FP16_H__
#define __CNN_DEFINED_FP16_H__

#include "../DSP_Libraries/FloatDefines.h"

#ifdef __pulp__
	#define Abs(a)          __builtin_pulp_abs((a))
	#define Min(a, b)       __builtin_pulp_minsi((a), (b))
	#define Max(a, b)       __builtin_pulp_maxsi((a), (b))
#else
	#define Abs(a)          (((int)(a)<0)?(-(a)):(a))
	#define Min(a, b)       (((a)<(b))?(a):(b))
	#define Max(a, b)       (((a)>(b))?(a):(b))
#endif

#ifdef __gap9__
	#ifdef STD_FLOAT
		#define AbsF(a)		Absf16((a))
		#define MaxF(a, b)	Maxf16((a), (b))
		#define MinF(a, b)	Minf16((a), (b))

		#define AbsF2(a)	Absv2h((a))
		#define MaxF2(a, b)	Maxv2h((a), (b))
		#define MinF2(a, b)	Minv2h((a), (b))

		#define MIN_FLT16	MIN_f16
		#define MAX_FLT16	MAX_f16
		#define	LEAK_CONSTANT	((float16)0.1)
	#else
		#define AbsF(a)		Absf16a((a))
		#define MaxF(a, b)	Maxf16a((a), (b))
		#define MinF(a, b)	Maxf16a((a), (b))

		#define AbsF2(a)	((F16V) Absv2ah((a)))
		#define MaxF2(a, b)	((F16V) Maxv2ah((a), (b)))
		#define MinF2(a, b)	((F16V) Minv2ah((a), (b)))

		#define MIN_FLT16	MIN_f16a
		#define MAX_FLT16	MAX_f16a
		#define	LEAK_CONSTANT	((float16alt)0.1)
	#endif // STD_FLOAT
#else
    #define AbsF(a)         (((a)<0.0f)?(-(a)):(a))
    #define MinF Min
    #define MaxF Max
    #define AbsF32(a)         (((a)<0.0f)?(-(a)):(a))
    #define MinF32 Min
    #define MaxF32 Max
    #define AbsF2(a)	((F16V) {((float)(a)[0]<0.0)?(-(float)(a)[0]):((float)(a)[0]), \
        ((float)(a)[1]<0.0)?(-(float)(a)[1]):((float)(a)[1])})
    #define MaxF2(a, b)	((F16V) {((float)(a)[0]<(float)(b)[0])?((float)(a)[0]):((float)(b)[0]), \
        ((float)(a)[1]<(float)(b)[1])?((float)(a)[1]):((float)(b)[1])})
    #define MinF2(a, b)	((F16V) {((float)(a)[0]>(float)(b)[0])?((float)(a)[0]):((float)(b)[0]), \
        ((float)(a)[1]>(float)(b)[1])?((float)(a)[1]):((float)(b)[1])})

    #define MIN_FLT16	((float)(1.1754943508e-38f))
    #define MAX_FLT16	((float)(3.4028234664e38f))

	#define	LEAK_CONSTANT	((float)0.1)
#endif

#define Minu(a, b)            (( ((unsigned int)a)<((unsigned int)b) )?((unsigned int)a):((unsigned int)b) )

/* In the following n is the bound and x the value to be clamped */
/* R = Max(0, Min(x, n) */
#define AT_CLIP_POS(x, n)       gap_clipur((x), (n))

/* R = Max(0, Min(x, 2^(n-1)-1 */
#define AT_CLIP_POS_IMM(x, n)   gap_clipu((x), (n))

/* R = Max(n, Min(x, -(n+1)) */
#define AT_CLIP_(x, n)          gap_clipr((x), (n))


#endif
