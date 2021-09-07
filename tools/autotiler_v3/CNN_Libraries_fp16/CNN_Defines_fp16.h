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

#ifdef __pulp__
#define Abs(a)          __builtin_pulp_abs((a))
#define Min(a, b)       __builtin_pulp_minsi((a), (b))
#define Max(a, b)       __builtin_pulp_maxsi((a), (b))

#ifdef STD_FLOAT
#define AbsF(a)		__builtin_pulp_f16abs((a))
#define MaxF(a, b)	__builtin_pulp_f16max((a), (b))
#define MinF(a, b)	__builtin_pulp_f16min((a), (b))

#define AbsF2(a)	__builtin_pulp_f16abs2((a))
#define MaxF2(a, b)	__builtin_pulp_f16max2((a), (b))
#define MinF2(a, b)	__builtin_pulp_f16min2((a), (b))

#define MIN_FLT16	((float16)(6.10e-5f))
#define MAX_FLT16	((float16)(65504))
#define	LEAK_CONSTANT	((float16)0.1)
#else
#define AbsF(a)		__builtin_pulp_f16altabs((a))
#define MaxF(a, b)	__builtin_pulp_f16altmax((a), (b))
#define MinF(a, b)	__builtin_pulp_f16altmin((a), (b))

#define AbsF2(a)	((F16V) __builtin_pulp_f16altabs2((a)))
#define MaxF2(a, b)	((F16V) __builtin_pulp_f16altmax2((a), (b)))
#define MinF2(a, b)	((F16V) __builtin_pulp_f16altmin2((a), (b)))

#define MIN_FLT16	((float16alt)(1.1754943508e-38f))
#define MAX_FLT16	((float16alt)(3.4028234664e38f))
#define	LEAK_CONSTANT	((float16alt)0.1)

#endif

#else
#define Abs(a)          (((int)(a)<0)?(-(a)):(a))
#define Min(a, b)       (((a)<(b))?(a):(b))
#define Max(a, b)       (((a)>(b))?(a):(b))
#define MinF Min
#define MaxF Max
#define AbsF(a)         (((a)<0.0f)?(-(a)):(a))
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
