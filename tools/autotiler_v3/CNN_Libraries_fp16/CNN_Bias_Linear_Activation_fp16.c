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

#include "Gap.h"
#include "CNN_BasicKernels_fp16.h"

static int CoreCountDynamic = 1;
static int ActiveCore = gap_ncore();

static inline unsigned int __attribute__((always_inline)) ChunkSize(unsigned int X)

{
	unsigned int NCore;
	unsigned int Log2Core;
	unsigned int Chunk;

	if (CoreCountDynamic) NCore = ActiveCore; else NCore = gap_ncore();
	Log2Core = gap_fl1(NCore);
	Chunk = (X>>Log2Core) + ((X&(NCore-1))!=0);
	return Chunk;
}

#define APPROX_LUT_RNN

#ifdef ACCURATE_MATH_RNN

	#include "float_math_funcs.h"
	F16 Fast_Sigmoid_fp16(F16 x)
	{
		return (F16) fsigmoid( (float32_t) x);
	}

	#include "math.h"
	F16 Fast_Tanh_fp16(F16 x)
	{
		return (F16) tanh((double) x);
	}

#else

#ifdef APPROX_LUT_RNN
	// LUTs are valid for both implementations
	static F16 SIGMOID_LUT_f16[256] = {
	      0.500000, 0.5078118642792044, 0.5156199157230156, 0.523420348936324, 0.5312093733737563, 
	      0.5389832206876841, 0.5467381519846138, 0.5544704649604273, 0.5621765008857981, 0.5698526514141571, 
	      0.5774953651858118, 0.5851011542032312, 0.5926665999540697, 0.600188359260205, 0.6076631698328917, 
	      0.6150878555160665, 0.6224593312018546, 0.6297746074044134, 0.6370307944803831, 0.6442251064863697, 
	      0.6513548646660542, 0.658417500561683, 0.665410558746814, 0.672331699179286, 0.679178699175393, 
	      0.6859494550081925, 0.6926419831347361, 0.6992544210587585, 0.7057850278370112, 0.712232184238947, 
	      0.7185943925708561, 0.7248702761768248, 0.7310585786300049, 0.7371581626286834, 0.7431680086124811, 
	      0.7490872131147275, 0.7549149868676283, 0.7606506526772884, 0.7662936430859597, 0.7718434978390747, 
	      0.7772998611746911, 0.7826624789529376, 0.7879311956428947, 0.7931059511841119, 0.7981867777396212,
	      0.8031737963569016, 0.8080672135527632, 0.8128673178375735, 0.8175744761936437, 0.8221891305219503, 
	      0.8267117940706734, 0.8311430478583168, 0.8354835371034369, 0.8397339676722393, 0.8438951025545426, 
	      0.8479677583778257, 0.8519528019683106, 0.8558511469672594, 0.8596637505099167, 0.8633916099737806, 
	      0.8670357598021706, 0.870597268408361, 0.8740772351648677, 0.8774767874818407, 0.8807970779778823, 
	      0.8840392817460332, 0.8872045937171068, 0.8902942261220291, 0.8933094060543487, 0.8962513731336202, 
	      0.8991213772699436, 0.90192067652954, 0.9046505351008906, 0.9073122213606241, 0.9099070060380482, 
	      0.9124361604769414, 0.9149009549929797, 0.9173026573249612, 0.9196425311777929, 0.9219218348550491, 
	      0.9241418199787566, 0.9263037302939515, 0.9284088005554476, 0.9304582554941719, 0.9324533088603709, 
	      0.9343951625409306, 0.9362850057480346, 0.9381240142763573, 0.9399133498259924, 0.941654159388319, 
	      0.9433475746920261, 0.9449947117065459, 0.9465966702001757, 0.94815453335022, 0.9496693674025232, 
	      0.9511422213778251, 0.9525741268224334, 0.9539660976007597, 0.9553191297273498, 0.9566342012360932, 
	      0.9579122720843811, 0.9591542840900507, 0.96036116089903, 0.9615338079816756, 0.9626731126558706, 
	      0.9637799441350253, 0.9648551535992067, 0.9658995742876885, 0.9669140216112958, 0.9678992932829918, 
	      0.9688561694652216, 0.9697854129326061, 0.9706877692486436, 0.9715639669551474, 0.9724147177732099, 
	      0.9732407168145568, 0.9740426428022031, 0.9748211582993981, 0.9755769099458929, 0.9763105287006256, 
	      0.9770226300899744, 0.9777138144607745, 0.9783846672373524, 0.9790357591818724, 0.9796676466573412, 
	      0.9802808718926534, 0.9808759632491112, 0.9814534354878828, 0.9820137900379085, 0.9825575152637948, 
	      0.9830850867332734, 0.983596967483837, 0.9840936082881853, 0.9845754479181588, 0.98504291340685, 
	      0.985496420308618, 0.9859363729567544, 0.9863631647185701, 0.9867771782476948, 0.9871787857334058, 
	      0.9875683491468141, 0.9879462204837635, 0.9883127420043056, 0.9886682464686385, 0.9890130573694068, 
	      0.9893474891602738, 0.9896718474806949, 0.9899864293768268, 0.9902915235185259, 0.9905874104123901, 
	      0.9908743626108194, 0.9911526449170697, 0.991422514586288, 0.9916842215225259, 0.9919380084717284, 
	      0.9921841112107114, 0.9924227587321393, 0.9926541734255229, 0.9928785712542649, 0.993096161928778, 
	      0.9933071490757153, 0.9935117304033434, 0.9937100978631075, 0.9939024378074247, 0.9940889311437562, 
	      0.994269753485008, 0.994445075296309, 0.9946150620382221, 0.9947798743064417, 0.9949396679680339, 
	      0.9950945942942779, 0.9952448000901669, 0.9953904278206259, 0.9955316157335089, 0.9956684979794361, 
	      0.9958012047285281, 0.995929862284104, 0.9960545931933981, 0.9961755163553627, 0.9962927471256138, 
	      0.9964063974185798, 0.9965165758069192, 0.9966233876182606, 0.9967269350293284, 0.9968273171575148, 
	      0.9969246301499518, 0.9970189672701452, 0.9971104189822263, 0.9971990730328789, 0.9972850145309949, 
	      0.9973683260251155, 0.9974490875787134, 0.9975273768433653, 0.997603269129871, 0.9976768374773688, 
	      0.997748152720497, 0.9978172835546547, 0.9978842965994068, 0.9979492564600826, 0.998012225787616, 
	      0.9980732653366725, 0.9981324340221067, 0.9981897889737974, 0.9982453855899006, 0.9982992775885648, 
	      0.9983515170581486, 0.9984021545059827, 0.998451238905712, 0.998498817743263, 0.9985449370614672, 
	      0.9985896415033819, 0.9986329743543437, 0.9986749775827868, 0.9987156918798661, 0.9987551566979116, 
	      0.9987934102877537, 0.9988304897349445, 0.9988664309949126, 0.9989012689270753, 0.9989350373279441, 
	      0.9989677689632452, 0.99899949559909, 0.9990302480322174, 0.9990600561193372, 0.9990889488055994, 
	      0.9991169541522155, 0.9991440993632549, 0.9991704108116409, 0.9991959140643708, 0.9992206339069791, 
	      0.9992445943672705, 0.9992678187383391, 0.9992903296008995, 0.9993121488449448, 0.9993332976907565, 
	      0.9993537967092804, 0.9993736658418905, 0.9993929244195585, 0.9994115911814431, 0.9994296842929216, 
	      0.9994472213630764, 0.9994642194616526, 0.999480695135505, 0.9994966644245472, 0.9995121428772178, 
	      0.9995271455654797, 0.999541687099365, 0.9995557816410785, 0.9995694429186754, 0.9995826842393223, 
	      0.9995955185021579, 0.9996079582107623, 0.999620015485248, 0.999631702073984, 0.9996430293649623, 
	      0.9996540083968218   
	};

#ifdef STD_FLOAT
	static inline int f16_to_Q12(F16 x){
	    int result;

	    short * x_b = &x;
	    short value = *x_b;
	    int sign = value & 0x8000;
	    int exp = (value >> 10) & 0x001F;
	    int significand = value & 0x07FF;
	    significand |= 0x0400;
	    exp = exp - 15;

	    int shift = 12 - 10 + exp;
	    if (shift > 0)
	        result = significand << shift;
	    else
	        result = significand >> (-shift);
	    
	    if(sign)
	        result = - result;

	    return result;
	}
#else
	static inline int f16_to_Q12(F16 x){
	    int result;

	    short * x_b = &x;
	    short value = *x_b;
	    int sign = value & 0x8000;
	    int exp = (value >> 7) & 0x00FF;
	    int significand = value & 0x007F;
	    significand |= 0x0080;
	    exp = exp - 127;

	    int shift = 12 - 7 + exp;
	    if (shift > 0)
	        result = significand << shift;
	    else
	        result = significand >> (-shift);
	    
	    if(sign)
	        result = - result;

	    return result;
	}
#endif
		
	F16 Fast_Sigmoid_fp16(F16 x_f16){
	    F16 result;

	    int x = f16_to_Q12(x_f16);
	    int abs_x = (Abs(x) ) >> 7;
	     if (abs_x > 255) {
	        result = (F16) 1.0; // result = 1 in Q16
	    } else {
	        result = SIGMOID_LUT_f16[abs_x]; // LUT in Q16
	    }
	    if (x>0) return result;
	    else     return ((F16) 1.0) - result;
	}

	F16 Fast_Tanh_fp16(F16 x_f16){
	    F16 result;

	    int x = f16_to_Q12(x_f16);
	    int abs_x = (Abs(x)) >> 6; // clip to Q5 to [0:255] (>>7) to be an index of the LUT

	    if (abs_x > 255) {
	        result = (F16) 1.0; // result = 1 in Q16
	    } else {
	        result = SIGMOID_LUT_f16[abs_x]; // LUT in Q16
	    }
	    if (x<0) return -((F16) 2.0)*result + ((F16) 1.0);
	    else 	 return  ((F16) 2.0)*result - ((F16) 1.0);
	}

#else

	F16 Fast_Sigmoid_fp16(F16 x)
	{
		return ((F16)0.5f * (x / ((F16)1.0f + AbsF(x))) + (F16)0.5f);
	}

	F16 Fast_Tanh_fp16(F16 x)
	{
        return (((F16)2.0 * x) / ((F16) 1.0f + AbsF((F16)2.0 * x)));
	}

#endif
#endif

F16 Fast_Swish_fp16(F16 x)

{
	F16 Beta = (F16) 1.0;
        return (x * Fast_Sigmoid_fp16(Beta * x));
}

F16 LeakyReLU_fp16(F16 x)

{
	if (x<0) x *= LEAK_CONSTANT;
	return x;
}

/*
	Uses exp(x) = lim (1 + x/n)^n	when n approaches infinite
	here we use n = 1024, x^1024 = (x^2)^512 = ... = ((((((((((x^2)^2)^2)^2)^2)^2)^2)^2)^2)^2)

	When x is small accuracy is pretty good, order of 10-3
*/
F16 Fast_Exp_fp16(F16 x)

{
  	x = (F16) 1.0 + x / (F16) 1024.0;
  	x *= x; x *= x; x *= x; x *= x; x *= x; x *= x; x *= x; x *= x; x *= x; x *= x;
  	return x;
}

F16 Fast_Log_fp16(F16 x)

{
	/**
	 * ln(x) = 0.9991150 * (x -1) - 0.4899597 * (x -1) ^ 2
	 *       + 0.2856751 * (x -1) ^ 3 - 0.1330566 * (x -1) ^ 4
	 *       + 0.03137207 * (x -1) ^ 5
	 */
	x = x - (F16) 1.0;
	F16 result = x * (F16) 0.9991150;

	F16 z = x*x;
	result -= z * (F16) 0.4899597;
	z = z * x;
	result += z * (F16) 0.2856751;
	z = z * x;
	result -= z * (F16) 0.1330566;
	z = z * x;
	result += z * (F16) 0.03137207;
	return result;
}

/* Set output features maps initial bias group
 
	Single precision form is called by KOP_SETBIAS

	KerParSetBias_fp16		Bias and out are short floats, output feature maps are evaluated in parallel (one per core)
	KerSetBias_fp16			Bias and out are short floats, a single output feature map is evaluated in parallel on all cores
*/

/* Set Bias, all outputs evalauted in parallel */

void KerParSetBias_fp16(KerSetBias_fp16_T *Arg)

{
	/* Bias in non DP context: point position b, no need to adjust */
	F16 * __restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int OutFeatures = Arg->OutFeatures;
	F16 * __restrict__ Bias = Arg->Bias;

	unsigned int CoreId = gap_coreid();
	unsigned int Chunk = ChunkSize(OutFeatures);
	unsigned int First = Chunk*CoreId;
	unsigned int Last = Min(First+Chunk, OutFeatures);

	for (unsigned int of=First; of<Last; of++) {
		F16V *LineOut = (F16V *) (Out+W*H*of);
		F16 B = Bias[of];
		F16V Bv = (F16V) {B, B};
		for (unsigned int i=0; i<((W*H)/4); i++) {
			LineOut[2*i] = Bv; LineOut[2*i+1] = Bv;
		}
		for (unsigned int i=(4*((W*H)/4)); i<(W*H); i++) Out[W*H*of+i] = B;
	}
	gap_waitbarrier(0);
}

/* Set bias, one output evaluated in parallel */

void KerSetBias_fp16(KerSetBias_fp16_T *Arg)

{
	/* Bias in non DP context: point position b. No need to shift this is just a container size change. */
	F16 * __restrict__ Out = Arg->Out;
	int W = Arg->W;
	int H = Arg->H;

	unsigned int CoreId = gap_coreid();
	unsigned int ChunkCell = ChunkSize(W*H);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Minu(First+ChunkCell, W*H);
	F16V * __restrict__ VectOut = (F16V *) (Out+First);
	int Iter = Max(0, Last-First);
	int i;

	F16 Bias = *Arg->Bias;
	F16V Vb = (F16V) {Bias, Bias};
	for (i=0; i<(Iter/2); i++) VectOut[i] = Vb;
	if (Iter&0x1) Out[Last-1] = Bias;
	gap_waitbarrier(0);
}

/* Activation group:

	Feature maps in and out are short floats:
		KerParReLU_fp16				All Feature maps are evaluated in parallel
		KerParReLUN_fp16			All Feature maps are evaluated in parallel
		KerParSwish_fp16			All Feature maps are evaluated in parallel
		KerParSigmoid_fp16			All Feature maps are evaluated in parallel
		KerParLeakyReLU_fp16			All Feature maps are evaluated in parallel
*/

void KerParReLU_fp16(KerActivation_fp16_T *Arg)

{
	F16 * __restrict__ In = Arg->In;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int OutFeatures = Arg->Feat;
	F16 * __restrict__ Out = Arg->Out;

	unsigned int S = OutFeatures*W*H, CoreId = gap_coreid(), Chunk = ChunkSize(S), First = Chunk*CoreId, Last = Min(First+Chunk, S), Iter = Last-First;
	F16V * VectIn  = (F16V *) (In+First);
	F16V * VectOut = (F16V *) (Out+First);
	F16V LB = (F16V) {0.0, 0.0};

	for (unsigned int i=0; i<Iter/4; i++) {
		F16V X = MaxF2(VectIn[2*i  ], LB);
		F16V Y = MaxF2(VectIn[2*i+1], LB);
		VectOut[2*i] = X; VectOut[2*i+1] = Y;
	}
	for (unsigned int i=4*(Iter/4); i<Iter; i++) Out[i] = MaxF(In[i], (F16)0.0);
	gap_waitbarrier(0);
}

void KerParReLUN_fp16(KerActivation_fp16_T *Arg)

{
	F16 * __restrict__ In = Arg->In;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int OutFeatures = Arg->Feat;
	F16 * __restrict__ Out = Arg->Out;
	F16 UB = Arg->UB;

	unsigned int S = OutFeatures*W*H, CoreId = gap_coreid(), Chunk = ChunkSize(S), First = Chunk*CoreId, Last = Min(First+Chunk, S), Iter = Last-First;
	F16V * VectIn  = (F16V *) (In+First);
	F16V * VectOut = (F16V *) (Out+First);

	F16V LBv = (F16V) {0.0, 0.0};
	F16V UBv = (F16V) {UB, UB};

	for (unsigned int i=0; i<Iter/4; i++) {
		F16V X = MinF2(MaxF2(VectIn[2*i  ], LBv), UBv);
		F16V Y = MinF2(MaxF2(VectIn[2*i+1], LBv), UBv);
		VectOut[2*i] = X; VectOut[2*i+1] = Y;
	}
	for (unsigned int i=4*(Iter/4); i<Iter; i++) Out[i] = MinF(MaxF(In[i], (F16)0.0), UB);

	gap_waitbarrier(0);
}

void KerParSwish_fp16(KerActivation_fp16_T *Arg)

{
	F16 * __restrict__ In = Arg->In;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int OutFeatures = Arg->Feat;
	F16 * __restrict__ Out = Arg->Out;

	unsigned int S = OutFeatures*W*H, CoreId = gap_coreid(), Chunk = ChunkSize(S), First = Chunk*CoreId, Last = Min(First+Chunk, S), Iter = Last-First;

	In  = (In+First); Out = (Out+First);
	for (unsigned int i=0; i<Iter/2; i++) {
		F16 Acc0 = In[2*i], Acc1 = In[2*i+1];
		Out[2*i]   = Fast_Swish_fp16(Acc0);
		Out[2*i+1] = Fast_Swish_fp16(Acc1);
	}
	if (Iter&0x1) Out[Iter-1] = Fast_Swish_fp16(In[Iter-1]);
	gap_waitbarrier(0);
}

void KerParSigmoid_fp16(KerActivation_fp16_T *Arg)

{
	F16 * __restrict__ In = Arg->In;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int OutFeatures = Arg->Feat;
	F16 * __restrict__ Out = Arg->Out;

	unsigned int S = OutFeatures*W*H, CoreId = gap_coreid(), Chunk = ChunkSize(S), First = Chunk*CoreId, Last = Min(First+Chunk, S), Iter = Last-First;

	In  = (In+First); Out = (Out+First);
	for (unsigned int i=0; i<Iter/2; i++) {
		F16 Acc0 = In[2*i], Acc1 = In[2*i+1];
		Out[2*i]   = Fast_Sigmoid_fp16(Acc0);
		Out[2*i+1] = Fast_Sigmoid_fp16(Acc1);
	}
	if (Iter&0x1) Out[Iter-1] = Fast_Sigmoid_fp16(In[Iter-1]);
	gap_waitbarrier(0);
}

void KerParTanh_fp16(KerActivation_fp16_T *Arg)

{
	F16 * __restrict__ In = Arg->In;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int OutFeatures = Arg->Feat;
	F16 * __restrict__ Out = Arg->Out;

	unsigned int S = OutFeatures*W*H, CoreId = gap_coreid(), Chunk = ChunkSize(S), First = Chunk*CoreId, Last = Min(First+Chunk, S), Iter = Last-First;

	In  = (In+First); Out = (Out+First);
	for (unsigned int i=0; i<Iter/2; i++) {
		F16 Acc0 = In[2*i], Acc1 = In[2*i+1];
		Out[2*i]   = Fast_Tanh_fp16(Acc0);
		Out[2*i+1] = Fast_Tanh_fp16(Acc1);
	}
	if (Iter&0x1) Out[Iter-1] = Fast_Tanh_fp16(In[Iter-1]);
	gap_waitbarrier(0);
}

void KerParLeakyReLU_fp16(KerActivation_fp16_T *Arg)

{
	F16 * __restrict__ In = Arg->In;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Wo = W;
	unsigned int Ho = H;
	unsigned int OutFeatures = Arg->Feat;
	F16 * __restrict__ Out = Arg->Out;


	unsigned int S = OutFeatures*W*H, CoreId = gap_coreid(), Chunk = ChunkSize(S), First = Chunk*CoreId, Last = Min(First+Chunk, S), Iter = Last-First;

	In  = (In+First); Out = (Out+First);
	for (unsigned int i=0; i<Iter/2; i++) {
		F16 Acc0 = In[2*i], Acc1 = In[2*i+1];
		Out[2*i]   = LeakyReLU_fp16(Acc0);
		Out[2*i+1] = LeakyReLU_fp16(Acc1);
	}
	if (Iter&0x1) Out[Iter-1] = LeakyReLU_fp16(In[Iter-1]);
	gap_waitbarrier(0);
}

/**************************************************************************************************************************************
	Linear Layer followed by optional activation 
**************************************************************************************************************************************/

/*
	All the KerParLinearLayer family evaluates output channels in parallel
*/

void KerParLinearLayer_fp16(KerLinear_fp16_T *Arg)

{
	F16 * __restrict__ In = Arg->In;
	int TotalInDim = Arg->TotalInDim;
	int InDim = Arg->InDim;
	int Init = Arg->TileIndex==0;
	const F16 * __restrict__ Filter = Arg->Filter;
	const F16 * __restrict__ Bias = Arg->Bias;
	F16 * __restrict__ Out = Arg->Out;

	int OutDim = Arg->OutDim;
	unsigned int CoreId = gap_coreid();
	unsigned int ChunkCell = ChunkSize(OutDim);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Min(First+ChunkCell, OutDim);
	int i,j;
	F16V * __restrict__ VectIn = (F16V *) In;

	for (i=First; i<Last; i++) {
		F16V * __restrict__ Filt = (F16V *) (&Filter[i*TotalInDim]);
		F16V Acc = (F16V) {0.0, 0.0};
		for (j = 0; j<(InDim/(2*2)); j++) {
			Acc += VectIn[2*j] * Filt[2*j];
			Acc += VectIn[2*j+1] * Filt[2*j+1];
		}
		if (InDim&0x2) Acc += VectIn[InDim/2-1] * Filt[InDim/2-1];
		F16 A = Acc[0]+Acc[1];
		if (InDim&0x1) A += In[InDim-1]*Filter[i*InDim+InDim-1];
		if (Init) A += Bias[i]; else A += Out[i];
		Out[i] = A;
	}
	gap_waitbarrier(0);
}

void KerParLinearLayerReLU_fp16(KerLinear_fp16_T *Arg)

{
	F16 * __restrict__ In = Arg->In;
	int TotalInDim = Arg->TotalInDim;
	int InDim = Arg->InDim;
	int Init = Arg->TileIndex==0;
	const F16 * __restrict__ Filter = Arg->Filter;
	const F16 * __restrict__ Bias = Arg->Bias;
	F16 * __restrict__ Out = Arg->Out;

	int OutDim = Arg->OutDim;
	unsigned int CoreId = gap_coreid();
	unsigned int ChunkCell = ChunkSize(OutDim);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Min(First+ChunkCell, OutDim);
	int i,j;
	F16V * __restrict__ VectIn = (F16V *) In;

	for (i=First; i<Last; i++) {
		F16V * __restrict__ Filt = (F16V *) (&Filter[i*TotalInDim]);
		F16V Acc = (F16V) {0.0, 0.0};
		for (j = 0; j<(InDim/(2*2)); j++) {
			Acc += VectIn[2*j] * Filt[2*j];
			Acc += VectIn[2*j+1] * Filt[2*j+1];
		}
		if (InDim&0x2) Acc += VectIn[InDim/2-1] * Filt[InDim/2-1];
		F16 A = Acc[0]+Acc[1];
		if (InDim&0x1) A += In[InDim-1]*Filter[i*InDim+InDim-1];
		if (Init) A += Bias[i]; else A += Out[i];
		Out[i] = MaxF(A, (F16) 0.0);
	}
	gap_waitbarrier(0);
}

void KerParLinearLayerReLUN_fp16(KerLinear_fp16_T *Arg)

{
	F16 * __restrict__ In = Arg->In;
	int TotalInDim = Arg->TotalInDim;
	int InDim = Arg->InDim;
	int Init = Arg->TileIndex==0;
	const F16 * __restrict__ Filter = Arg->Filter;
	const F16 * __restrict__ Bias = Arg->Bias;
	F16 * __restrict__ Out = Arg->Out;
	F16 UB = Arg->UB;

	int OutDim = Arg->OutDim;
	unsigned int CoreId = gap_coreid();
	unsigned int ChunkCell = ChunkSize(OutDim);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Min(First+ChunkCell, OutDim);
	int i,j;
	F16V * __restrict__ VectIn = (F16V *) In;

	for (i=First; i<Last; i++) {
		F16V * __restrict__ Filt = (F16V *) (&Filter[i*TotalInDim]);
		F16V Acc = (F16V) {0.0, 0.0};
		for (j = 0; j<(InDim/(2*2)); j++) {
			Acc += VectIn[2*j] * Filt[2*j];
			Acc += VectIn[2*j+1] * Filt[2*j+1];
		}
		if (InDim&0x2) Acc += VectIn[InDim/2-1] * Filt[InDim/2-1];
		F16 A = Acc[0]+Acc[1];
		if (InDim&0x1) A += In[InDim-1]*Filter[i*InDim+InDim-1];
		if (Init) A += Bias[i]; else A += Out[i];
		Out[i] = MinF(MaxF(A, (F16) 0.0), UB);
	}
	gap_waitbarrier(0);
}

void KerParLinearLayerSwish_fp16(KerLinear_fp16_T *Arg)

{
	F16 * __restrict__ In = Arg->In;
	int TotalInDim = Arg->TotalInDim;
	int InDim = Arg->InDim;
	int Init = Arg->TileIndex==0;
	const F16 * __restrict__ Filter = Arg->Filter;
	const F16 * __restrict__ Bias = Arg->Bias;
	F16 * __restrict__ Out = Arg->Out;

	int OutDim = Arg->OutDim;
	unsigned int CoreId = gap_coreid();
	unsigned int ChunkCell = ChunkSize(OutDim);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Min(First+ChunkCell, OutDim);
	int i,j;
	F16V * __restrict__ VectIn = (F16V *) In;

	for (i=First; i<Last; i++) {
		F16V * __restrict__ Filt = (F16V *) (&Filter[i*TotalInDim]);
		F16V Acc = (F16V) {0.0, 0.0};
		for (j = 0; j<(InDim/(2*2)); j++) {
			Acc += VectIn[2*j] * Filt[2*j];
			Acc += VectIn[2*j+1] * Filt[2*j+1];
		}
		if (InDim&0x2) Acc += VectIn[InDim/2-1] * Filt[InDim/2-1];
		F16 A = Acc[0]+Acc[1];
		if (InDim&0x1) A += In[InDim-1]*Filter[i*InDim+InDim-1];
		if (Init) A += Bias[i]; else A += Out[i];
		Out[i] = Fast_Swish_fp16(A);
	}
	gap_waitbarrier(0);
}

void KerParLinearLayerSigmoid_fp16(KerLinear_fp16_T *Arg)

{
	F16 * __restrict__ In = Arg->In;
	int TotalInDim = Arg->TotalInDim;
	int InDim = Arg->InDim;
	int Init = Arg->TileIndex==0;
	const F16 * __restrict__ Filter = Arg->Filter;
	const F16 * __restrict__ Bias = Arg->Bias;
	F16 * __restrict__ Out = Arg->Out;

	int OutDim = Arg->OutDim;
	unsigned int CoreId = gap_coreid();
	unsigned int ChunkCell = ChunkSize(OutDim);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Min(First+ChunkCell, OutDim);
	int i,j;
	F16V * __restrict__ VectIn = (F16V *) In;

	for (i=First; i<Last; i++) {
		F16V * __restrict__ Filt = (F16V *) (&Filter[i*TotalInDim]);

		F16V Acc = (F16V) {0.0, 0.0};
		for (j = 0; j<(InDim/(2*2)); j++) {
			Acc += VectIn[2*j] * Filt[2*j];
			Acc += VectIn[2*j+1] * Filt[2*j+1];
		}
		if (InDim&0x2) Acc += VectIn[InDim/2-1] * Filt[InDim/2-1];
		F16 A = Acc[0]+Acc[1];
		if (InDim&0x1) A += In[InDim-1]*Filter[i*InDim+InDim-1];
		if (Init) A += Bias[i]; else A += Out[i];
		Out[i] = Fast_Sigmoid_fp16(A);
	}
	gap_waitbarrier(0);
}

void KerParLinearLayerTanh_fp16(KerLinear_fp16_T *Arg)

{
	F16 * __restrict__ In = Arg->In;
	int TotalInDim = Arg->TotalInDim;
	int InDim = Arg->InDim;
	int Init = Arg->TileIndex==0;
	const F16 * __restrict__ Filter = Arg->Filter;
	const F16 * __restrict__ Bias = Arg->Bias;
	F16 * __restrict__ Out = Arg->Out;

	int OutDim = Arg->OutDim;
	unsigned int CoreId = gap_coreid();
	unsigned int ChunkCell = ChunkSize(OutDim);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Min(First+ChunkCell, OutDim);
	int i,j;
	F16V * __restrict__ VectIn = (F16V *) In;

	for (i=First; i<Last; i++) {
		F16V * __restrict__ Filt = (F16V *) (&Filter[i*TotalInDim]);

		F16V Acc = (F16V) {0.0, 0.0};
		for (j = 0; j<(InDim/(2*2)); j++) {
			Acc += VectIn[2*j] * Filt[2*j];
			Acc += VectIn[2*j+1] * Filt[2*j+1];
		}
		if (InDim&0x2) Acc += VectIn[InDim/2-1] * Filt[InDim/2-1];
		F16 A = Acc[0]+Acc[1];
		if (InDim&0x1) A += In[InDim-1]*Filter[i*InDim+InDim-1];
		if (Init) A += Bias[i]; else A += Out[i];
		Out[i] = Fast_Tanh_fp16(A);
	}
	gap_waitbarrier(0);
}

void KerParLinearLayerLeakyReLU_fp16(KerLinear_fp16_T *Arg)

{
	F16 * __restrict__ In = Arg->In;
	int TotalInDim = Arg->TotalInDim;
	int InDim = Arg->InDim;
	int Init = Arg->TileIndex==0;
	const F16 * __restrict__ Filter = Arg->Filter;
	const F16 * __restrict__ Bias = Arg->Bias;
	F16 * __restrict__ Out = Arg->Out;

	int OutDim = Arg->OutDim;
	unsigned int CoreId = gap_coreid();
	unsigned int ChunkCell = ChunkSize(OutDim);
	unsigned int First = CoreId*ChunkCell;
	unsigned int Last  = Min(First+ChunkCell, OutDim);
	int i,j;
	F16V * __restrict__ VectIn = (F16V *) In;

	for (i=First; i<Last; i++) {
		F16V * __restrict__ Filt = (F16V *) (&Filter[i*TotalInDim]);

		F16V Acc = (F16V) {0.0, 0.0};
		for (j = 0; j<(InDim/(2*2)); j++) {
			Acc += VectIn[2*j] * Filt[2*j];
			Acc += VectIn[2*j+1] * Filt[2*j+1];
		}
		if (InDim&0x2) Acc += VectIn[InDim/2-1] * Filt[InDim/2-1];
		F16 A = Acc[0]+Acc[1];
		if (InDim&0x1) A += In[InDim-1]*Filter[i*InDim+InDim-1];
		if (Init) A += Bias[i]; else A += Out[i];
		if (A<0.0) A *= LEAK_CONSTANT;
		Out[i] = A;
	}
	gap_waitbarrier(0);
}

