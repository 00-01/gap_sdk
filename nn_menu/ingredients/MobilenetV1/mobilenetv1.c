/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */

#include <stdio.h>

#include "mobilenetv1.h"
#include "mobilenetv1Kernels.h"
#include "gaplib/ImgIO.h"
#include "mobilenetv1Info.h"

#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s


#define AT_INPUT_SIZE (AT_INPUT_WIDTH*AT_INPUT_HEIGHT*AT_INPUT_COLORS)

#ifndef STACK_SIZE
#define STACK_SIZE     2048 
#endif

AT_HYPERFLASH_FS_EXT_ADDR_TYPE __PREFIX(_L3_Flash) = 0;
AT_HYPERRAM_T HyperRam;

static uint32_t l3_buff;
signed short int Output_1[1000];
typedef signed char IMAGE_IN_T;
unsigned char *Input_1;

static void RunNetwork()
{
	printf("Running on cluster\n");
#ifdef PERF
	printf("Start timer\n");
	gap_cl_starttimer();
	gap_cl_resethwtimer();
#endif
	__PREFIX(CNN)(l3_buff, Output_1);
	printf("Runner completed\n");

	printf("\n");
}

int start()
{
	char *ImageName = __XSTR(AT_IMAGE);
	struct pi_device cluster_dev;
	struct pi_cluster_task *task;
	struct pi_cluster_conf conf;

	//Input image size
	printf("Entering main controller\n");
	pi_freq_set(PI_FREQ_DOMAIN_FC,FREQ_FC*1000*1000);
	
	struct pi_hyperram_conf hyper_conf;
	pi_hyperram_conf_init(&hyper_conf);
	pi_open_from_conf(&HyperRam, &hyper_conf);
	if (pi_ram_open(&HyperRam))
	{
		printf("Error ram open !\n");
		pmsis_exit(-3);
	}

	if (pi_ram_alloc(&HyperRam, &l3_buff, (uint32_t) AT_INPUT_WIDTH*AT_INPUT_HEIGHT*AT_INPUT_COLORS))
	{
		printf("Ram malloc failed !\n");
		pmsis_exit(-4);
	}

	printf("Reading image: %s\n",ImageName);
	//Reading Image from Bridge
	Input_1 = (unsigned char*)pmsis_l2_malloc(AT_INPUT_WIDTH*AT_INPUT_HEIGHT*AT_INPUT_COLORS);
	if (ReadImageFromFile(ImageName, AT_INPUT_WIDTH, AT_INPUT_HEIGHT, AT_INPUT_COLORS,
						  Input_1, AT_INPUT_SIZE*sizeof(IMAGE_IN_T), IMGIO_OUTPUT_CHAR, 0)) {
		printf("Failed to load image %s\n", ImageName);
		return 1;
	}
	pi_ram_write(&HyperRam, (l3_buff), Input_1, (uint32_t)AT_INPUT_WIDTH*AT_INPUT_HEIGHT*AT_INPUT_COLORS);
	pmsis_l2_malloc_free(Input_1,AT_INPUT_WIDTH*AT_INPUT_HEIGHT*AT_INPUT_COLORS);

	printf("Finished reading image\n");


	pi_cluster_conf_init(&conf);
	pi_open_from_conf(&cluster_dev, (void *)&conf);
	pi_cluster_open(&cluster_dev);
	pi_freq_set(PI_FREQ_DOMAIN_CL,FREQ_CL*1000*1000);
	task = pmsis_l2_malloc(sizeof(struct pi_cluster_task));
	if (!task) {
		printf("failed to allocate memory for task\n");
	}
	memset(task, 0, sizeof(struct pi_cluster_task));
	task->entry = &RunNetwork;
	task->stack_size = STACK_SIZE;
	task->slave_stack_size = SLAVE_STACK_SIZE;
	task->arg = NULL;

	printf("Constructor\n");

	// IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
	if (__PREFIX(CNN_Construct)())
	{
		printf("Graph constructor exited with an error\n");
		return 1;
	}

	printf("Call cluster\n");
	// Execute the function "RunNetwork" on the cluster.
	pi_cluster_send_task_to_cl(&cluster_dev, task);
	//Check Results
	int max_res=0;
	float max_res_f=0;
	int outclass=0;
	for(int i=0;i<1000;i++){
		if(Output_1[i]>max_res){
			max_res_f = (float) FIX2FP((Output_1[i]*S89_Op_output_1_OUT_QSCALE), S89_Op_output_1_OUT_QNORM) ;
			outclass=i;
		}
	}

	printf("Detected Output Class: %d with value: %f \n", outclass, max_res_f);
	printf("\n");

#ifdef PERF
	{
		unsigned int TotalCycles = 0, TotalOper = 0;
		printf("\n");
		for (int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
			printf("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", AT_GraphNodeNames[i],
				   AT_GraphPerf[i], AT_GraphOperInfosNames[i], ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
			TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
		}
		printf("\n");
		printf("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
		printf("\n");
	}
#endif

	__PREFIX(CNN_Destruct)();
	pi_cluster_close(&cluster_dev);

	if(outclass==2 && max_res_f>0.99){
		printf("Correctet Results\n");
		pmsis_exit(0);
		
	}
	else{
		printf("Wrong Results: Output calss %d with accuracy %f expected outclass 2 with accuracy >0.99\n",outclass,max_res_f);
		pmsis_exit(-1);
	}
	return 0;
}

int main(void)
{
	return pmsis_kickoff((void *) start);
}
