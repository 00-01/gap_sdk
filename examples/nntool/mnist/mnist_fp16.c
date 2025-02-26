/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */


#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s
/* PMSIS includes. */
#include "pmsis.h"

/* Autotiler includes. */
#include "mnist.h"
#include "mnistKernels.h"
#include "gaplib/ImgIO.h"

#define pmsis_exit(n) exit(n)

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif

#if defined(USE_HYPER)
   AT_HYPERFLASH_FS_EXT_ADDR_TYPE mnist_L3_Flash = 0;
#elif defined(USE_SPI)
   AT_QSPIFLASH_FS_EXT_ADDR_TYPE mnist_L3_Flash = 0;
#endif

// Softmax always outputs Q15 short int even from 8 bit input
L2_MEM F16 *ResOut;
//Image in is unsigned but the model is trained with -1:1 inputs
//The preprocessing to scale the image is done in the CNN AT graph
L2_MEM unsigned char *Img_In;
L2_MEM F16 *Input_NN;

#define AT_INPUT_WIDTH  28
#define AT_INPUT_HEIGHT 28
#define AT_INPUT_COLORS 1
#define AT_INPUT_SIZE (AT_INPUT_WIDTH*AT_INPUT_HEIGHT*AT_INPUT_COLORS)
#define AT_INPUT_SIZE_BYTES (AT_INPUT_SIZE*sizeof(char))
//#define PRINT_IMAGE

char *ImageName = NULL;

static void cluster()
{
  printf("Running on cluster\n");
#ifdef PERF
  printf("Start timer\n");
  gap_cl_starttimer();
  gap_cl_resethwtimer();
#endif
  mnistCNN(Input_NN, ResOut);
  printf("Runner completed\n");
    //Checki Results
    int rec_digit = 0;
    float highest = (float) ResOut[0];
    for(int i = 0; i < 10; i++) {
      printf("class %d: %f \n", i, (float) ResOut[i]);
      if(ResOut[i] > highest) {
        highest = ResOut[i];
        rec_digit = i;
      }
    }
    printf("\n");

    printf("Recognized: %d\n", rec_digit);
}

int test_mnist(void)
{
    printf("Entering main controller\n");
    printf("Reading image\n");
    //Reading Image from Bridge
    /*------------------- Allocate Image Buffer ------------------------*/
    printf("Going to alloc the image buffer!\n");
    Img_In = (unsigned char *) AT_L2_ALLOC(0, AT_INPUT_SIZE_BYTES);
    Input_NN = (F16 *) AT_L2_ALLOC(0, AT_INPUT_SIZE_BYTES*sizeof(F16));
    if(Img_In==NULL || Input_NN==NULL) {
      printf("Image buffer alloc Error!\n");
      pmsis_exit(-1);
    } 

    char *ImageName = __XSTR(AT_IMAGE);
    if (ReadImageFromFile(ImageName, AT_INPUT_WIDTH, AT_INPUT_HEIGHT, AT_INPUT_COLORS, Img_In, AT_INPUT_SIZE_BYTES, IMGIO_OUTPUT_CHAR, 0)) {
        printf("Failed to load image %s\n", ImageName);
        pmsis_exit(-2);
    }

    /* Configure And open cluster. */
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    cl_conf.id = 0;
    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-4);
    }
    for (int i=0; i<AT_INPUT_SIZE; i++) Input_NN[i] = ((F16) Img_In[i]) / 128 - 1;
    // for (int i=0; i<AT_INPUT_SIZE; i++) printf("%f, ", Input_NN[i]); printf("\n");
    ResOut = (F16 *) AT_L2_ALLOC(0, 10 * sizeof(F16));
    if (ResOut == NULL) {
        printf("Failed to allocate Memory for Result (%d bytes)\n", 10*sizeof(F16));
        pmsis_exit(-3);
    }

    printf("Constructor\n");
    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    if (mnistCNN_Construct())
    {
        printf("Graph constructor exited with an error\n");
        pmsis_exit(-5);
    }

    printf("Call cluster\n");
    struct pi_cluster_task task = {0};
    task.entry = cluster;
    task.arg = NULL;
    task.stack_size = (unsigned int) STACK_SIZE;
    task.slave_stack_size = (unsigned int) SLAVE_STACK_SIZE;

    pi_cluster_send_task_to_cl(&cluster_dev, &task);

    mnistCNN_Destruct();

#ifdef PERF
    {
      unsigned int TotalCycles = 0, TotalOper = 0;
      printf("\n");
      for (int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        printf("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], AT_GraphOperInfosNames[i], ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
        TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
      }
      printf("\n");
      printf("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
      printf("\n");
    }
#endif

    // Close the cluster
    pi_cluster_close(&cluster_dev);

    AT_L2_FREE(0, Img_In, AT_INPUT_SIZE_BYTES);
    AT_L2_FREE(0, ResOut, 10 * sizeof(F16));
    printf("Ended\n");

    pmsis_exit(0);
    return 0;
}

int main()
{
    printf("\n\n\t *** NNTOOL Mnist FP16 Example ***\n\n");
    return pmsis_kickoff((void *) test_mnist);
}
