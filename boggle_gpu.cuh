/*
Boggle GPU Kernels Header
Miguel Aroca-Ouellette
05/28/2016
*/

#ifndef BOGGLE_GPU_H
#define BOGGLE_GPU_H

#include "boggle_env.cuh"

void cudaCallSingleSolveKernel(
	const unsigned int blocks,
	const unsigned int threadsPerBlock,
	char *dictionary,
	int dict_size,
	int max_word_len,
	Board* board,
	int *word_count);

#endif