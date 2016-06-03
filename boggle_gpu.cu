/*
Boggle GPU Kernels
Miguel Aroca-Ouellette
05/28/2016
*/

#define MIN_WORD_LEN 3
#define MAX_WORD_LEN 32
#define VERBOSE 0

#include <cstdio>
#include <cuda_runtime.h>
#include "boggle_gpu.cuh"

/* Paralle Single Word Solve Kernel */
__global__
void cudaSingleSolveKernel(char *dictionary, int dict_size, int max_word_len, Board* orig_board, int *word_count)
{
	//get index
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//copy board
	Board new_board = *orig_board;
	Board *board = &new_board;

	//board vars
	unsigned int num_tiles = board->getNumTiles();
	Tile **path_tiles = (Tile **)malloc(sizeof(Tile *)*max_word_len);
	Tile **start_tiles = (Tile **)malloc(sizeof(Tile *)*num_tiles);
	Tile *adj_tiles[NUM_ADJ];
	Tile *curr_tile;
	int start_count;

	while (idx < dict_size)
	{
		//reset
		memset(path_tiles, 0, sizeof(Tile *)*max_word_len);
		memset(start_tiles, 0, sizeof(Tile *)*num_tiles);
		memset(adj_tiles, 0, sizeof(Tile *)*NUM_ADJ);

		char* word = &dictionary[idx*max_word_len];

		//get start tiles
		start_count = board->getLetterCount(word[0]);
		board->getTilesByLetter(start_tiles, word[0]);

		for (int i = 0; i < start_count; i++)
		{
			bool success = false;
			int char_idx = 1;

			curr_tile = start_tiles[i];
			curr_tile->used = true;
			path_tiles[0] = curr_tile;
			
			while (true)
			{
				if (word[char_idx] == '\0')
				{
					success = true;
					break;
				}

				//get adjacency
				board->getAdjList(curr_tile, adj_tiles);
				int j;
				for (j = 0; j < NUM_ADJ; j++)
				{
					if (adj_tiles[j] == NULL)
						continue; //skip nulls
					else if ((adj_tiles[j]->letter == word[char_idx]) && (!checkTileList(adj_tiles[j], path_tiles, char_idx)))
					{
						//mark tile as used
						curr_tile->adj_available[j] = false;
						path_tiles[char_idx] = adj_tiles[j];
						curr_tile = adj_tiles[j];
						break;
					}
				}
				if (j == NUM_ADJ)
				{
					//check if done!
					if (char_idx == 1)
						break; //done!

					//reset availability
					for (int k = 0; k < NUM_ADJ; k++)
						curr_tile->adj_available[k] = true;
						
					//failure to find anything, remove from used tile list
					curr_tile = path_tiles[char_idx - 2]; //go back up one letter
					path_tiles[--char_idx] = NULL;		
				}
				else
					char_idx++; //success, go to next letter
			}

			//check if done
			if (success)
			{
				if (char_idx >= MIN_WORD_LEN)
				{
#if VERBOSE
					printf("%s\n",word);
#endif
					atomicAdd(word_count, 1);
				}
				break;
			}

			//reset for next start letter
			board->resetBoard();
		}
		//INCREMENT IDX
		idx += blockDim.x * gridDim.x;
	}
	free(path_tiles);
	free(start_tiles);
}

/* Kernel Call Functions */

void cudaCallSingleSolveKernel(
	const unsigned int blocks,
	const unsigned int threadsPerBlock,
	char *dictionary,
	int dict_size,
	int max_word_len,
	Board* board,
	int *word_count)
{
	cudaSingleSolveKernel << < blocks, threadsPerBlock >> >(dictionary, dict_size, max_word_len, board, word_count);
}