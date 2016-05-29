/*
Boggle GPU Kernels
Miguel Aroca-Ouellette
05/28/2016
*/

#define MIN_WORD_LEN 3
#define MAX_WORD_LEN 32

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
	Tile **start_tiles = (Tile **)malloc(sizeof(Tile *)*num_tiles);
	Tile **path_tiles = (Tile **)malloc(sizeof(Tile *)*max_word_len);
	Tile *adj_tiles[NUM_ADJ];
	memset(start_tiles, 0, sizeof(Tile *)*num_tiles);
	memset(path_tiles, 0, sizeof(Tile *)*max_word_len);
	Tile *curr_tile;

	char* check_word = "waywardson";
	while (idx < dict_size)
	{
		bool verbose = true;
		char* word = &dictionary[idx*max_word_len];
		for (int i = 0; i < 10; i++)
		{
			if (word[i] != check_word[i])
			{
				verbose = false;
				break;
			}
		}
		//get start tiles
		bool success = false;
		board->getAllTiles(start_tiles);
		for (int i = 0; i < num_tiles; i++)
		{
			curr_tile = start_tiles[i];
			if (curr_tile->letter == word[0])
			{
				if (verbose) { printf("%c\n", word[0]); }
				int char_idx = 1;//always 1
				//mark start tile as used
				curr_tile->used = true;
				path_tiles[0] = curr_tile;
				//look for rest of word
				int count = 0;
				while (true)
				{
					if (count > 100)
					{
						printf("FAILURE on %s!\n", word);
						break;
					}
					if (word[char_idx] == '\0')
					{
						success = true;
						break;
					}
					//get adjacency
					memset(adj_tiles, 0, sizeof(Tile *)*NUM_ADJ);
					board->getAdjList(curr_tile, adj_tiles);
					int j;
					if (verbose) { printf("Goal %c Curr: %c (x,y)=(%d,%d)\n", word[char_idx], curr_tile->letter, curr_tile->x, curr_tile->y); }
					for (j = 0; j < NUM_ADJ; j++)
					{
						if (adj_tiles[j] == NULL)
							continue; //skip nulls

						if ((adj_tiles[j]->letter == word[char_idx]) && (!checkTileList(adj_tiles[j], path_tiles, char_idx)))
						{
							//mark tile as used
							if (verbose) { printf("Found %c\n", adj_tiles[j]->letter); }
							curr_tile->adj_available[j] = false;
							path_tiles[char_idx] = adj_tiles[j];
							curr_tile = adj_tiles[j];
							if (verbose) { printf("Found %c\n", adj_tiles[j]->letter); }
							break;
						}
					}
					if (j == NUM_ADJ)
					{
						if (verbose) { printf("-%c\n", word[char_idx]); }
						//check if done!
						if (char_idx == 1)
						{
							if (verbose) { printf("Couldn't find it!\n"); }
							break; //done!
						}

						//reset availability
						for (int k = 0; k < NUM_ADJ; k++)
							curr_tile->adj_available[k] = true;
						
						//failure to find anything, remove from used tile list
						curr_tile = path_tiles[char_idx - 2]; //go back up one letter
						path_tiles[--char_idx] = NULL;
						if (verbose) { printf("Went back to %c\n", curr_tile->letter); }
							
					}
					else
						char_idx++; //success, go to next letter
					count++;
					if (verbose)
					{
						for (int i = 0; i < max_word_len; i++)
						{
							if (path_tiles[i] != NULL)
								printf("%c ", path_tiles[i]->letter);
							else
								printf(" ");
						}

						printf("\n");
					}
				}

				//check if done
				if (success)
				{
					if (char_idx >= MIN_WORD_LEN)
					{
						printf("Found %s\n", word);
						atomicAdd(word_count, 1);
					}
					break;
				}

				//reset for next start letter
				board->resetBoard();
			}
		}
		//INCREMENT IDX
		idx += blockDim.x * gridDim.x;
	}
	free(start_tiles);
	free(path_tiles);
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