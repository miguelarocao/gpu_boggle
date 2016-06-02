/*
Boggle Solver
Version 1 - CPU Solver
Miguel Aroca-Ouellette
05/14/2016
*/

#define DICT_SIZE 109583 //words_clean.txt
#define MAX_WORD_LEN 32 //includes null terminating character
#define MIN_WORD_LEN 3 //minimum acceptable word length boggle
#define MAX_BLOCKS 1024
#define THREADS_PER_BLOCK 5
#define VERBOSE 0

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include <time.h>
#include <windows.h> // For Timing via "QueryPerformanceCounter"
#include <cuda_runtime.h>
#include "boggle_main.h"
#include "boggle_gpu.cuh"



/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(
	cudaError_t code,
	const char *file,
	int line,
	bool abort = true)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n",
			cudaGetErrorString(code), file, line);
		exit(code);
	}
}

int main(int argc, char** argv)
{

	//input is board width and board height
	if (argc < 3){
		printf("Usage: (Board width) (Board height)\n");
		exit(-1);
	}
	int board_width = atoi(argv[1]);
	int board_height = atoi(argv[2]);

	//parameters
	string dict_file = "wordsEn.txt";
	int num_trials = 2;

	//initialize timer
	initTiming();
	double time_elapsed;
	
	//set random seed
	srand(time(NULL));

	//setup board
	//string letters = "zyxzrfuddwntbjjdqjyrtaovkqiijmayjbufwtairdqytwapr";
	//cout << letters.length() << "\n";
	//board.setLetters(letters);

	//read dictionary
	char **dictionary = (char **)malloc(sizeof(char*)*DICT_SIZE);
	for (int i = 0; i < DICT_SIZE; i++)
		dictionary[i] = (char *)malloc(sizeof(char)*MAX_WORD_LEN);
	readFile(dictionary, dict_file);


	//build prefix tree
	Trie prefix;
	prefix.buildFromDict(dictionary, DICT_SIZE);

	// ---------- initialize GPU memory
	Board *dev_board;
	char *dev_dict;
	Tile *dev_grid;
	int *dev_word_count;
	//dictionary setup: flatten
	gpuErrChk(cudaMalloc((void **)&dev_dict, sizeof(char)*MAX_WORD_LEN*DICT_SIZE));
	for (int i = 0; i < DICT_SIZE; i++)
		gpuErrChk(cudaMemcpy(&dev_dict[i*MAX_WORD_LEN], dictionary[i], sizeof(char)*MAX_WORD_LEN, cudaMemcpyHostToDevice));

	//board setup
	cudaMalloc((void **)&dev_board, sizeof(Board));
	cudaMalloc((void **)&dev_grid, sizeof(Tile)*(board_width*board_height));

	//word count
	cudaMalloc((void **)&dev_word_count, sizeof(int));
	// ---------- done initializing  GPU memory

	//Timing loop
	cout << "Single CPU \t Prefix CPU \t GPU (ms) \n";
	for (int i = 0; i < num_trials; i++)
	{
		//generate random letters
		Board board(board_height, board_width);
		board.genRandLetters();

#if VERBOSE
		board.printBoard();
#endif

		//solve in single mode
#if VERBOSE
		cout << "\nSingle mode solver...\n";
#endif
		time_elapsed = singleSolve(dictionary, DICT_SIZE, &board);
		cout << time_elapsed << "\t\t";

		board.resetBoard();

		//Prefix solve
#if VERBOSE
		cout << "\nPrefix mode solver...\n";
#endif
		time_elapsed = prefixSolve(&prefix, &board);
		cout << time_elapsed << "\t";

		board.resetBoard();

		//GPU Solve
#if VERBOSE
		cout << "\nGPU solver...\n";
#endif
		time_elapsed = single_gpu(dev_dict, &board, dev_board, dev_grid, dev_word_count);
		cout << time_elapsed << "\t\n";
	}

	//free dictionary memory
	for (int i = 0; i < DICT_SIZE; i++)
		free(dictionary[i]);
	free(dictionary);

	return 1;
}

/*--- GPU Solver ---*/


// Solve and Returns computation time.
float single_gpu(char *dev_dict, Board *board, Board *dev_board, Tile *dev_grid, int *dev_word_count)
{
	//set blocks & threads per block
	const unsigned int threadsPerBlock = THREADS_PER_BLOCK;
	const unsigned int blocks = min(MAX_BLOCKS, (unsigned int)ceil(
		DICT_SIZE / float(threadsPerBlock)));
	
	//copy over board
	cudaMemcpy(dev_board, board, sizeof(Board), cudaMemcpyHostToDevice);

	//copy over grid
	int grid_size = board->getNumTiles();
	cudaMemcpy(dev_grid, board->grid, sizeof(Tile)*grid_size, cudaMemcpyHostToDevice);
	cudaMemcpy(&(dev_board->grid), &dev_grid, sizeof(Tile *), cudaMemcpyHostToDevice); //assign to object

	//memset word count
	int word_count;
	cudaMemset(dev_word_count, 0, sizeof(int));

	//run and time
	double time_initial, time_final;
	time_initial = preciseClock();
	cudaCallSingleSolveKernel(blocks, threadsPerBlock, dev_dict, DICT_SIZE, MAX_WORD_LEN, dev_board, dev_word_count);
	time_final = preciseClock();

	//copy back and print results
	cudaMemcpy(&word_count, dev_word_count, sizeof(int), cudaMemcpyDeviceToHost);

#if VERBOSE
	cout << "Found " << word_count << " words.\n";
#endif

	if (word_count == 0)
	{
		printf("GPU Solver failed! Try reducing board size or threads/block.\n");
	}

	//Free memory
	//cudaFree(dev_dict);

	return (time_final - time_initial);
}

/*--- Timing Functions --- */

// Global variables to assist in timing
double PCFreq = 0.0;
__int64 CounterStart = 0;

// Initialize Windows-specific precise timing 
void initTiming()
{
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		printf("QueryPerformanceFrequency failed! Timing routines won't work. \n");

	PCFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}

// Get precise time
double preciseClock()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart) / PCFreq;
}

/*--- PREFIX SOLVE --- */

/* Finds all words in Boggle grid by using prefix trees to traverse grid. Returns computation time. */
float prefixSolve(Trie *prefix, Board *board)
{

	double time_initial, time_final;
	time_initial = preciseClock();
	int char_idx = 0;
	int word_cnt = 0;
	char word[MAX_WORD_LEN];
	Node* root = prefix->getRoot();
	prefixTraversal(prefix, root, board, NULL, &word_cnt, word, char_idx);
	time_final = preciseClock();
#if VERBOSE
	cout << "Found " << word_cnt << " words.\n";
#endif
	return (time_final - time_initial);
}

/* Recursively travels prefix tree finding and printing words. */
void prefixTraversal(Trie *prefix, Node *curr_node, Board *board, Tile *curr_tile, int *word_cnt, char word[], int char_idx)
{
	Tile **check_tiles;
	int num_check;
	bool success = false;

	//check if word found and long enough -> print
	if ((curr_node->isEndWord()) && (char_idx >= MIN_WORD_LEN) && (!curr_node->isUsed()))
	{
#if VERBOSE
		cout << *word_cnt << ": ";
		for (int i = 0; i < char_idx; i++)
			cout << word[i];
		cout << "\n";
#endif
		(*word_cnt)++; //increase word count
		curr_node->setUsed(true);
	}

	if (char_idx == 0)
	{
		//First letter, check whole board
		//Assign size of board
		check_tiles = (Tile **)malloc(sizeof(Tile *)*board->getNumTiles());

		//get all tiles
		board->getAllTiles(check_tiles);
		num_check = board->getNumTiles();
	}
	else
	{
		//any other letter -> search adjacency
		check_tiles = (Tile **)malloc(sizeof(Tile *)*NUM_ADJ);

		num_check = board->getAdj(curr_tile, check_tiles);
	}

	for (int i = 0; i < num_check; i++)
	{
		Node* next_node = prefix->getChild(curr_node, check_tiles[i]->letter);

		//if not none then traverse
		if (next_node != NULL)
		{
			//add to current word!
			word[char_idx] = check_tiles[i]->letter;
			check_tiles[i]->used = true; //mark as used
			prefixTraversal(prefix, next_node, board, check_tiles[i], word_cnt, word, char_idx + 1);
			check_tiles[i]->used = false; //mark as unused
		}
	}

	//if adjacency is empty -> done
}

/*--- SINGLE WORD SOLVE ---*/

/* Finds all words in Boggle grid one word at a time. returns computation time.*/
float singleSolve(char **dictionary, int dict_size, Board *board)
{
	double time_initial, time_final;
	time_initial = preciseClock();
	int count = 0;
	for (int i = 0; i < dict_size; i++)
	{
		int length = wordLength(dictionary[i]);
		if (length < MIN_WORD_LEN){ continue; }//skip words too short

		if (recursiveFind(dictionary[i], length, 0, board, NULL))
		{
			count++;
#if VERBOSE
			cout << count << " " << dictionary[i] << "\n";
#endif
		}

		board->resetBoard();
	}
	time_final = preciseClock();
#if VERBOSE
	cout << "Found " << count << " words.\n";
#endif
	return (time_final - time_initial);
}

/* Recursively searches for a word.
INPUTS: Word to search for.
Character index of word. Increments by 1 at every recursion level.
Boggle board.*/
bool recursiveFind(char *word, int length, int char_idx, Board *board, Tile* curr_tile)
{
	Tile **check_tiles;
	int num_check;
	bool success = false;

	//check if done
	if (length == (char_idx))
		return true;

	if (char_idx == 0)
	{
		//First letter, check whole board
		//Assign size of board
		check_tiles = (Tile **)malloc(sizeof(Tile *)*board->getNumTiles());

		//get all tiles
		board->getAllTiles(check_tiles);
		num_check = board->getNumTiles();
	}
	else
	{
		//any other letter -> search adjacency
		check_tiles = (Tile **)malloc(sizeof(Tile *)*NUM_ADJ);

		num_check = board->getAdj(curr_tile, check_tiles);
	}

	/*cout << num_check << "\n";
	for (int i = 0; i < num_check; i++)
	cout << check_tiles[i]->letter << " ";
	cout << "\n";*/

	//Check potential tiles
	for (int i = 0; i < num_check; i++)
	{
		if (check_tiles[i]->letter == word[char_idx])
		{
			check_tiles[i]->used = true; //mark as used
			if (recursiveFind(word, length, char_idx + 1, board, check_tiles[i]))
			{
				success = true;
				break;
			}
			else
				check_tiles[i]->used = false; //mark as unused
		}
	}

	//free data
	free(check_tiles);
	return success;
}

/* Returns length of word. Must be terminated by null character! */
int wordLength(char *word)
{
	int length = 0;
	while (true)
	{
		if (word[length++] == '\0')
			return length - 1;
	}
}

/*Reads frome file into string array.*/
void readFile(char **dictionary, string filename)
{
	int count = 0;
	string line;
	ifstream myfile(filename.c_str());

	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			strcpy(dictionary[count++], line.c_str());
		}
	}
	else cout << "Unable to open file.";
}