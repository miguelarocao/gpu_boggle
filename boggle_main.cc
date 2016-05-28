/*
Boggle Solver
Version 1 - CPU Solver
Miguel Aroca-Ouellette
05/14/2016
*/

#define DICT_SIZE 109583 //words_clean.txt
#define NUM_ADJ 8 //tiles can have at most 8 adjacent tiles
#define MIN_WORD_LEN 3 //minimum acceptable word length boggle
#define VERBOSE 1

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cassert>
#include <time.h>
#include <windows.h> // For Timing via "QueryPerformanceCounter"
#include "boggle_main.h"

int main(int argc, char** argv)
{
	//parameters
	string dict_file = "wordsEn.txt";

	//initialize timer
	initTiming();
	double time_initial, time_final;

	//setup board
	string letters = "abcdefghijklmnop";
	//cout << letters.length() << "\n";
	Board board(4, 4);
	//board.genRandLetters();
	board.setLetters(letters);
	board.printBoard();

	//read dictionary
	char **dictionary = (char **)malloc(sizeof(char*)*DICT_SIZE);
	for (int i = 0; i < DICT_SIZE; i++)
		dictionary[i] = (char *)malloc(sizeof(char)*MAX_WORD_LEN);
	readFile(dictionary, dict_file);
	

	//build prefix tree
	Trie prefix;
	prefix.buildFromDict(dictionary, DICT_SIZE);

	
	//solve in single mode
	time_initial = preciseClock();
	singleSolve(dictionary, DICT_SIZE, &board);
	time_final = preciseClock();
	cout << "Single word solve: " << (time_final - time_initial) << " ms.\n";

	//Prefix solve
	time_initial = preciseClock();
	prefixSolve(&prefix, &board);
	time_final = preciseClock();
	cout << "Prefix solve: " << (time_final - time_initial) << " ms.\n";

	return 1;
}

/*--- GPU Solving Handlers ---*/
/*
void single_gpu(string dictionary[], Board *board)
{
	//allocate memory on GPU
	Board *dev_board;
	string *dev_dic[];
	cudaMalloc((void **)&dev_board, sizeof(Board));
	cudaMalloc((void **)&dev_dict, sizeof(string)*DICT_SIZE);
}*/

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

/* Finds all words in Boggle grid by using prefix trees to traverse grid. */
int prefixSolve(Trie *prefix, Board *board)
{
	int char_idx = 0;
	int word_cnt = 0;
	char word[MAX_WORD_LEN];
	Node* root = prefix->getRoot();
	prefixTraversal(prefix, root, board, NULL, &word_cnt, word, char_idx);

	cout << "Found " << word_cnt << " words.\n";
	return word_cnt;
}

/* Recursively travels prefix tree finding and printing words. */
void prefixTraversal(Trie *prefix, Node *curr_node, Board *board, Tile *curr_tile, int *word_cnt, char word[], int char_idx)
{
	Tile **check_tiles;
	int num_check;
	bool success = false;
	
	//check if word found and long enough -> print
	if ( (curr_node->isEndWord()) && (char_idx>=MIN_WORD_LEN) && (!curr_node->isUsed()) )
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

/* Finds all words in Boggle grid one word at a time.*/
int singleSolve(char **dictionary, int dict_size, Board *board)
{
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
	cout << "Found " <<count<< " words.\n";
	return count;
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
			if (recursiveFind(word, length, char_idx+1, board, check_tiles[i]))
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