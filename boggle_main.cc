/*
Boggle Solver
Version 1 - CPU Solver
Miguel Aroca-Ouellette
05/14/2016
*/

#define DICT_SIZE 109583 //words_clean.txt
#define MAX_WORD_LEN 31
#define NUM_ADJ 8 //tiles can have at most 8 adjacent tiles
#define MIN_WORD_LEN 3 //minimum acceptable word length boggle

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <cassert>
#include "boggle_main.h"
using namespace std;

/*Constructor: Set dimensions of board.*/
Board::Board(int _width, int _height)
{
	width = _width;
	height = _height;
	grid = (Tile **)malloc(sizeof(Tile*)*height);
	for (int i = 0; i < height; i++)
		grid[i] = (Tile *)malloc(sizeof(Tile)*width);
}

/* Destructor */
Board::~Board()
{
	for (int i = 0; i < height; i++)
		free(grid[i]);
	free(grid);
}

/*Set letters. Ensure that it is the right size.*/
void Board::set_letters(string letters)
{
	assert(letters.length() == width*height);
	
	int x, y;
	for (int i = 0; i < (width*height); i++)
	{
		x = i % width;
		y = i / width;
		grid[y][x].x = x;
		grid[y][x].y = y;
		grid[y][x].letter = letters[i];
		grid[y][x].used = false;
	}
}

/*Get adjacent tile letters which are not used. Origin is top left.
Inputs: (x,y) coordinate of center tile.
		Pointer to adjacency list to fill. MUST BE ABLE TO ACCOMODATE 8 LETTERS.
		Returns indicates size of filled adjacenct list.*/
int Board::get_adj(Tile* center, Tile **adj)
{
	int size = 0;
	int x = center->x;
	int y = center->y;
	if ((x > 0) && (y > 0)) //top left
		push_tile(adj, x - 1, y - 1, &size);
	if ((x > 0) && (y < (height - 1))) //bottom left
		push_tile(adj, x - 1, y + 1, &size);
	if ((x < (width - 1)) && (y > 0)) //top right
		push_tile(adj, x + 1, y - 1, &size);
	if ((x < (width - 1)) && (y < (height - 1))) //bottom right
		push_tile(adj, x + 1, y + 1, &size);
	if (y > 0) //top center
		push_tile(adj, x, y - 1, &size);
	if (y < (height - 1)) //bottom center
		push_tile(adj, x, y + 1, &size);
	if (x > 0) //left center
		push_tile(adj, x - 1, y, &size);
	if (x < (width - 1)) //right center
		push_tile(adj, x + 1, y, &size);

	return size;
}

/*Private: Helper func'tion for get_adj. Only assigns the tile if it is unused. Increments counts.*/
void Board::push_tile(Tile **target, int x, int y, int *count)
{
	if (!grid[y][x].used)
		target[(*count)++] = &grid[y][x];
}

/* Prints the boggle board*/
void Board::print_board()
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
			cout << grid[y][x].letter<<" ";
		cout << "\n";
	}
}

/* Prints the used tiles as 1, otherwise 0*/
void Board::print_used()
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
			cout << ((grid[y][x].used)?1:0) << " ";
		cout << "\n";
	}
}

/*Getter for tile.*/
Tile* Board::get_tile(int x, int y)
{
	return &grid[y][x];
}

/*Returns all tiles in the board.*/
void Board::get_all_tiles(Tile *all_tiles[])
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
			all_tiles[y*width + x] = get_tile(x, y);
	}
}

/* Resets all tiles on board to unused*/
void Board::reset()
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
			grid[y][x].used = false;
	}
}

int main(int argc, char** argv)
{
	//parameters
	string dict_file = "wordsEn.txt";

	string letters = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvqxy";
	cout << letters.length() << "\n";
	Board smallBoard(10, 10);
	smallBoard.set_letters(letters);
	smallBoard.print_board();

	string dictionary[DICT_SIZE];

	read_file(dictionary, DICT_SIZE, dict_file);

	//test adjacency
	//string mystring = "plonk";
	//cout << ((recursive_find(mystring, 0, &smallBoard, NULL))? "true": "false");

	single_solve(dictionary, DICT_SIZE, &smallBoard);
	return 1;
}

/* Finds all words in Boggle grid one word at a time.*/
void single_solve(string dictionary[], int dict_size, Board *board)
{
	int count = 0;
	for (int i = 0; i < dict_size; i++)
	{
		if (dictionary[i].length() < MIN_WORD_LEN){ continue; }//skip words too short

		if (recursive_find(dictionary[i], 0, board, NULL))
			cout << ++count <<" "<<dictionary[i] << "\n";
		board->reset();
	}
	cout << "Found " <<count<< " words.";
}

/* Recursively searches for a word.
INPUTS: Word to search for.
		Character index of word. Increments by 1 at every recursion level.
		Boggle board.*/
bool recursive_find(string word, int char_idx, Board *board, Tile* curr_tile)
{
	Tile **check_tiles;
	int num_check;
	bool success = false;

	//check if done
	if (word.length() == (char_idx))
		return true;

	//cout << "------- Looking for: " << word[char_idx] << "\n";
	//board->print_used();

	if (char_idx == 0)
	{
		//First letter, check whole board
		//Assign size of board
		check_tiles = (Tile **)malloc(sizeof(Tile *)*board->height*board->width);

		//get all tiles
		board->get_all_tiles(check_tiles);
		num_check = board->height*board->width;
	}
	else
	{
		//any other letter -> search adjacency
		check_tiles = (Tile **)malloc(sizeof(Tile *)*NUM_ADJ);

		num_check = board->get_adj(curr_tile, check_tiles);
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
			if (recursive_find(word, char_idx+1, board, check_tiles[i]))
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

/*Reads frome file into string array.*/
void read_file(string dictionary[], int size, string filename)
{
	int count = 0;
	string line;
	ifstream myfile(filename.c_str());

	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			dictionary[count++] = line;
		}
	}
	else cout << "Unable to open file.";
}