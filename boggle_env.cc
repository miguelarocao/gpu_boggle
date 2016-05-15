/*
Boggle Environment - Board and Tile Class
Miguel Aroca-Ouellette
05/14/2016
*/

#include <iostream>
#include <stdlib.h>
#include <string>
#include <cassert>
#include <time.h>
#include "boggle_env.h"

#define NUM_LETTERS 26

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

/*Populates the board with random letters.*/
void Board::genRandLetters()
{
	//random seed
	srand(time(NULL));

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			//get random number between 0 and 25 -> letters
			char rand_char = 'a' + rand() % NUM_LETTERS;
			grid[y][x].letter = rand_char;
			grid[y][x].x = x;
			grid[y][x].y = y;
			grid[y][x].used = false;
		}
	}
}

/*Set letters. Ensure that it is the right size.*/
void Board::setLetters(string letters)
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
int Board::getAdj(Tile* center, Tile **adj)
{
	int size = 0;
	int x = center->x;
	int y = center->y;
	if ((x > 0) && (y > 0)) //top left
		pushTile(adj, x - 1, y - 1, &size);
	if ((x > 0) && (y < (height - 1))) //bottom left
		pushTile(adj, x - 1, y + 1, &size);
	if ((x < (width - 1)) && (y > 0)) //top right
		pushTile(adj, x + 1, y - 1, &size);
	if ((x < (width - 1)) && (y < (height - 1))) //bottom right
		pushTile(adj, x + 1, y + 1, &size);
	if (y > 0) //top center
		pushTile(adj, x, y - 1, &size);
	if (y < (height - 1)) //bottom center
		pushTile(adj, x, y + 1, &size);
	if (x > 0) //left center
		pushTile(adj, x - 1, y, &size);
	if (x < (width - 1)) //right center
		pushTile(adj, x + 1, y, &size);

	return size;
}

/*Private: Helper function for get_adj. Only assigns the tile if it is unused. Increments counts.*/
void Board::pushTile(Tile **target, int x, int y, int *count)
{
	if (!grid[y][x].used)
		target[(*count)++] = &grid[y][x];
}

/* Prints the boggle board*/
void Board::printBoard()
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
			cout << grid[y][x].letter << " ";
		cout << "\n";
	}
}

/* Prints the used tiles as 1, otherwise 0*/
void Board::printUsed()
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
			cout << ((grid[y][x].used) ? 1 : 0) << " ";
		cout << "\n";
	}
}

/*Getter for tile.*/
Tile* Board::getTile(int x, int y)
{
	return &grid[y][x];
}

/*Returns all tiles in the board.*/
void Board::getAllTiles(Tile *all_tiles[])
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
			all_tiles[y*width + x] = getTile(x, y);
	}
}

/* Resets all tiles on board to unused*/
void Board::resetBoard()
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
			grid[y][x].used = false;
	}
}