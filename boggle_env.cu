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
#include "boggle_env.cuh"

#define NUM_LETTERS 26

/*Constructor: Set dimensions of board.*/
CUDA_CALLABLE_MEMBER
Board::Board(int _width, int _height)
{
	width = _width;
	height = _height;
	grid = (Tile *)malloc(sizeof(Tile)*height*width);	
}

/* Destructor */
CUDA_CALLABLE_MEMBER
Board::~Board()
{
	free(grid);
}

/*Copy construtor*/
CUDA_CALLABLE_MEMBER
Board::Board(const Board &obj)
{
	//printf("Copy constructor allocating grid.\n");
	width = obj.width;
	height = obj.height;
	grid = (Tile *)malloc(sizeof(Tile)*height*width);
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			grid[x + y*width].letter = obj.grid[x + y*width].letter;
			grid[x + y*width].x = x;
			grid[x + y*width].y = y;
			grid[x + y*width].used = false;
			for (int i = 0; i < NUM_ADJ; i++)
			{
				grid[x + y*width].adj_available[i] = true;
				grid[x + y*width].adj_list[i] = NULL;
			}
			getAllAdj(&grid[x + y*width], grid[x + y*width].adj_list);
		}
	}
}

/*Populates the board with random letters.*/
void Board::genRandLetters()
{
	//random seed

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			//get random number between 0 and 25 -> letters
			char rand_char = 'a' + rand() % NUM_LETTERS;
			grid[x + y*width].letter = rand_char;
			grid[x + y*width].x = x;
			grid[x + y*width].y = y;
			grid[x + y*width].used = false;
			for (int i = 0; i < NUM_ADJ; i++)
			{
				grid[x + y*width].adj_available[i] = true;
				grid[x + y*width].adj_list[i] = NULL;
			}
			getAllAdj(&grid[x + y*width], grid[x + y*width].adj_list);
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
		grid[x+y*width].x = x;
		grid[x+y*width].y = y;
		grid[x+y*width].letter = letters[i];
		grid[x+y*width].used = false;
		for (int i = 0; i < NUM_ADJ; i++)
		{
			grid[x + y*width].adj_available[i] = true;
			grid[x + y*width].adj_list[i] = NULL;
		}
		getAllAdj(&grid[x + y*width], grid[x + y*width].adj_list);
	}
}

/* Returns list of adjancent tiles filtered by adj_available list */
void Board::getAdjList(Tile *center, Tile **adj)
{
	
	for (int i = 0; i < NUM_ADJ; i++)
	{
		if ((center->adj_list[i] != NULL) && (center->adj_available[i]))
		{
			adj[i] = center->adj_list[i];
		}
		else
			adj[i] = NULL;
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
		pushTile(adj, x - 1, y - 1, &size, false);
	if ((x > 0) && (y < (height - 1))) //bottom left
		pushTile(adj, x - 1, y + 1, &size, false);
	if ((x < (width - 1)) && (y > 0)) //top right
		pushTile(adj, x + 1, y - 1, &size, false);
	if ((x < (width - 1)) && (y < (height - 1))) //bottom right
		pushTile(adj, x + 1, y + 1, &size, false);
	if (y > 0) //top center
		pushTile(adj, x, y - 1, &size, false);
	if (y < (height - 1)) //bottom center
		pushTile(adj, x, y + 1, &size, false);
	if (x > 0) //left center
		pushTile(adj, x - 1, y, &size, false);
	if (x < (width - 1)) //right center
		pushTile(adj, x + 1, y, &size, false);

	return size;
}

/*Private: Helper function for get_adj. Only assigns the tile if it is unused. Increments counts.*/
void Board::pushTile(Tile **target, int x, int y, int *count, bool all)
{
	if ((!grid[x+y*width].used) || all)
		target[(*count)++] = &grid[x+y*width];
}


/*Get adjacent tile letters which are not used. Origin is top left.
Inputs: (x,y) coordinate of center tile.
Pointer to adjacency list to fill. MUST BE ABLE TO ACCOMODATE 8 LETTERS.
Returns indicates size of filled adjacenct list.*/
int Board::getAllAdj(Tile* center, Tile **adj)
{
	int size = 0;
	int x = center->x;
	int y = center->y;
	if ((x > 0) && (y > 0)) //top left
		pushTile(adj, x - 1, y - 1, &size, true);
	if ((x > 0) && (y < (height - 1))) //bottom left
		pushTile(adj, x - 1, y + 1, &size, true);
	if ((x < (width - 1)) && (y > 0)) //top right
		pushTile(adj, x + 1, y - 1, &size, true);
	if ((x < (width - 1)) && (y < (height - 1))) //bottom right
		pushTile(adj, x + 1, y + 1, &size, true);
	if (y > 0) //top center
		pushTile(adj, x, y - 1, &size, true);
	if (y < (height - 1)) //bottom center
		pushTile(adj, x, y + 1, &size, true);
	if (x > 0) //left center
		pushTile(adj, x - 1, y, &size, true);
	if (x < (width - 1)) //right center
		pushTile(adj, x + 1, y, &size, true);

	return size;
}

/* Prints the boggle board*/
void Board::printBoard()
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
			printf("%c ",grid[x+y*width].letter);
		printf("\n");
	}
}

/* Prints the used tiles as 1, otherwise 0*/
void Board::printUsed()
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
			printf("%c ", ((grid[x + y*width].used) ? grid[x + y*width].letter : ' '));
		printf("\n");
	}
}

/*Getter for tile.*/
Tile* Board::getTile(int x, int y)
{
	return &grid[x+y*width];
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
		{
			grid[x + y*width].used = false;
			for (int i = 0; i < NUM_ADJ; i++)
				grid[x + y*width].adj_available[i] = true;
		}
	}
}

/*Checks if the Tile is in the list of Tiles */
bool checkTileList(Tile *check, Tile **list, int size)
{
	for (int i = 0; i < size; i++)
	{
		if (check == list[i])
			return true;
	}
	return false;
}

/* Returns the number of tiles with this letter. */
int Board::getLetterCount(char c)
{
	int count = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (grid[x + y*width].letter == c)
				count++;
		}
	}
	return count;
}

/*Returns all tiles in the board that match a given letter. 
  Should call getLetterCount before hand to get size.*/
void Board::getTilesByLetter(Tile *all_tiles[], char c)
{
	int i = 0;
	for (int x = 0; x < height*width; x++)
	{
		if (grid[x].letter == c)
			all_tiles[i++] = &grid[x];
	}
}