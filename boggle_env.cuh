#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#ifndef BOGGLE_ENV_H
#define BOGGLE_ENV_H

/* --- Boggle Environment header --- */

#include <string>
#include <cuda_runtime.h>
using namespace std;

#define NUM_ADJ 8 //tiles can have at most 8 adjacent tiles

/*Letter Tile struct*/
struct Tile
{
	int x;
	int y;
	char letter;
	bool used;
	bool adj_available[NUM_ADJ];	//list which denotes if adjacency available
	Tile* adj_list[NUM_ADJ];			//list which holds adjacencies
};

/*Boggle Board Class.*/
class Board {
	int width, height;
	CUDA_CALLABLE_MEMBER void pushTile(Tile **target, int x, int y, int *count, bool all);
public:
	CUDA_CALLABLE_MEMBER Board(int _width, int _height);
	CUDA_CALLABLE_MEMBER Board(const Board &obj);
	CUDA_CALLABLE_MEMBER ~Board();
	void genRandLetters();
	void setLetters(string letters);
	CUDA_CALLABLE_MEMBER int getAdj(Tile* center, Tile **adj);
	CUDA_CALLABLE_MEMBER void getAdjList(Tile *center, Tile **adj);
	CUDA_CALLABLE_MEMBER int getAllAdj(Tile* center, Tile **adj);
	CUDA_CALLABLE_MEMBER Tile* getTile(int x, int y);
	CUDA_CALLABLE_MEMBER void getAllTiles(Tile *all_tiles[]);
	CUDA_CALLABLE_MEMBER int getNumTiles() { return width*height; }
	int getHeight() { return height; }
	int getWidth() { return width; }
	CUDA_CALLABLE_MEMBER void resetBoard();
	CUDA_CALLABLE_MEMBER void printBoard();
	CUDA_CALLABLE_MEMBER void printUsed();
	Tile *grid;					  //necessary for cuda memory allocation
};

CUDA_CALLABLE_MEMBER bool checkTileList(Tile *check, Tile **list, int size); //returns true if in list
#endif