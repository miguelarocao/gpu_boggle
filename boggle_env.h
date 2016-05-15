#ifndef BOGGLE_ENV_H
#define BOGGLE_ENV_H

/* --- Boggle Environment header --- */

#include <string>
using namespace std;

/*Letter Tile struct*/
struct Tile
{
	int x;
	int y;
	char letter;
	bool used;
};

/*Boggle Board Class.*/
class Board {
	int width, height;
	Tile **grid;
	void pushTile(Tile **target, int x, int y, int *count);
public:
	Board(int _width, int _height);
	~Board();
	void genRandLetters();
	void setLetters(string letters);
	int getAdj(Tile* center, Tile **adj);
	Tile* getTile(int x, int y);
	void getAllTiles(Tile *all_tiles[]);
	int getNumTiles() { return width*height; }
	void resetBoard();
	void printBoard();
	void printUsed();
};

#endif