#ifndef BOGGLE_MAIN_H
#define BOGGLE_MAIN_H

#include <string>
using namespace std;

/* --- Classes & Structs --- */


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
	Tile **grid;
	void push_tile(Tile **target, int x, int y, int *count);
public:
	int width, height;
	Board(int _width, int _height);
	~Board();
	void set_letters(string letters);
	int get_adj(Tile* center, Tile **adj);
	Tile* get_tile(int x, int y);
	void get_all_tiles(Tile *all_tiles[]);
	void reset();
	void print_board();
	void print_used();
};

/* --- Functions --- */
void read_file(string dictionary[], int size, string filename);
void single_solve(string dictionary[], int dict_size, Board *board);
bool recursive_find(string word, int char_idx, Board *board, Tile* curr_tile);

#endif