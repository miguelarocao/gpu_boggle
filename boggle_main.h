#ifndef BOGGLE_MAIN_H
#define BOGGLE_MAIN_H

#include <string>
#include "boggle_env.cuh"
#include "trie.h"

using namespace std;

/* --- Functions --- */
void initTiming();
double preciseClock();
int wordLength(char *word);
void readFile(char **dictionary, string filename);
float singleSolve(char **dictionary, int dict_size, Board *board);
bool recursiveFind(char *word, int length, int char_idx, Board *board, Tile* curr_tile);
float prefixSolve(Trie *prefix, Board *board);
void prefixTraversal(Trie *prefix, Node *curr_node, Board *board, Tile *curr_tile, int *word_cnt, char word[], int char_idx);
float single_gpu(char **dict, int size, int max_word_len, Board *board);
float single_gpu(char *dev_dict, Board *board, Board *dev_board, Tile *dev_grid, int *dev_word_count);

#endif