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
int singleSolve(char **dictionary, int dict_size, Board *board);
bool recursiveFind(char *word, int length, int char_idx, Board *board, Tile* curr_tile);
int prefixSolve(Trie *prefix, Board *board);
void prefixTraversal(Trie *prefix, Node *curr_node, Board *board, Tile *curr_tile, int *word_cnt, char word[], int char_idx);
void single_gpu(char **dict, int size, int max_word_len, Board *board);

#endif