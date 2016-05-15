#ifndef BOGGLE_MAIN_H
#define BOGGLE_MAIN_H

#include <string>
#include "boggle_env.h"
#include "trie.h"

using namespace std;

/* --- Functions --- */
void initTiming();
double preciseClock();
void readFile(string dictionary[], int size, string filename);
int singleSolve(string dictionary[], int dict_size, Board *board);
bool recursiveFind(string word, int char_idx, Board *board, Tile* curr_tile);
int prefixSolve(Trie *prefix, Board *board);
void prefixTraversal(Trie *prefix, Node *curr_node, Board *board, Tile *curr_tile, int *word_cnt, char word[], int char_idx);
#endif