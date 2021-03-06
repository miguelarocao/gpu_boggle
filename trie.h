#ifndef TRIE_H
#define TRIE_H

/* --- Trie header --- */
#include <string>
#include <cassert>
#include <vector>
#include <deque> //only used for printing!

using namespace std;

class Node
{
	char letter;
	bool end_word;
	bool used; //indicates that word has already been printed
	Node *parent;
	vector<Node*> children;
public:
	Node(char, bool, Node*);
	void addChild(Node*);
	Node* isChild(char);
	Node* getChild(int i) { assert(i<children.size()); return children[i]; }
	char getLetter() { return letter; }
	void setEndWord(bool end) { end_word = end; }
	bool isEndWord() { return end_word; }
	bool isUsed() { return used; }
	void setUsed(bool _used) { used = _used; }
	int numChildren() { return children.size(); }
	vector<Node*> *getChildren() { return &children; }
};

class Trie
{
	Node* root;
public:
	Trie();
	void addWord(char *word);
	bool searchWord(char *word);
	void printTrie();
	Node* getRoot() { return root; }
	Node* getChild(Node* node, char letter) { return node->isChild(letter); } //returns NULL if doesn't exist
	void buildFromDict(char **dict, int length);
};

#endif