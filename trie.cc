/*
Trie Class
Miguel Aroca-Ouellette
05/14/2016
*/

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "trie.h"

/*---- Node methods ----*/

Node::Node(char _letter, bool _end_word, Node* _parent)
{
	letter = _letter;
	end_word = _end_word;
	used = false;
	parent = _parent;
}


/*Checks if child exists. If so, then it returs pointer to child.*/
Node* Node::isChild(char letter)
{
	for (int i = 0; i < children.size(); i++)
	{
		if (children[i]->letter == letter)
			return children[i];
	}
	return NULL;
}

void Node::addChild(Node* new_child)
{
	children.push_back(new_child);
}

/*---- Trie methods ----*/

Trie::Trie()
{
	root = new Node(0, false, NULL);
}

void Trie :: addWord(string word)
{
	Node *curr_node = root;
	for (int i = 0; i < word.length();i++)
	{
		Node *next_node = curr_node->isChild(word[i]);
		if (next_node == NULL)
		{
			Node* new_node = new Node(word[i], false, curr_node);
			curr_node->addChild(new_node);
			curr_node = new_node;
		}
		else
			curr_node = next_node;
	}
	curr_node->setEndWord(true);
}

bool Trie::searchWord(string word)
{
	Node *curr_node = root;
	for (int i = 0; i < word.length(); i++)
	{
		Node *next_node = curr_node->isChild(word[i]);
		if (next_node == NULL)
		{
			return false;
		}
		else
			curr_node = next_node;
	}
	//true if end of word
	if (curr_node->isEndWord())
		return true;
	else
		return false;
}

void Trie::printTrie()
{
	deque<Node *> to_print;
	to_print.push_back(root);
	while (to_print.size()>0)
	{
		Node *curr_node = to_print.front();
		to_print.pop_front();
		cout << "P: " << curr_node->getLetter() << "\t C: ";
		for (int i = 0; i < curr_node->numChildren(); i++)
		{
			to_print.push_back(curr_node->getChild(i));
			cout << curr_node->getChild(i)->getLetter() << " ";
		}
		cout << "\n";
	}
}

void Trie::buildFromDict(string dict[], int length)
{
	for (int i = 0; i < length; i++)
		addWord(dict[i]);
}

//TODO: Remove later
/*Reads frome file into string array.*/
/*
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

int main(int argc, char** argv)
{
	Trie myTrie;
	string dictionary[109583];
	read_file(dictionary, 109583, "wordsEn.txt");

	myTrie.buildFromDict(dictionary, 109583);

	cout << (myTrie.searchWord("yukk")?"true":"false") << "\n";
	cout << (myTrie.searchWord("yukked") ? "true" : "false") << "\n";
	//myTrie.printTrie();
	return 1;
}*/