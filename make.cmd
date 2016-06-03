@echo off

nvcc -arch=sm_20 -c boggle_env.cu  boggle_gpu.cu -dc
nvcc -arch=sm_20 -c trie.cc
nvcc -arch=sm_20 -c boggle_main.cc
nvcc -arch=sm_20 boggle_env.obj boggle_gpu.obj trie.obj boggle_main.obj -o boggle_main

IF EXIST boggle_main.lib. (
    del boggle_main.lib
)
IF EXIST boggle_main.exp. (
    del boggle_main.exp
)