# gpu_boggle

An efficient GPU based Boggle solver. 

Boggle is a word game whose purpose it is to find all possible words which can be formed by travelling a given grid of letters. By using GPUs, I leveraged the parallelizable nature of the problem in order to provide substantial speed ups over equivalent CPU algorithms.

## Installation/Usage Instructions

Please read this whole section before attempting to run the program.
Make sure you have NVIDIA CUDA setup on your computer. [Setup Guide](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/#axzz4JnkbbzUr).

To use the boggle solving program on Windows perform the following steps:

1.	Unzip gpu_boggle.zip.
2.	Rename make.txt to make.cmd. (This was necessary to send the program by e-mail).
3.	Open the command line in the gpu_boggle folder.
4.	Run make.cmd in the command line.
5.	Run boggle_main.exe <board width> <board height> in the command line.

    <board_width> is the desired board width.
    <board_height> is the desired board height.

This will by default run 10 boggle solving iterations and print the running time of each algorithm. To see the words found by each algorithm do the following:

1.	In boggle_main.cc set VERBOSE to 1. (So that the CPU algorithms prints words found)
2.	In boggle_gpu.cu set VERBOSE to 1. (So that the GPU algorithm prints words found)
3.	Reduce the number of trials in boggle_main.cc. The variable is called num_trials and can be found on line 62. For VERBOSE tests I suggest setting this to 1.
4.	Run make.cmd again.
5.	Run boggle_main.exe <board width> <board height>.

Note that the program will print the randomly generated board, each word found for each algorithm, the total number of words found for each algorithm and the running time. It will also print if any errors were encountered by the GPU algorithm. Due to limitations in GPU memory it is recommended that the board width and height are kept relatively small, no more than a side length of 10; this limit will of course vary depending on the host system. More explanation on this is provided in my [final report](final_report.pdf).

If you desire to change the number of threads or blocks per threads, the variables can be found in boggle_main.cc and are called MAX_BLOCKS and THREADS_PER_BLOCK respectively. Once again, due to memory limitations on the GPU, if the threads per block are increased above 10 then the GPU solver may fail for larger board sizes; the exact number will vary depending on the host machine. On the author’s machine a 12x12 Boggle grid could be run using 5 threads per block.
