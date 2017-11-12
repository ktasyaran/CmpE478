# CmpE478Homeworks

In HW1, program is expected to find very large given numbers from 2 to given large number using CREW PRAM algorithm in openmp. 
But program  works as follows: prime numbers from 2 to sqrt(n) is found sequential and remaining ones are found in parallel using numbers found in sequential part. There are two result files: .txt file is for holding ordered prime numbers and .csv file is for holding performance metrics with given thread numbers. Txt file is created in /tmp directory and .csv file is created in current user's home directory. 
To run this code, use this command: gcc -fopenmp parallelPrimeFinder.c -lm && ./a.out N
