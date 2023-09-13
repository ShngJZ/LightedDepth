#ifndef _HIDDEN_6_H_
#define _HIDDEN_6_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Dimensions of the matrices that we will be using
const int Maxdegree = 20; // Big enough for 5 or 6 point problem

// Basic Data Types
typedef double Matches[][3];
template <int n>
using Matches_n = double[n][3];

typedef double Ematrix[3][3];
typedef double Pmatrix[3][4];

typedef double EquationSet[5][10][10];

// For holding polynomials of matrices
typedef double Polynomial[Maxdegree+1];


/*
 * CUDA macros, constants and functions
 */
const int subset_size = 5;
const unsigned long long seed = 1234;

#define CudaErrorCheck(ans) {__CudaErrorCheck((ans), __FILE__, __LINE__);}
void __CudaErrorCheck(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    std::cout << "CUDA Error (" << file << ":" << line << "): "
	      << cudaGetErrorString(code) << std::endl;
    exit(code);
  }
}

#endif

