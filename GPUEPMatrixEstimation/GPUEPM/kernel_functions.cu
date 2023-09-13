/*
 * Kernel Functions
 */
#include <curand_kernel.h>
#include <math.h> // truncf
#include <stdio.h>

#include "sturm.cu"
#include "cheirality.cu"
#include "essential_matrix_5pt.cu"

// Declare constant memory (64KB maximum)
__constant__ int c_num_points;
__constant__ int c_num_test_chirality;
__constant__ int c_ransac_num_test_points;
__constant__ int c_ransac_num_iterations;
__constant__ double c_inlier_threshold;

// Function declarations
__device__ int RandomInt(curandState* state,
                         const int global_index,
                         const int min_int,
                         const int max_int);

template<int n>
__device__ void SelectSubset(const double *qs,
                             const double *qps,
                             curandState* state,
                             const int global_index,
                             Matches_n<n>& q,
                             Matches_n<n>& qp);

template<typename T>
__device__ void ComputeError(const T *q,
                             const T *qp,
                             const Ematrix &E,
                             T &error);

template<typename T>
__device__ void InitZeros2D(const T array,
                            const int height,
                            const int width);

template<typename T>
__device__ void InitZeros1D(const T array,
                            const int len);

/*
 * Initialise a state for each thread
 */
__global__ void SetupRandomState(const unsigned long long seed, curandState* state) {
  int global_index = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(seed, global_index, 0, &state[global_index]);
}

/*
 * Estimate the essential matrix and camera matrix, using the 5-point algorithm and RANSAC
 */
template <int subset_size>
__global__ void EstimateProjectionMatrix(const double *qs, // Two sets of matching points
                                        const double *qps, // (flattened 2D arrays)
                                        curandState* state, // Random number generator state
                                        int *num_inliers, // Number of inliers per thread
                                        double (*essential_matrices)[3][3], // Essential matrices with greatest number of inliers
                                        double (*projection_matrices)[3][4], // camera matrices with greates number of inliers
                                        bool (*inlier_indicator_tmp), // Tmp Recording of Inlier Indicator
                                        bool (*inlier_indicator) // Output Inlier Indicator
                                        ){

  int global_index = threadIdx.x + blockDim.x * blockIdx.x;

  int num_essential_matrices;
  Matches_n<subset_size> q, qp;
  Ematrix essential_matrix;
  Pmatrix projection_matrix;
  Ematrix essential_matrix_set[10];

  // Initialization
  num_inliers[global_index] = 0;
  InitZeros2D(essential_matrices[global_index], 3, 3);
  InitZeros2D(projection_matrices[global_index], 3, 4);
  InitZeros1D(&inlier_indicator[global_index * c_ransac_num_test_points], c_ransac_num_test_points);

  // RANSAC
  int best_num_inliers = 0;
  for (int i = 0; i < c_ransac_num_iterations; ++i) {
    // Generate hyposthesis set
    SelectSubset<subset_size>(qs, qps, state, global_index, q, qp);

    // Compute up to 10 essential matrices using the 5 point algorithm
    compute_E_matrices_optimized(q, qp, essential_matrix_set, num_essential_matrices);

    // Find those that give correct cheirality
    int num_projection_matrices = num_essential_matrices;
    Pmatrix projection_matrix_set[10];
    compute_P_matrices(q, qp, essential_matrix_set, (double *)0, projection_matrix_set, num_projection_matrices, subset_size);

    if (num_projection_matrices == 0){
        continue;
    }

    // Test essential matrices in solution set
    // Choose the solution with the greatest number of inliers
    int best_num_inliers_subset = 0;
    int best_index = 0;
    for (int j = 0; j < num_projection_matrices; ++j) {
      int inlier_count = 0;
      for (int k = 0; k < c_num_test_chirality; ++k) {
        double error;
        double q_test[3] = {qs[2 * k], qs[2 * k + 1], 1.0};
        double qp_test[3] = {qps[2 * k], qps[2 * k + 1], 1.0};
        ComputeError<double>(q_test, qp_test, essential_matrix_set[j], error);
        if (error <= c_inlier_threshold) {
          inlier_count++;
        }
      }
      if (inlier_count > best_num_inliers_subset) {
        best_num_inliers_subset = inlier_count;
        best_index = j;
      }
    }

    // Evaluate best essential matrix on full set of test matches
    int inlier_count = 0;
    for (int k = 0; k < c_ransac_num_test_points; ++k) {
      double error;
      double q_test[3] = {qs[2 * k], qs[2 * k + 1], 1.0};
      double qp_test[3] = {qps[2 * k], qps[2 * k + 1], 1.0};
      ComputeError<double>(q_test, qp_test, essential_matrix_set[best_index], error);
      if (error <= c_inlier_threshold) {
        inlier_count++;
        inlier_indicator_tmp[global_index * c_ransac_num_test_points + k] = 1;
      }
      else {
        inlier_indicator_tmp[global_index * c_ransac_num_test_points + k] = 0;
      }
    }
    if (inlier_count > best_num_inliers) {
      best_num_inliers = inlier_count;
      memcpy(essential_matrix, &essential_matrix_set[best_index], 3 * 3 * sizeof(double));
      memcpy(projection_matrix, &projection_matrix_set[best_index], 3 * 4 * sizeof(double));
      memcpy(&inlier_indicator[global_index * c_ransac_num_test_points], &inlier_indicator_tmp[global_index * c_ransac_num_test_points], c_ransac_num_test_points * sizeof(bool));
    }
  }

  // Copy to output
  num_inliers[global_index] = best_num_inliers;
  memcpy(&essential_matrices[global_index], essential_matrix, 3 * 3 * sizeof(double));
  memcpy(&projection_matrices[global_index], projection_matrix, 3 * 4 * sizeof(double));

}

/*
 * Compute Sampson distance given a pair of matched points and an essential matrix
 */
template<typename T>
__device__ void ComputeError(const T *q,
                             const T *qp,
                             const Ematrix &E,
                             T &error) {
  // Compute Ex
  T Ex[3];
  for (int k = 0; k < 3; k++) {
    T sum = 0.0;
    for (int l = 0; l < 3; l++) {
      sum += E[k][l] * q[l];
    Ex[k] = sum;
    }
  }
  // Compute x^TE
  T xE[3];
  for (int k = 0; k < 3; k++) {
    T sum = 0.0;
    for (int l = 0; l < 3; l++) {
      sum += qp[l] * E[l][k];
    xE[k] = sum;
    }
  }
  // Compute xEx
  T xEx = 0.0;
  for (int k = 0; k < 3; k++) {
    xEx += qp[k] * Ex[k];
  }
  // Compute Sampson error
  T d = sqrt(Ex[0]*Ex[0]+Ex[1]*Ex[1]+xE[0]*xE[0]+xE[1]*xE[1]);
  error = xEx / d;

  if (error < 0.0) error = -error;
}

/*
 * Generate an integer in the range [min_int, max_int]
 */
__device__ int RandomInt(curandState* state,
                         const int global_index,
                         const int min_int,
                         const int max_int) {
  // Generate a random float in (0,1)
  float random_float = curand_uniform(&state[global_index]);
  random_float *= (max_int - min_int + 0.999999f);
  random_float += min_int;
  return (int) truncf(random_float);
}

/*
 * Generate a random subset of qs and qps, each thread selects a different subset
 * Optimised for speed, no checking that elements are unique
 */
template<int n>
__device__ void SelectSubset(const double* qs,
                             const double* qps,
                             curandState* state,
                             const int global_index,
                             Matches_n<n>& q,
                             Matches_n<n>& qp) {
  for (int i = 0; i < n; ++i) {
    int index = RandomInt(state, global_index, 0, c_num_points - 1);
    for (int j = 0; j < 2; ++j) {
      q[i][j] = qs[2 * index + j];
      qp[i][j] = qps[2 * index + j];
    }
    q[i][2] = 1.0;
    qp[i][2] = 1.0;
  }
}


/*
 * Initialize Zeros to Array
 */
template<typename T>
__device__ void InitZeros2D(const T array,
                            const int height,
                            const int width) {
  for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            array[i][j] = 0;
        }
  }
}


template<typename T>
__device__ void InitZeros1D(const T array,
                            const int len) {
  for (int i = 0; i < len; ++i) {
        array[i] = 0;
  }
}


__global__ void VoteForOptimalScaleGPUKernel(
         int* voting_vector,
         int* minfillidx,
         const int* maxfillidx,
         const int* valid,
         const int voting_vector_len,
         const int topk,
         const int npts
         ) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    for (; global_index < topk * npts; global_index += blockDim.x * gridDim.x){
        int idx_topk = global_index / npts;
        int idx_vote = global_index - idx_topk * npts;

        if (valid[global_index] == 1){
            for (int i = minfillidx[global_index]; i <= maxfillidx[global_index]; ++i){
                atomicAdd(&voting_vector[idx_topk * voting_vector_len + i], 1);
            }
        }
    }
}
