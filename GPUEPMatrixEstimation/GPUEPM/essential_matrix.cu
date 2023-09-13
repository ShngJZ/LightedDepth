#include <vector>
#include <iostream>
#include <chrono>
#include <tuple>
#include <ATen/ATen.h>

#include "common.h"
#include "kernel_functions.cu"

/*
 * Five point algorithm cuda initialization with TopK
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> EssentialProjectionRansacGPU(
        at::Tensor input1, // input 1 has size nx2, type double
        at::Tensor input2,
        const int num_test_chirality, // 10
        const int num_ransac_test_points, // 1000
        const int num_ransac_iterations, // number of iterations to run RANSAC
        const double inlier_threshold) {

    const int num_points = input1.size(0);

    const int num_threads_per_block = 64;
    const int num_blocks = 8;
    const int num_threads = num_blocks * num_threads_per_block;

    cudaSetDevice(input1.get_device());

    // Input data pointer (on GPU)
    double *input1_ptr = input1.data_ptr<double>();
    double *input2_ptr = input2.data_ptr<double>();

    int *num_inliers;
    double (*essential_matrices)[3][3];
    double (*projection_matrices)[3][4];
    curandState* state;

    CudaErrorCheck(cudaMallocManaged((void **) &num_inliers, num_threads * sizeof(int)));
    CudaErrorCheck(cudaMallocManaged((void **) &essential_matrices, num_threads * 3 * 3 * sizeof(double)));
    CudaErrorCheck(cudaMallocManaged((void **) &projection_matrices, num_threads * 3 * 4 * sizeof(double)));
    CudaErrorCheck(cudaMallocManaged((void **) &state, num_threads * sizeof(curandState)));

    // Tmp Recording of Inlier Indicator
    bool (*inlier_indicator_tmp);
    CudaErrorCheck(cudaMallocManaged((void **) &inlier_indicator_tmp, num_threads * num_ransac_test_points * sizeof(bool)));
    // Output Inlier Indicator
    bool (*inlier_indicator);
    CudaErrorCheck(cudaMallocManaged((void **) &inlier_indicator, num_threads * num_ransac_test_points * sizeof(bool)));

    // Copy constants to device constant memory
    CudaErrorCheck(cudaMemcpyToSymbol(c_num_points, &num_points, sizeof(int)));
    CudaErrorCheck(cudaMemcpyToSymbol(c_num_test_chirality, &num_test_chirality, sizeof(int)));
    CudaErrorCheck(cudaMemcpyToSymbol(c_ransac_num_test_points, &num_ransac_test_points, sizeof(int)));
    CudaErrorCheck(cudaMemcpyToSymbol(c_ransac_num_iterations, &num_ransac_iterations, sizeof(int)));
    CudaErrorCheck(cudaMemcpyToSymbol(c_inlier_threshold, &inlier_threshold, sizeof(double)));

    // Generate random states, one for each thread
    SetupRandomState<<<num_blocks, num_threads_per_block>>>(seed, state);

    EstimateProjectionMatrix<subset_size><<<num_blocks, num_threads_per_block>>>(
                                               input1_ptr, // Two sets of matching points
                                               input2_ptr, // (flattened 2D arrays)
                                               state,      // Random number generator state
                                               num_inliers, // Number of inliers per thread
                                               essential_matrices,
                                               projection_matrices, // Essential matrices per thread
                                               inlier_indicator_tmp, // Tmp Recording of Inlier Indicator
                                               inlier_indicator // Output Inlier Indicator
                                               );

    CudaErrorCheck(cudaPeekAtLastError()); // Check for kernel launch error
    CudaErrorCheck(cudaDeviceSynchronize()); // Check for kernel execution error

    // Copy to output
    auto num_inliers_out_options = at::TensorOptions().dtype(at::kInt).device(at::kCUDA, input1.device().index());
    at::Tensor num_inliers_out = at::empty(num_threads, num_inliers_out_options);
    at::Tensor essential_matrices_out = at::empty(num_threads * 3 * 3, input1.options());
    at::Tensor projection_matrices_out = at::empty(num_threads * 3 * 4, input1.options());
    auto inlier_indicator_options = at::TensorOptions().dtype(at::kBool).device(at::kCUDA, input1.device().index());
    at::Tensor inlier_indicator_out = at::empty(num_threads * num_ransac_test_points, inlier_indicator_options);

    int* dataptr1 = num_inliers_out.data_ptr<int>();
    double* dataptr2 = essential_matrices_out.data_ptr<double>();
    double* dataptr3 = projection_matrices_out.data_ptr<double>();
    bool* dataptr4 = inlier_indicator_out.data_ptr<bool>();

    CudaErrorCheck(cudaMemcpy(dataptr1, &num_inliers[0], num_threads * sizeof(int), cudaMemcpyDeviceToDevice));
    CudaErrorCheck(cudaMemcpy(dataptr2, &essential_matrices[0], num_threads * 3 * 3 * sizeof(double), cudaMemcpyDeviceToDevice));
    CudaErrorCheck(cudaMemcpy(dataptr3, &projection_matrices[0], num_threads * 3 * 4 * sizeof(double), cudaMemcpyDeviceToDevice));
    CudaErrorCheck(cudaMemcpy(dataptr4, &inlier_indicator[0], num_threads * num_ransac_test_points * sizeof(bool), cudaMemcpyDeviceToDevice));

    essential_matrices_out.resize_({num_threads, 3, 3});
    projection_matrices_out.resize_({num_threads, 3, 4});
    inlier_indicator_out.resize_({num_threads, num_ransac_test_points});

    CudaErrorCheck(cudaFree(num_inliers));
    CudaErrorCheck(cudaFree(essential_matrices));
    CudaErrorCheck(cudaFree(projection_matrices));
    CudaErrorCheck(cudaFree(inlier_indicator_tmp));
    CudaErrorCheck(cudaFree(inlier_indicator));
    CudaErrorCheck(cudaFree(state));

    auto t = std::make_tuple(essential_matrices_out, projection_matrices_out, num_inliers_out, inlier_indicator_out);

    return t;
}



void VoteForOptimalScaleGPU(
    at::Tensor voting_vector,
    at::Tensor minfillidx,
    at::Tensor maxfillidx,
    at::Tensor valid,
    const int voting_vector_len,
    const int topk,
    const int npts
    ){

    const int num_threads_per_block = 64;
    const int num_blocks = 8;
    const int num_threads = num_blocks * num_threads_per_block;

    int* voting_vector_ = voting_vector.data_ptr<int>();
    int* minfillidx_ = minfillidx.data_ptr<int>();
    int* maxfillidx_ = maxfillidx.data_ptr<int>();
    int* valid_ = valid.data_ptr<int>();


    VoteForOptimalScaleGPUKernel<<<num_blocks, num_threads_per_block>>>(
           voting_vector_,
           minfillidx_,
           maxfillidx_,
           valid_,
           voting_vector_len,
           topk,
           npts
           );

    }