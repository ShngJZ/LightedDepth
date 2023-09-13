#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <iostream>

// CUDA forward declarations

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> EssentialProjectionRansacGPU(
    at::Tensor input1, // point set 1, dimension nx2
    at::Tensor input2, // point set 2, dimension nx2
    const int num_test_chirality, // 10
    const int num_ransac_test_points, // 1000
    const int num_ransac_iterations, // number of iterations to run RANSAC
    const double inlier_threshold);


void VoteForOptimalScaleGPU(
    at::Tensor voting_vector,
    at::Tensor minfillidx,
    at::Tensor maxfillidx,
    at::Tensor valid,
    const int voting_vector_len,
    const int topk,
    const int npts
    );


// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x, " must be a CUDA tensor")
#define CHECK_DOUBLE(x) TORCH_CHECK(x.scalar_type()==at::ScalarType::Double, #x, " must be a double tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous")
#define CHECK_INPUT_INIT(x) CHECK_CUDA(x); CHECK_DOUBLE(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_OPT(x) CHECK_DOUBLE(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_INIT_VOTE(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> EssentialProjectionRansac(
    at::Tensor input1,
    at::Tensor input2,
    const int num_test_chirality,
    const int num_ransac_test_points,
    const int num_ransac_iterations,
    const double inlier_threshold) {

    CHECK_INPUT_INIT(input1);
    CHECK_INPUT_INIT(input2);

    return EssentialProjectionRansacGPU(input1, input2, num_test_chirality, num_ransac_test_points, num_ransac_iterations, inlier_threshold);
}

void VoteForOptimalScale(
    at::Tensor voting_vector,
    at::Tensor minfillidx,
    at::Tensor maxfillidx,
    at::Tensor valid,
    const int voting_vector_len,
    const int topk,
    const int npts
    ) {

    CHECK_INPUT_INIT_VOTE(voting_vector);
    CHECK_INPUT_INIT_VOTE(minfillidx);
    CHECK_INPUT_INIT_VOTE(maxfillidx);
    CHECK_INPUT_INIT_VOTE(valid);

    return VoteForOptimalScaleGPU(voting_vector, minfillidx, maxfillidx, valid, voting_vector_len, topk, npts);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gpu_ep_ransac", &EssentialProjectionRansac, "Compute E and P matrix using RANSAC-5pt");
    m.def("vote_for_optimal_scale", &VoteForOptimalScale, "Fast GPU Voting Implementation");
}
