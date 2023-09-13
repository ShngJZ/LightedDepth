import torch
import GPUEPM

def gpuepm_function(kpts0, kpts1, ransac_iter=5, ransac_threshold=0.0001, num_test_chirality=10, topk=1):
    '''
    Function follow the convention of OpenCV - cv2.findEssentialMat

    Args:
        kpts0: Source Frame 2D Normalized Coordinates, i.e., inv(intrinsic_source) @ points2D
        kpts1: Target (Support) Frame 2D Normalized Coordinates, i.e., inv(intrinsic_target) @ points2D
        ransac_iter: Local RANSAC Iteration, Global Iteration is Fixed as 512 x ransac_iter
        ransac_threshold: RANSAC threshold
        num_test_chirality: Five Point Algorithm Chirality Check Points

    Returns:
        R: Rotation Matrix
        t: Normalized Translation Vector
        inliers: indicator vector suggest inlier ---- 1: inlier, 0: outlier. All Zeros suggest failure cases
    '''
    ransac_test_points = kpts0.shape[0]
    # Homogeneous Coordinates will Cause Unspecified Behavior
    assert ransac_test_points > 5
    assert (kpts0.shape[1] == 2) and (kpts1.shape[1] == 2)
    # GPU Scoring Function
    kpts0, kpts1 = kpts0.double().contiguous(), kpts1.double().contiguous()
    essential_matrices, projection_matrices, inliers_count, inliers_indicator = GPUEPM.gpu_ep_ransac(
        kpts0,
        kpts1,
        num_test_chirality,
        ransac_test_points,
        ransac_iter,
        ransac_threshold
    )

    # Sorting
    _, from_good2bad = torch.sort(-inliers_count)
    best_ids = from_good2bad[0] if topk == 1 else from_good2bad[0:topk]
    pose, inliers = projection_matrices[best_ids], inliers_indicator[best_ids]
    R, t = torch.split(pose, [3, 1], dim=-1)
    return R.float(), t.float(), inliers


"""
# This is a Archived Code to Reproduce Scoring Funciton in Cuda Code
def compute_inliers(kpts0, kpts1, em, ransac_threshold):
    '''
    A python implementation of the Scoring Function, for debugging purposes
    '''
    
    import einops
    import kornia
    import numpy as np
    
    npts, _ = kpts0.shape
    kpts0torch = kornia.geometry.conversions.convert_points_to_homogeneous(kpts0)
    kpts1torch = kornia.geometry.conversions.convert_points_to_homogeneous(kpts1)

    topk, ransac_iterations = em.shape[0:2]
    em = einops.rearrange(em, 't r h w -> (t r) h w')
    topk_batch = int(topk * ransac_iterations)

    ex = em.view([topk_batch, 1, 3, 3]) @ kpts0torch.view([1, npts, 3, 1])
    xe = kpts1torch.view([1, npts, 1, 3]) @ em.view([topk_batch, 1, 3, 3])
    ex1, ex2, _ = torch.split(ex, 1, dim=2)
    xe1, xe2, _ = torch.split(xe, 1, dim=3)
    d = torch.sqrt(ex1 ** 2 + ex2 ** 2 + xe1 ** 2 + xe2 ** 2 + 1e-10)
    xex = kpts1torch.view([1, npts, 1, 3]) @ em.view([topk_batch, 1, 3, 3]) @ kpts0torch.view([1, npts, 3, 1])
    error = xex / d
    error = error.view([topk_batch, npts]).abs()
    inliers = (error > 0) * (error < ransac_threshold)
    inliers = inliers.sum(axis=1)
    inliers = einops.rearrange(inliers, '(t r) -> t r', t=topk, r=ransac_iterations)
    inliers, ransac_iter_max_index = torch.max(inliers, dim=1, keepdim=False)
    ransac_batch_max_index = torch.argmax(inliers)

    return inliers, ransac_batch_max_index.item(), ransac_iter_max_index[ransac_batch_max_index].item()
"""