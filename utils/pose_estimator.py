"""
 Copyright 2023 Shengjie Zhu

 Licensed under the GNU GENERAL PUBLIC LICENSE, Version 3 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.gnu.org/licenses/

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import GPUEPM
import torch, kornia
import numpy as np
import torch.nn as nn

from kornia.geometry.epipolar.essential import cross_product_matrix
from GPUEPMatrixEstimation.gpuepm import gpuepm_function

def padding_pose(poses_w2c):
    B, mh, mw = poses_w2c.shape
    poses_w2c_ = torch.zeros([B, 4, 4], device=poses_w2c.device)
    poses_w2c_[:, 3, 3] = 1.0
    poses_w2c_[:, 0:mh, 0:mw] = poses_w2c
    return poses_w2c_.contiguous()

class PoseEstimator(nn.Module):
    def __init__(
            self,
            npts=10000,
            device=None,
            prj_w=0.05,
            maxscale_vote=5.0,
            precision=0.01,
            ransac_threshold=0.0001,
            scale_th=1,
            topk=5
    ):

        """
        npts: RANSAC Sampled Points
        device: GPU Device Handler
        prj_w: weight of projection inliers (in additional to epp inliers)
        maxscale_vote: max camera movement distance between two frames, share unit with depthmap
        precision: resolution of voting, smaller requires more memory. Performance is insensitive to this parameter once sufficiently large.
        ransac_threshold: threshold in Sampson Error, defined in normalized pixel coordinates
        scale_th: beyond pixel distance scale_th, the projection is no longer considered an inlier in camera scale voting
        topk: top k candidates to apply the additional projection consrtaint
        """

        # Points Sampled
        self.npts, self.device = npts, device

        # Source Pixel Grid for different intrinsic
        self.pts_source = dict()

        # Mask for Sampling
        self.ignore_regeion = dict()

        # Scale Estimate Inlier Threshold
        self.scale_th, self.topk, self.eps = scale_th, topk, 1e-10

        # Assign Precision of Voting
        self.maxscale_vote, self.precision = maxscale_vote, precision

        # Prj Constraint
        self.prj_w = prj_w

        self.ransac_threshold = ransac_threshold

    def generate_voting_vector(self, device):
        voting_vector = torch.zeros([self.topk, int(self.maxscale_vote / self.precision)]).to(device)
        return voting_vector

    def pose_estimation(self, flow, depth, intrinsic, valid_regeion=None, seed=None):
        """
        flow: H x W x 2
        depth: H x W
        intrinsic: 3 x 3
        valid_regeion: H x W
        seed: random seed for reproduction
        """
        assert flow.ndim == 3 and depth.ndim == 2 and intrinsic.ndim == 2
        h, w = depth.shape

        x = torch.arange(w, dtype=torch.long, device=self.device)
        y = torch.arange(h, dtype=torch.long, device=self.device)
        yy, xx = torch.meshgrid(y, x)
        pts_source = torch.stack([xx, yy], dim=-1)

        # Yield Correspondence
        xx_source, yy_source = torch.split(pts_source, 1, dim=-1)
        xx_target, yy_target = torch.split(pts_source + flow, 1, dim=-1)
        xx_source, yy_source, xx_target, yy_target = xx_source.squeeze(), yy_source.squeeze(), xx_target.squeeze(), yy_target.squeeze()

        # Ignore Regeion
        if valid_regeion is None:
            valid_regeion = torch.ones_like(depth)
        valid_regeion = valid_regeion * (xx_target > 0) * (xx_target < w) * (yy_target > 0) * (yy_target < h) * (depth > 0)

        # Seeding for reproduction
        if seed is not None: np.random.seed(seed)

        pts_idx = np.random.randint(0, torch.sum(valid_regeion).item(), self.npts)

        # Sample Correspondence
        xxf_source, yyf_source = xx_source[valid_regeion][pts_idx], yy_source[valid_regeion][pts_idx]
        xxf_target, yyf_target = xx_target[valid_regeion][pts_idx], yy_target[valid_regeion][pts_idx]
        pts_source, pts_target = torch.stack([xxf_source, yyf_source], axis=1).float(), torch.stack([xxf_target, yyf_target], axis=1).float()
        depthf = depth[valid_regeion][pts_idx]

        pts_source, pts_target = kornia.geometry.conversions.convert_points_to_homogeneous(pts_source), kornia.geometry.conversions.convert_points_to_homogeneous(pts_target)
        pts_source_normed = pts_source @ intrinsic.inverse().T
        pts_target_normed = pts_target @ intrinsic.inverse().T

        R, t, inliers = gpuepm_function(
            kornia.geometry.conversions.convert_points_from_homogeneous(pts_source_normed),
            kornia.geometry.conversions.convert_points_from_homogeneous(pts_target_normed), ransac_iter=5, ransac_threshold=self.ransac_threshold, num_test_chirality=10, topk=self.topk
        )

        if self.topk == 1:
            R, t = R.unsqueeze(0), t.unsqueeze(0)

        pose, scale = self.scale_estimation(intrinsic, R, t, pts_source, pts_target, depthf, inliers)
        prj_inliers = self.projection_constraint(pts_source, pts_target, depthf, intrinsic, pose)

        summed_inliers = torch.sum(inliers, 1) + prj_inliers * self.prj_w

        idx_best = torch.argmax(summed_inliers).item()
        pose_best, scale_best = pose[idx_best], scale[idx_best]
        return padding_pose(pose_best.unsqueeze(0)), scale_best

    def projection_constraint(self, pts_source, pts_target, depthf, intrinsic, pose):
        pts_source, pts_target, intrinsic, pose = pts_source.view([1, self.npts, 3]), pts_target.view([1, self.npts, 3]), padding_pose(intrinsic.view([1, 3, 3])), padding_pose(pose)
        pts_target, depthf = kornia.geometry.conversions.convert_points_from_homogeneous(pts_target), depthf.view([1, self.npts, 1])
        prjM = intrinsic @ pose @ intrinsic.inverse()

        pts_source_3D = kornia.geometry.conversions.convert_points_to_homogeneous(pts_source * depthf)
        pts_source_prj = pts_source_3D @ prjM.transpose(-1, -2)
        pts_source_prj = kornia.geometry.conversions.convert_points_from_homogeneous(pts_source_prj)
        pts_source_prj = kornia.geometry.conversions.convert_points_from_homogeneous(pts_source_prj)

        prj_inliers = torch.sqrt(torch.sum((pts_source_prj - pts_target) ** 2 + 1e-10, dim=-1))
        prj_inliers = torch.sum(prj_inliers < self.scale_th, dim=-1)

        return prj_inliers

    def scale_estimation(self, intrinsic, R, t, pts_source, pts_target, depthf, inliers):
        device = intrinsic.device
        pts_source, pts_target, depthf = pts_source.view([1, self.npts, 3]), pts_target.view([1, self.npts, 3]), depthf.view([1, self.npts, 1])
        intrinsic = intrinsic.view([1, 3, 3])

        # Compute Epipolar Line and Epp point
        essential_mtx = cross_product_matrix(t.view([self.topk, 3])) @ R
        fundament_mtx = intrinsic.inverse().transpose(-1, -2) @ essential_mtx @ intrinsic.inverse()
        epp_line = pts_source @ fundament_mtx.transpose(-1, -2)

        epp_point = intrinsic @ t
        epp_point = kornia.geometry.conversions.convert_points_from_homogeneous(epp_point.transpose(-1, -2))

        # Compute Target Projection to Line
        # point_on_line_check = (epp_line * kornia.geometry.conversions.convert_points_to_homogeneous(epp_point)).sum(dim=-1).abs()
        vec1x, vec1y, _ = torch.split(epp_line, 1, dim=-1)
        vec1 = torch.cat([-vec1y, vec1x], dim=-1)
        vec1 = vec1 / torch.sqrt(torch.sum(vec1 ** 2, dim=-1, keepdim=True) + self.eps)
        vec2 = (kornia.geometry.conversions.convert_points_from_homogeneous(pts_target) - epp_point)
        vec2prj = torch.sum(vec1 * vec2, dim=-1, keepdim=True) / torch.sum(vec1 ** 2, dim=-1, keepdim=True) * vec1
        pts_target_eppline = vec2prj + epp_point
        # point_on_line_check = (epp_line * kornia.geometry.conversions.convert_points_to_homogeneous(pts_target_eppline)).sum(dim=-1).abs()

        target2prj_dist = ((pts_target_eppline - kornia.geometry.conversions.convert_points_from_homogeneous(pts_target)) ** 2 + self.eps).sum(dim=-1, keepdim=True).sqrt()
        valid = (target2prj_dist < self.scale_th) * inliers.unsqueeze(2)

        pts_target_eppline_maxscale = pts_target_eppline + vec1 * (self.scale_th - target2prj_dist)
        # point_on_line_check = (epp_line * kornia.geometry.conversions.convert_points_to_homogeneous(pts_target_eppline_maxscale)).sum(dim=-1).abs()
        pts_target_eppline_minscale = pts_target_eppline - vec1 * (self.scale_th - target2prj_dist)
        # point_on_line_check = (epp_line * kornia.geometry.conversions.convert_points_to_homogeneous(pts_target_eppline_maxscale)).sum(dim=-1).abs()

        # Camera Scale range where the projection and correspondence has L2 distance smaller than scale_th
        maxscale = self.projection2scale(pts_source, pts_target_eppline_maxscale, depthf, intrinsic, R, t)
        minscale = self.projection2scale(pts_source, pts_target_eppline_minscale, depthf, intrinsic, R, t)

        minbinidx, maxbinidx = 0, int(self.maxscale_vote / self.precision - 2)
        minfillidx = torch.clamp(torch.round(minscale / self.precision), min=minbinidx, max=maxbinidx)
        maxfillidx = torch.clamp(torch.round(maxscale / self.precision), min=minbinidx, max=maxbinidx)

        voting_vector = self.generate_voting_vector(device).int()
        minfillidx, maxfillidx, valid = minfillidx.int().squeeze(-1), maxfillidx.int().squeeze(-1), valid.int().squeeze(-1)
        GPUEPM.vote_for_optimal_scale(
            voting_vector, minfillidx, maxfillidx, valid, voting_vector.shape[1], self.topk, self.npts
        )
        scale = torch.argmax(voting_vector, axis=1).float() * self.precision

        # Projection Constraint Computation
        pose = torch.cat([R, t * scale.view([self.topk, 1, 1])], dim=-1)

        return pose, scale

    def projection2scale(self, pts_source, pts_target, depthf, intrinsic, R, t):
        xxf_target, yyf_target = torch.split(pts_target, 1, dim=-1)

        intrinsic_t = intrinsic @ t
        M = intrinsic @ R @ intrinsic.inverse()

        x, y, z = torch.split(intrinsic_t, 1, dim=1)
        m0x, m1x, m2x = torch.split(pts_source @ M.transpose(-1, -2), 1, dim=2)

        scale = depthf * (m2x * xxf_target - m0x) / (x - xxf_target * z + self.eps)
        # scale_prx = depthf * (m2x * xxf_target - m0x) / (x - xxf_target * z + self.eps)
        # scale_pry = depthf * (m2x * yyf_target - m1x) / (y - yyf_target * z + self.eps)
        return scale