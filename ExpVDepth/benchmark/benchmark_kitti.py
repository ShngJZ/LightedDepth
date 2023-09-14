import os, sys, argparse
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, project_root)

import einops
import numpy as np
import PIL.Image as Image

from tqdm import tqdm
from tabulate import tabulate

import torch
import torch.utils.data as data
from utils.utils import InputPadder, to_cuda, readFlowKITTI, writeFlowKITTI
from utils.evaluation import compute_errors
from utils.pose_estimator import PoseEstimator

from ExpVDepth.RAFT.raft import RAFT
from ExpVDepth.LDNet.LightedDepthNet import LightedDepthNet
from ExpVDepth.datasets.kitti_eigen_test import KITTI_eigen

def read_splits_kitti_test():
    split_root = os.path.join(project_root, 'misc', 'kitti_splits')
    entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'test_files.txt'), 'r')]
    return entries
def upgrade_measures(vdepth, gtdepth, selector, measures, SfM=False):
    vdepthf = vdepth[selector == 1]
    gtdepthf = gtdepth[selector == 1]

    if SfM:
        vdepthf = vdepthf * np.median(gtdepthf) / np.median(vdepthf)

    measures[:10] += compute_errors(gt=gtdepthf, pred=vdepthf)
    measures[10] += 1
    return measures

@torch.no_grad()
def validate_kitti(raft, lighteddepth, args, iters=24):
    """
    Validation on Kitti Eigen Split
    scale_th: even scale_th is 0.0, it may still run over frames, as algorithm automatically seek next frame when pose esitmation fail
    """
    assert args.scale_th >= 0.0

    raft.eval()
    lighteddepth.eval()

    pose_estimator = PoseEstimator(npts=10000, device=torch.device("cuda"))

    val_dataset = KITTI_eigen(
        data_root=args.kitti_root,
        entries=read_splits_kitti_test(),
        net_ht=args.net_ht, net_wd=args.net_wd,
        mono_depth_root=os.path.join(args.estimated_inputs_root, 'monodepth_kitti', args.mono_depth_init),
        grdt_depth_root=args.grdt_depth_root
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False
    )

    vmeasures, vmeasures_sfm, mmeasures, mmeasures_sfm = np.zeros(11), np.zeros(11), np.zeros(11), np.zeros(11)

    for val_id, data_blob in enumerate(tqdm(val_loader)):
        data_blob = to_cuda(data_blob)

        seq, frmidx, _ = data_blob['tag'][0].split(' ')
        h, w = data_blob['uncropped_size'][0]

        frmidx, h, w = int(frmidx), h.item(), w.item()

        # Lighted Depth Takes Image of size 320 x 1216
        crph, crpw = 320, 1216
        left, top = int((w - crpw) / 2), int(h - crph)

        eigen_crop_mask = np.ones([int(h), int(w)])
        eigen_crop_mask[int(0.3324324 * h):int(0.91351351 * h), int(0.0359477 * w):int(0.96405229 * w)] = 1
        eigen_crop_mask = eigen_crop_mask[top:top + crph, left:left + crpw]
        eigen_crop_mask = eigen_crop_mask == 1

        # Pose Estimation Stage
        coords1_init, interval = None, 1

        while(True):
            img1_idx = frmidx
            img2_idx = frmidx + interval

            img1path = os.path.join(
                args.kitti_root, seq, 'image_02/data', '{}.png'.format(str(img1_idx).zfill(10))
            )
            img2path = os.path.join(
                args.kitti_root, seq, 'image_02/data', '{}.png'.format(str(img2_idx).zfill(10))
            )

            if not os.path.exists(img2path):
                img2path = os.path.join(
                    args.kitti_root, seq, 'image_02/data', '{}.png'.format(str(frmidx + interval - 1).zfill(10))
                )
                break

            image1 = torch.from_numpy(np.array(Image.open(img1path))).permute([2, 0, 1]).float()[None].cuda()
            image2 = torch.from_numpy(np.array(Image.open(img2path))).permute([2, 0, 1]).float()[None].cuda()

            padder = InputPadder(image1.shape, mode='kitti')
            image1, image2 = padder.pad(image1, image2)

            flow_root = os.path.join(args.estimated_inputs_root, 'opticalflow_kitti', args.flow_init, seq)
            flow_path = os.path.join(flow_root, '{}_{}.png'.format(str(img1_idx).zfill(10), str(img2_idx).zfill(10)))
            if not os.path.exists(flow_path):
                os.makedirs(flow_root, exist_ok=True)
                flow, coords1_init = raft(image1, image2, iters=iters, coords1_init=coords1_init)
                flow = padder.unpad(flow)
                flow = einops.rearrange(flow.squeeze(), 'd h w -> h w d')

                flownp = flow.cpu().numpy()
                writeFlowKITTI(flow_path, flownp)
            else:
                flownp, _ = readFlowKITTI(flow_path)
                flow = torch.from_numpy(flownp).float().cuda()

            mdepth_uncropped, intrinsic_uncropped = data_blob['mono_depth_uncropped'].squeeze(), data_blob['intrinsic_uncropped'].squeeze()
            pose, scale_md = pose_estimator.pose_estimation(flow, mdepth_uncropped, intrinsic_uncropped[0:3, 0:3], seed=val_id)

            if scale_md > args.scale_th:
                break
            else:
                interval += 1

        image1 = torch.from_numpy(np.array(Image.open(img1path))).permute([2, 0, 1]).float().unsqueeze(0).cuda()
        image2 = torch.from_numpy(np.array(Image.open(img2path))).permute([2, 0, 1]).float().unsqueeze(0).cuda()

        image1 = image1[:, :, top:top + crph, left:left + crpw].contiguous() / 255.0
        image2 = image2[:, :, top:top + crph, left:left + crpw].contiguous() / 255.0

        mdepth_cropped = data_blob['mono_depth_cropped']
        mdepth_cropped = torch.clamp_min(mdepth_cropped, min=args.min_depth_pred)

        intrinsic_cropped = data_blob['intrinsic_cropped']

        outputs = lighteddepth(image1, image2, mdepth_cropped, intrinsic_cropped, pose)

        gtdepth = data_blob['grdt_depth_cropped'].squeeze().cpu().numpy()
        vdepth = outputs[('depth', 2)].squeeze().cpu().numpy()
        mdepth = data_blob['mono_depth_cropped'].squeeze().cpu().numpy()
        selector = (gtdepth > 1e-3) * (gtdepth < 80) * eigen_crop_mask

        mmeasures = upgrade_measures(mdepth, gtdepth, selector, mmeasures, SfM=False)
        mmeasures_sfm = upgrade_measures(mdepth, gtdepth, selector, mmeasures_sfm, SfM=True)
        vmeasures = upgrade_measures(vdepth, gtdepth, selector, vmeasures, SfM=False)
        vmeasures_sfm = upgrade_measures(vdepth, gtdepth, selector, vmeasures_sfm, SfM=True)

    mmeasures[0:10] = mmeasures[0:10] / mmeasures[10]
    mmeasures_sfm[0:10] = mmeasures_sfm[0:10] / mmeasures_sfm[10]
    vmeasures[0:10] = vmeasures[0:10] / vmeasures[10]
    vmeasures_sfm[0:10] = vmeasures_sfm[0:10] / vmeasures_sfm[10]

    print('KITTI Performance Reported over %f eval samples:' % (vmeasures[10].item()))
    table = [
        ['', 'ScInv', 'log10', 'silog', 'abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1', 'd2', 'd3'],
        ['MDepth'] + list(mmeasures[0:10]),
        ['MDepth-SfM'] + list(mmeasures_sfm[0:10]),
        ['VDepth'] + list(vmeasures[0:10]),
        ['VDepth-SfM'] + list(vmeasures_sfm[0:10]),
    ]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', numalign="center", floatfmt=".3f"))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raft_ckpt',                  help="checkpoint of RAFT",              type=str,   default=os.path.join(project_root, 'misc/checkpoints/raft-sintel.pth'))
    parser.add_argument('--ldepth_ckpt',                help="checkpoint of LightedDepth",      type=str,   default=os.path.join(project_root, 'misc/checkpoints/lighteddepth_kitti.pth'))

    parser.add_argument('--kitti_root',                 help="root of kitti",                   type=str,   default=os.path.join(project_root, 'data', 'kitti'))
    parser.add_argument('--grdt_depth_root',            help="root of grdt_depth",              type=str,   default=os.path.join(project_root, 'data', 'smidensegt_kitti'))
    parser.add_argument('--estimated_inputs_root',      help="root of estimated_inputs",        type=str,   default=os.path.join(project_root, 'estimated_inputs'))
    parser.add_argument('--mono_depth_init',            help="method name of mono_depth",       type=str,   default='bts')
    parser.add_argument('--flow_init',                  help="method name of optical flow",     type=str,   default='raft')

    parser.add_argument('--net_ht',                                                             type=int,   default=320)
    parser.add_argument('--net_wd',                                                             type=int,   default=1216)
    parser.add_argument('--min_depth_pred',                                                     type=float, default=1)
    parser.add_argument('--maxlogscale',                                                        type=float, default=1.5)
    parser.add_argument('--scale_th',                                                           type=float, default=0.0)

    args = parser.parse_args()

    assert args.mono_depth_init in ['bts', 'adabins', 'newcrfs', 'monodepth2']

    raft = RAFT(args)
    raft.load_state_dict(torch.load(args.raft_ckpt, map_location="cpu"), strict=True)
    raft.cuda()
    raft.eval()

    lighteddepth = LightedDepthNet(args=args)
    lighteddepth.load_state_dict(torch.load(args.ldepth_ckpt, map_location="cpu"), strict=True)
    lighteddepth.cuda()
    lighteddepth.eval()

    validate_kitti(raft, lighteddepth, args)