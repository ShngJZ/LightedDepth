import copy
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
from utils.evaluation import compute_errors, upgrade_measures
from utils.pose_estimator import PoseEstimator

from ExpVDepth.RAFT.raft import RAFT
from ExpVDepth.LDNet.LightedDepthNet import LightedDepthNet
from ExpVDepth.datasets.nyuv2_test import NYUv2

def read_splits():
    split_root = os.path.join(project_root, 'misc/nyuv2_splits')
    entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'nyuv2_test.txt'), 'r')]
    return entries

@torch.no_grad()
def validate_nyuv2(raft, lighteddepth, args):

    raft.eval()
    lighteddepth.eval()

    pose_estimator = PoseEstimator(
        npts=50000, device=torch.device("cuda"),
        prj_w=0.05, maxscale_vote=1.0, precision=0.001,
        ransac_threshold=0.0001, scale_th=2
    )

    entries = read_splits()

    val_dataset = NYUv2(
        data_root=args.nyuv2_root,
        net_ht=args.net_ht, net_wd=args.net_wd,
        entries=entries,
        est_inputs_root=os.path.join(args.est_inputs_root, 'monodepth_nyuv2', args.mono_depth_method),
        frmgap=args.frmgap
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False
    )

    vmeasures, vmeasures_sfm, mmeasures, mmeasures_sfm = np.zeros(11), np.zeros(11), np.zeros(11), np.zeros(11)

    crph, crpw = 448, 640
    top, left = int((480 - crph) / 2), int((640 - crpw) / 2)

    eigen_crop_mask = np.zeros([480, 640])
    eigen_crop_mask[45:471, 41:601] = 1
    eigen_crop_mask = eigen_crop_mask[top:top + crph, left:left + crpw]

    for val_id, data_blob in enumerate(tqdm(val_loader)):
        data_blob = to_cuda(data_blob)

        seq, frmidx = data_blob['tag'][0].split(' ')
        frmidx = int(frmidx)
        flow_root = os.path.join(args.est_inputs_root, 'opticalflow_nyuv2', args.flow_init, seq)
        flow_path = os.path.join(flow_root, '{}_{}.png'.format(str(frmidx).zfill(5), str(frmidx+args.frmgap).zfill(5)))
        if not os.path.exists(flow_path):
            os.makedirs(flow_root, exist_ok=True)
            flow, _ = raft(data_blob['img1_cropped'], data_blob['img2_cropped'])
            flow = einops.rearrange(flow.squeeze(), 'd h w -> h w d')
            writeFlowKITTI(flow_path, flow.cpu().numpy())
        else:
            flownp, _ = readFlowKITTI(flow_path)
            flow = torch.from_numpy(flownp).float().cuda()

        mono_depth_cropped, intrinsic_cropped = data_blob['mono_depth_cropped'].squeeze(), data_blob['intrinsic_cropped'].squeeze()
        valid_regeion = torch.zeros([crph, crpw], device=mono_depth_cropped.device, dtype=torch.bool)
        valid_regeion[:, 41:601] = 1
        pose, scale_md = pose_estimator.pose_estimation(
            flow, mono_depth_cropped, intrinsic_cropped[0:3, 0:3], valid_regeion=valid_regeion, seed=val_id
        )

        outputs = lighteddepth(
            data_blob['img1_cropped'] / 255.0,
            data_blob['img2_cropped'] / 255.0,
            data_blob['mono_depth_cropped'],
            data_blob['intrinsic_cropped'],
            pose
        )

        gtd, prd_vido, prd_mono = data_blob['grdt_depth_cropped'], outputs[('depth', 2)], data_blob['mono_depth_cropped']
        gtd, prd_vido, prd_mono = gtd.squeeze().cpu().numpy(), prd_vido.squeeze().cpu().numpy(), prd_mono.squeeze().cpu().numpy()
        selector = (gtd > args.min_depth_pred) * (gtd < args.max_depth_pred) * eigen_crop_mask

        mmeasures = upgrade_measures(prd_mono, gtd, selector, mmeasures, SfM=False)
        mmeasures_sfm = upgrade_measures(prd_mono, gtd, selector, mmeasures_sfm, SfM=True)
        vmeasures = upgrade_measures(prd_vido, gtd, selector, vmeasures, SfM=False)
        vmeasures_sfm = upgrade_measures(prd_vido, gtd, selector, vmeasures_sfm, SfM=True)

    mmeasures[0:10] = mmeasures[0:10] / mmeasures[10]
    mmeasures_sfm[0:10] = mmeasures_sfm[0:10] / mmeasures_sfm[10]
    vmeasures[0:10] = vmeasures[0:10] / vmeasures[10]
    vmeasures_sfm[0:10] = vmeasures_sfm[0:10] / vmeasures_sfm[10]

    print('NYUv2 Performance Reported over %f eval samples:' % (vmeasures[10].item()))
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
    parser.add_argument('--raft_ckpt',                  help="checkpoint of RAFT",                      type=str,   default=os.path.join(project_root, 'misc/checkpoints/raft-sintel.pth'))
    parser.add_argument('--ldepth_ckpt',                help="checkpoint of LightedDepth",              type=str,   default=os.path.join(project_root, 'misc/checkpoints/lighteddepth_nyuv2.pth'))

    parser.add_argument('--nyuv2_root',                 help="root of nyuv2",                           type=str,   default=os.path.join(project_root, 'data', 'nyuv2_organized'))
    parser.add_argument('--est_inputs_root',            help="root of estimated_inputs",                type=str,   default=os.path.join(project_root, 'estimated_inputs'))
    parser.add_argument('--mono_depth_method',          help="method name of mono_depth",               type=str,   default='bts')
    parser.add_argument('--flow_init',                  help="method name of optical flow",             type=str,   default='raft')

    parser.add_argument('--net_ht',                                                                     type=int,   default=448)
    parser.add_argument('--net_wd',                                                                     type=int,   default=640)
    parser.add_argument('--min_depth_pred',             help="minimal evaluate depth",                  type=float, default=0.8)
    parser.add_argument('--max_depth_pred',             help="maximal evaluate depth",                  type=float, default=10.0)
    parser.add_argument('--maxlogscale',                help="maximum stereo residual value",           type=float, default=0.8)
    parser.add_argument('--scale_th',                   help="minimal camera translation magnitude",    type=float, default=0.0)
    parser.add_argument('--dataset_type',               help="which experiment dataset",                type=str,   default="nyuv2")
    parser.add_argument('--frmgap',                     help="index gap between sampled two frames",    type=int,   default=3)

    args = parser.parse_args()

    assert args.mono_depth_method in ['bts', 'adabins', 'newcrfs', 'monodepth2']

    raft = RAFT(args)
    raft.load_state_dict(torch.load(args.raft_ckpt, map_location="cpu"), strict=True)
    raft.cuda()
    raft.eval()

    lighteddepth = LightedDepthNet(args=args)
    lighteddepth.load_state_dict(torch.load(args.ldepth_ckpt, map_location="cpu"), strict=True)
    lighteddepth.cuda()
    lighteddepth.eval()

    validate_nyuv2(raft, lighteddepth, args)