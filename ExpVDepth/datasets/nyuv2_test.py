from __future__ import print_function, division
import os, sys, inspect, cv2, time, copy, glob, natsort
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

def query_nextimgs(data_root, seq):
    rgbs = glob.glob(os.path.join(data_root, seq, 'rgb_*'))
    return natsort.natsorted(rgbs)

class NYUv2(data.Dataset):
    def __init__(self, data_root, entries, net_ht, net_wd, est_inputs_root, frmgap=1):
        super(NYUv2, self).__init__()
        self.net_ht, self.net_wd = net_ht, net_wd

        self.images, self.mono_depths, self.grdt_depths = list(), list(), list()
        self.entries = entries

        fx_rgb, fy_rgb = 5.1885790117450188e+02, 5.1946961112127485e+02
        cx_rgb, cy_rgb = 3.2558244941119034e+02, 2.5373616633400465e+02
        self.intrinsic = np.array([
            [fx_rgb, 0, cx_rgb, 0],
            [0, fy_rgb, cy_rgb, 0],
            [0, 0,      1,      0],
            [0, 0,      0,      1]
        ])

        for entry in self.entries:
            seq, index = entry.split(' ')
            index = int(index)

            img1path = os.path.join(data_root, seq, 'rgb_{}.png'.format(str(index).zfill(5)))
            img2paths = query_nextimgs(data_root, seq)
            if len(img2paths) >= frmgap:
                img2path = img2paths[frmgap]
            else:
                img2path = img2paths[-1]

            mono_depth_path = os.path.join(est_inputs_root, seq, "sync_depth_{}.png".format(str(index).zfill(5)))
            grdt_depth_path = os.path.join(data_root, seq, 'sync_depth_{}.png'.format(str(index).zfill(5)))

            self.images.append([img1path, img2path])
            self.mono_depths.append(mono_depth_path)
            self.grdt_depths.append(grdt_depth_path)

    def __getitem__(self, index):
        img1path, img2path = self.images[index]
        img1 = np.array(Image.open(img1path))
        img2 = np.array(Image.open(img2path))

        intrinsic_uncropped = copy.deepcopy(self.intrinsic)
        mono_depth_uncropped = cv2.imread(self.mono_depths[index], -1) / 1000.0
        grdt_depth_uncropped = cv2.imread(self.grdt_depths[index], -1) / 1000.0

        img1_cropped, img2_cropped, mono_depth_cropped, grdt_depth_cropped, intrinsic_cropped = self.aug_crop(
            img1, img2, mono_depth_uncropped, grdt_depth_uncropped, intrinsic_uncropped
        )

        data_blob = {
            'img1_uncropped': torch.from_numpy(img1).permute([2, 0, 1]).float(),
            'img2_uncropped': torch.from_numpy(img2).permute([2, 0, 1]).float(),
            'img1_cropped': torch.from_numpy(img1_cropped).permute([2, 0, 1]).float(),
            'img2_cropped': torch.from_numpy(img2_cropped).permute([2, 0, 1]).float(),
            'mono_depth_cropped': torch.from_numpy(mono_depth_cropped).unsqueeze(0).float(),
            'grdt_depth_cropped': torch.from_numpy(grdt_depth_cropped).unsqueeze(0).float(),
            'intrinsic_cropped': torch.from_numpy(intrinsic_cropped).float(),
            'mono_depth_uncropped': torch.from_numpy(mono_depth_uncropped).unsqueeze(0).float(),
            'intrinsic_uncropped': torch.from_numpy(intrinsic_uncropped).unsqueeze(0).float(),
            'tag': self.entries[index],
            'uncropped_size': np.array([img1.shape[0], img1.shape[1]])
        }

        return data_blob

    def aug_crop(self, img1, img2, mono_depth, grdt_depth, intrinsic):
        h, w = img1.shape[0:2]

        eval_mask = np.zeros(grdt_depth.shape)
        eval_mask[20:459, 24:615] = 1
        grdt_depth = grdt_depth * eval_mask

        crph, crpw = self.net_ht, self.net_wd

        if crph >= h:
            crph = h

        if crpw >= w:
            crpw = w

        top = int((h - crph) / 2)
        left = int((w - crpw) / 2)

        intrinsic[0, 2] -= left
        intrinsic[1, 2] -= top

        img1 = self.crop_img(img1, left=left, top=top, crph=crph, crpw=crpw)
        img2 = self.crop_img(img2, left=left, top=top, crph=crph, crpw=crpw)
        mono_depth = self.crop_img(mono_depth, left=left, top=top, crph=crph, crpw=crpw)
        grdt_depth = self.crop_img(grdt_depth, left=left, top=top, crph=crph, crpw=crpw)


        return img1, img2, mono_depth, grdt_depth, intrinsic

    def crop_img(self, img, left, top, crph, crpw):
        img_cropped = img[top:top+crph, left:left+crpw]
        return img_cropped

    def __len__(self):
        return len(self.entries)