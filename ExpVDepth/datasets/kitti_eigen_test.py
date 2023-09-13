import os, sys, copy
project_rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, project_rootdir)

import numpy as np
import PIL.Image as Image

import torch, torchvision
import torch.utils.data as data
from utils.utils import read_calib_file, get_intrinsic_extrinsic

def crop_img(img, left, top, crph, crpw):
    if img.ndim == 2:
        img_cropped = img[top:top + crph, left:left + crpw]
    elif img.ndim == 3:
        img_cropped = img[top:top + crph, left:left + crpw, :]
    return img_cropped

class KITTI_eigen(data.Dataset):
    def __init__(self, data_root, entries, net_ht, net_wd, mono_depth_root, grdt_depth_root):
        super(KITTI_eigen, self).__init__()
        self.data_root = data_root
        self.net_ht, self.net_wd = net_ht, net_wd

        self.entries, self.images, self.intrinsics, self.mono_depths, self.grdt_depths = list(), list(), list(), list(), list()

        for entry in entries:
            seq, index, _ = entry.split(' ')
            index = int(index)

            img1path = os.path.join(data_root, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))
            img2path = os.path.join(data_root, seq, 'image_02', 'data', "{}.png".format(str(index + 1).zfill(10)))
            if not os.path.exists(img2path):
                img2path = img1path

            # Load Intrinsic for each frame
            calib_dir = os.path.join(data_root, seq.split('/')[0])

            cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
            velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
            imu2cam = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
            intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)

            mono_depth_path = os.path.join(mono_depth_root, seq, "image_02/{}.png".format(str(index).zfill(10)))
            grdt_depth_path = os.path.join(grdt_depth_root, seq, "image_02/{}.png".format(str(index).zfill(10)))

            if not os.path.exists(grdt_depth_path):
                # Remove Files without Test Files if without GT Semidense Depthmap
                continue

            self.mono_depths.append(mono_depth_path)
            self.grdt_depths.append(grdt_depth_path)

            self.images.append([img1path, img2path])
            self.intrinsics.append(intrinsic)
            self.entries.append(entry)

        assert len(self.images) == len(self.entries) == len(self.mono_depths) == len(self.grdt_depths) == len(self.intrinsics)

    def read_rgb(self, file_path):
        return np.array(Image.open(file_path)).astype(np.uint8)

    def read_depth(self, file_path):
        return np.array(Image.open(file_path)).astype(np.float32) / 256.0

    def __getitem__(self, index):
        img1_path, img2_path = self.images[index]
        img1, img2 = self.read_rgb(img1_path), self.read_rgb(img2_path)

        h, w, _ = img1.shape

        intrinsic_uncropped = self.intrinsics[index]
        mono_depth_uncropped = self.read_depth(self.mono_depths[index])
        grdt_depth_uncropped = self.read_depth(self.grdt_depths[index])

        mono_depth_cropped, grdt_depth_cropped, intrinsic_cropped = self.crop(mono_depth_uncropped, grdt_depth_uncropped, copy.deepcopy(intrinsic_uncropped))

        data_blob = {
            'img1': torch.from_numpy(img1).permute([2, 0, 1]).float(),
            'img2': torch.from_numpy(img2).permute([2, 0, 1]).float(),
            'grdt_depth_cropped': torch.from_numpy(grdt_depth_cropped).unsqueeze(0).float(),
            'mono_depth_cropped': torch.from_numpy(mono_depth_cropped).unsqueeze(0).float(),
            'intrinsic_cropped': torch.from_numpy(intrinsic_cropped).float(),
            'mono_depth_uncropped': torch.from_numpy(mono_depth_uncropped).unsqueeze(0).float(),
            'intrinsic_uncropped': torch.from_numpy(intrinsic_uncropped).unsqueeze(0).float(),
            'tag': self.entries[index],
            'uncropped_size': np.array([h, w])
        }

        return data_blob

    def crop(self, mono_depth, grdt_depth, intrinsic):
        h, w = mono_depth.shape

        crop = np.array([0.40810811 * h, 0.99189189 * h, 0.03594771 * w, 0.96405229 * w]).astype(np.int32)
        crop_mask = np.zeros_like(mono_depth)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1

        grdt_depth = grdt_depth * crop_mask

        assert self.net_ht <= h and self.net_wd <= w

        left, top = int((w - self.net_wd) / 2), int(h - self.net_ht)

        intrinsic[0:2, 2] = intrinsic[0:2, 2] - np.array([left, top])
        mono_depth = crop_img(mono_depth, left=left, top=top, crph=self.net_ht, crpw=self.net_wd)
        grdt_depth = crop_img(grdt_depth, left=left, top=top, crph=self.net_ht, crpw=self.net_wd)

        return mono_depth, grdt_depth, intrinsic

    def wrapup(self,
               img1, img2,
               grdt_depth_cropped, mono_depth_cropped, intrinsic_cropped,
               mono_depth_uncropped, intrinsic_uncropped, tag, uncropped_size):

        data_blob = {
            'img1': torch.from_numpy(img1).permute([2, 0, 1]).float(),
            'img2': torch.from_numpy(img2).permute([2, 0, 1]).float(),
            'grdt_depth_cropped': torch.from_numpy(grdt_depth_cropped).unsqueeze(0).float(),
            'mono_depth_cropped': torch.from_numpy(mono_depth_cropped).unsqueeze(0).float(),
            'intrinsic_cropped': torch.from_numpy(intrinsic_cropped).float(),
            'mono_depth_uncropped': torch.from_numpy(mono_depth_uncropped).unsqueeze(0).float(),
            'intrinsic_uncropped': torch.from_numpy(intrinsic_uncropped).unsqueeze(0).float(),
            'tag': tag,
            'uncropped_size': uncropped_size
        }

        return data_blob

    def __len__(self):
        return len(self.entries)