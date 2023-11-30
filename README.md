# LightedDepth
Code and data for **[LightedDepth: Video Depth Estimation in light of Limited Inference View Angles, Zhu et al, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_LightedDepth_Video_Depth_Estimation_in_Light_of_Limited_Inference_View_CVPR_2023_paper.pdf)** 

# Qualitative Results
<details>
<summary> KITTI </summary>

https://github.com/ShngJZ/LightedDepth/assets/128062217/5801f0fd-c3ed-4c9e-8120-3067ff518735

</details>

<details>
<summary> NYUv2 </summary>


https://github.com/ShngJZ/LightedDepth/assets/128062217/46d926a4-ac32-4865-b86b-f4de6822c1a4

</details>
  

# Two-View Pose Estimator
LightedDepth provides SoTA two-view scaled pose estimator.
```
poseestimator = PoseEstimator()
poseestimator.pose_estimation(flow, depth, intrinsic)
```


# Quick Start
### Data Preparation
**Step 0.** Download [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php) and its [semidense gt](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php).

**Step 1.** Symlink the dataset root to `./data`.
```
ln -s [your data root] ./data
ln -s [your input root] ./estimated_inputs
```
The directory will be as follows.
```
LightedDepth
├── data
│   ├── kitti
│   │   ├── 2011_09_26
│   │   ├── 2011_09_28
│   │   ├── 2011_09_29
│   │   ├── ...
│   ├── semidensegt_kitti
│   ├── nyuv2_organized (Hugging Face)
├── estimated_inputs
│   ├──monodepth_kitti (Hugging Face)
│   │   ├──adabins
│   │   ├──bts
│   │   ├──monodepth2
│   │   ├──newcrfs
│   ├──opticalflow_kitti (generated on-the-fly)
│   ├──monodepth_nyuv2 (Hugging Face)
│   │   ├──adabins
│   │   ├──bts
│   │   ├──newcrfs
├── misc
│   ├── chckpoints (Hugging Face)
```
Please download different Monodepth initializations / checkpoints in [Hugging Face](https://huggingface.co/datasets/Shengjie/LightedDepth-Dataset/tree/main).

### Installation
**Step 0.** Install [pytorch](https://pytorch.org/).

**Step 1.** Install requirements.
```
pip install -r scripts/requirements.txt
```

**Step 2.** Compile our two-view pose estimation algorithm.
```
cd GPUEPMatrixEstimation
python setup.py install
```
**Attention:** Make sure your system CUDA version is consistent with your Pytorch CUDA version.

### Benchmark
For KITTI Dataset:
```
python benchmark/benchmark_kitti.py --mono_depth_method newcrfs
```
You can choose from different initialization methods. The model is trained using [BTS](https://github.com/cleinc/bts) as mono-initialization.

| Exp                                                                                                                        |                  Mono-Init                   |  abs_rel  |  sq_rel   |    rms    |  log_rms  |    d1     |    d2     |    d3     |
|----------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| [DeepV2D](https://github.com/princeton-vl/DeepV2D)                                                                         |                      -                       |   0.037   |   0.167   |   1.984   |   0.073   |   0.978   |   0.994   |     -     |
| [LightedDepth](https://huggingface.co/datasets/Shengjie/LightedDepth-Dataset/blob/main/checkpoints/lighteddepth_kitti.pth) |    [BTS](https://github.com/cleinc/bts)      |   0.029   | 0.093     |  1.701    |   0.053   |   0.989   |   0.998   | **1.000** |
| [LightedDepth](https://huggingface.co/datasets/Shengjie/LightedDepth-Dataset/blob/main/checkpoints/lighteddepth_kitti.pth) | [NewCRFs](https://github.com/aliyun/NeWCRFs) | **0.028** | **0.077** | **1.567** | **0.049** | **0.991** | **0.999** | **1.000** |

For NYUv2 Dataset:
```
python benchmark/benchmark_nyuv2.py --mono_depth_method newcrfs
```
Currently, there is a performance drop in NYUv2 evaluation due slight change in the implementation of pose estimation algorithm.
We are working on resolving the issue.
Basically, we will release training codes. The drop should go away after retraining.

# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@inproceedings{zhu2023lighteddepth,
  title={LightedDepth: Video Depth Estimation in light of Limited Inference View Angles},
  author={Zhu, Shengjie and Liu, Xiaoming},
  booktitle={CVPR},
  year={2023}
}
```

# Acknowledgement
Part of our codes is from these excellent open source projects:
- [Deep-SfM-Revisited](https://github.com/jytime/Deep-SfM-Revisited) 
- [RAFT](https://github.com/princeton-vl/RAFT)