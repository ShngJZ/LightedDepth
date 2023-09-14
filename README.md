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
  

# Quick Start
### Data Preparation
**Step 0.** Download [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php) and its [semidense gt](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php).

**Step 1.** Symlink the dataset root to `./data/`.
```
ln -s [your data root] ./data/
ln -s [your input root] ./estimated_inputs/
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
├── estimated_inputs
│   ├──monodepth_kitti (download in Hugging Face)
│   │   ├──adabins
│   │   ├──bts
│   │   ├──monodepth2
│   │   ├──newcrfs
│   ├──opticalflow_kitti (generated on-the-fly)
├── misc
│   ├── chckpoints (download in Hugging Face)
```
For monodepth_kitti, we provide different Monodepth initializations in [Hugging Face](https://huggingface.co/datasets/Shengjie/LightedDepth-Dataset/tree/main).

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
**Attention:** Make sure your system CUDA version consistent with your Pytorch CUDA version.

### Benchmark
```
python benchmark/benchmark_kitti.py --mono_depth_init newcrfs
```
You can choose from different initialization methods. The model is trained using [BTS](https://github.com/cleinc/bts) as mono-initialization.

| Exp                                                                                                                        |                  Mono-Init                   |  abs_rel  |  sq_rel   |    rms    |  log_rms  |    d1     |    d2     |    d3     |
|----------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| [DeepV2D](https://github.com/princeton-vl/DeepV2D)                                                                         |                      -                       |   0.037   |   0.167   |   1.984   |   0.073   |   0.978   |   0.994   |     -     |
| [LightedDepth](https://huggingface.co/datasets/Shengjie/LightedDepth-Dataset/blob/main/checkpoints/lighteddepth_kitti.pth) |    [BTS](https://github.com/cleinc/bts)      |   0.029   | 0.093     |  1.701    |   0.053   |   0.989   |   0.998   | **1.000** |
| [LightedDepth](https://huggingface.co/datasets/Shengjie/LightedDepth-Dataset/blob/main/checkpoints/lighteddepth_kitti.pth) | [NewCRFs](https://github.com/aliyun/NeWCRFs) | **0.028** | **0.077** | **1.567** | **0.049** | **0.991** | **0.999** | **1.000** |