CP-Net: Contour-Perturbed Reconstruction Network for Self-Supervised Point Cloud Learning
===

## Usage

### Requirement

- Python 3
- Pytorch 0.4
- CMake > 2.8

**Note**: The code is not not compatible with Pytorch >= 1.0 due to the C++/CUDA extensions. 

### Building C++/CUDA Extensions for PointNet++

```
mkdir build && cd build
cmake .. && make
```


### Training & Evaluation

Self-supervised pretraining:
```
bash self-supervised_pretrain.sh RSCNN ShapeNetPart
```
After pretraining, you can get the pretrained features with 'train_features_saved.h5' and 'test_features_saved.h5', which is the input of nect stage: semi-supervised part segmentation:

```
python semi-supervised-finetuning.py --sample_rate 0.05 --exp exp_name

```

## Acknowledgement

[Relation-Shape CNN](https://github.com/Yochengliu/Relation-Shape-CNN) 

[Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).

[PointGLR](https://github.com/raoyongming/PointGLR)

