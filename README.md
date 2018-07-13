# CSRNet-pytorch

This is the PyTorch version repo for [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/abs/1802.10062) in CVPR 2018, which delivered a state-of-the-art, straightforward and end-to-end architecture for crowd counting tasks.

## Datasets
ShanghaiTech Dataset: [Google Drive](https://drive.google.com/open?id=16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI)

## Prerequisites
We strongly recommend Anaconda as the environment.

Python: 2.7

PyTorch: 0.4.0

CUDA: 9.2
## Ground Truth

Please follow the `make_dataset.ipynb ` to generate the ground truth.

## Training process




## References

If you find the CSRNet useful, please cite our paper. Thank you!

```
@inproceedings{li2018csrnet,
  title={CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes},
  author={Li, Yuhong and Zhang, Xiaofan and Chen, Deming}
}
```
Please cite the Shanghai datasets and other works if you use them.

```
@inproceedings{zhang2016single,
  title={Single-image crowd counting via multi-column convolutional neural network},
  author={Zhang, Yingying and Zhou, Desen and Chen, Siqin and Gao, Shenghua and Ma, Yi},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={589--597},
  year={2016}
}
```
