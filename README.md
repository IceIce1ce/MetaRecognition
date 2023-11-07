This is the official repository of 

**MetaRecognition: A Unified Framework for Video Action Recognition.**

## Setup
```bash
conda create -n meta_recognition python=3.10
conda activate meta_recognition
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Dataset Preparation
Download the [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) dataset and structure the data as follows:
```
dataset/UCF-101/
  ApplyEyeMakeup
    .avi
  ApplyLipstick
    .avi
  Archery
    .avi
  ...
```

## Usage
To use our model, follow the code snippet below:
```bash
cd Video_Recognition

# Train, Test and Demo C3D
bash train_c3d.sh
bash demo_c3d.sh

# Train, Test and Demo DenseNet 3D
bash train_densenet3d.sh
bash demo_densenet3d.sh

# Train, Test and Demo LeNet 3D
bash train_lenet3d.sh
bash demo_lenet3d.sh

# Train, Test and Demo Mobilenet_v1 3D
bash train_mobilenet3d_v1.sh
bash demo_mobilenet3d_v1.sh

# Train, Test and Demo Mobilenet_v2 3D
bash train_mobilenet3d_v2.sh
bash demo_mobilenet3d_v2.sh

# Train, Test and Demo R(2+1)D
bash train_r2plus1d.sh
bash demo_r2plus1d.sh

# Train, Test and Demo R3D
bash train_r3d.sh
bash demo_r3d.sh

# Train, Test and Demo ResNet 3D
bash train_resnet3d.sh
bash demo_resnet3d.sh

# Train, Test and Demo ResNet + I3D
bash train_resnet_i3d.sh
bash demo_resnet_i3d.sh

# Train, Test and Demo ResNeXt 3D
bash train_resnext3d.sh
bash demo_resnext3d.sh

# Train, Test and Demo ShuffleNet_v1 3D
bash train_shufflenet3d_v1.sh
bash demo_shufflenet3d_v1.sh

# Train, Test and Demo ShuffleNet_v2 3D
bash train_shufflenet3d_v2.sh
bash demo_shufflenet3d_v2.sh

# Train, Test and Demo SlowFast
bash train_slowfast.sh
bash demo_slowfast.sh

# Train, Test and Demo SqueezeNet 3D
bash train_squeezenet3d.sh
bash demo_squeezenet3d.sh

# Train, Test and Demo WideResNet 3D
bash train_wideresnet3d.sh
bash demo_wideresnet3d.sh

# Train, Test and Demo Baseline Nonlocal
bash train_baseline_nonlocal.sh
bash demo_baseline_nonlocal.sh

# Train, Test and Demo ARTNet
bash train_artnet.sh
bash demo_artnet.sh

# Train, Test and Demo S3D
bash train_s3d.sh
bash demo_s3d.sh

# Train, Test and Demo I3D
bash train_i3d.sh
bash demo_i3d.sh

# Train, Test and Demo FstCN
bash train_fstcn.sh
bash demo_fstcn.sh

# Train, Test and Demo LRCN
bash train_lrcn.sh
bash demo_lrcn.sh

# Train, Test and Demo FFC_C3D
bash train_ffc_c3d.sh
bash demo_ffc_c3d.sh

# Train, Test and Demo MnasNet
bash train_mnasnet.sh
bash demo_mnasnet.sh

# Train, Test and Demo Baseline_Spectral_norm
bash train_baseline_spectral_norm.sh
bash demo_baseline_spectral_norm.sh

# Train, Test and Demo I3D_pretrained
bash train_i3d_pretrained.sh
bash demo_i3d_pretrained.sh
```

## MetaAnomaly Model Zoo
TBA.

## Citation
If you find our work useful, please cite the following:
```
@misc{Chi2023,
  author       = {Chi Tran},
  title        = {MetaRecognition: A Unified Framework for Video Action Recognition},
  publisher    = {GitHub},
  booktitle    = {GitHub repository},
  howpublished = {https://github.com/IceIce1ce/MetaRecognition},
  year         = {2023}
}
```

## Contact
If you have any questions, feel free to contact `Chi Tran` 
([ctran743@gmail.com](ctran743@gmail.com)).

##  Acknowledgement
Our framework is built using multiple open source, thanks for their great contributions.
<!--ts-->
* [facebookarchive/C3D](https://github.com/facebookarchive/C3D)
* [kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)
* [okankop/Efficient-3DCNNs](https://github.com/okankop/Efficient-3DCNNs)
* [jfzhang95/pytorch-video-recognition](https://github.com/jfzhang95/pytorch-video-recognition)
* [r1c7/SlowFastNetworks](https://github.com/r1c7/SlowFastNetworks)
* [yangbang18/video-classification-3d-cnn](https://github.com/yangbang18/video-classification-3d-cnn)
* [AlexHex7/Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch)
* [MRzzm/action-recognition-models-pytorch](https://github.com/MRzzm/action-recognition-models-pytorch)
* [christiancosgrove/pytorch-spectral-normalization-gan](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan)
* [pkumivision/FFC](https://github.com/pkumivision/FFC)
* [GowthamGottimukkala/I3D_Feature_Extraction_resnet](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet)
* [Finspire13/pytorch-i3d-feature-extraction](https://github.com/Finspire13/pytorch-i3d-feature-extraction)
<!--te-->
