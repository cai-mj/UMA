# Code for the paper "Generalizing Hand Segmentation in Egocentric Videos with Uncertainty-Guided Model Adaptation" (CVPR2020)

This is the github repository containing the code for the paper "Generalizing Hand Segmentation in Egocentric Videos with 
Uncertainty-Guided Model Adaptation" by Minjie Cai, Feng Lu and Yoichi Sato.

Currently, the code is under update.

## Code usage

### Requirements
The code is tested to work correctly with:
- GPU environment
- Anaconda Python 3.7
- [Pytorch](https://pytorch.org/) v0.4.0
- NumPy
- OpenCV
- tqdm

### Dataset preparation
We use several different datasets for model adaptation (or domain adaptation), including: EGTEA, GTEA, EDSH2, EDSH-Kitchen, UTG, YHG, and EgoHands. The UTG and Yale_Human_Grasp datasets are prepared by the paper and included in the repository. Other datasets are public and please download by yourself and arrange them in filefolders like UTG.

### Pre-trained model
You can download the hand segmentation model (bayes_rf101_egtea_00040.pth.tar) pre-trained on EGTEA dataset from [BaiduDrive](https://pan.baidu.com/s/1DNFK_kFZc_Z0nQhOliCK0w) code: rvch
Please put the model inside the filefolder according to the filepath of the code.

### Running the code
To run the complete experiment, after preparing the data, run
```
python train_bayesian_adapt.py --dataset name_of_target_dataset --batchSize number_of_batchsize_for_adaptation
```
This would adapt the pre-trained segmentation model to the target dataset.

## Citation
Please cite the following paper if you feel this repository useful.
```
@inproceedings{cai2020generalizing,
  author    = {Minjie Cai and
               Feng Lu and
               Yoichi Sato},
  title     = {Generalizing Hand Segmentation in Egocentric Videos with Uncertainty-Guided Model Adaptation},
  booktitle   = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020},
}
```

## Contact
For any question, please contact
```
Minjie Cai: caiminjie@hnu.edu.cn
```
