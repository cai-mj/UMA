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

### Running the code
To run the complete experiment, after preparing the data, run
```
python train_bayesian_adapt.py --dataset name_of_target_dataset --batchSize number_of_batchsize_for_adaptation
```
This would adapt the segmentation model (pre-trained on EGTEA dataset) to the target dataset.

## Contact
For any question, please contact
```
Minjie Cai: caiminjie@hnu.edu.cn
```
