# 3d-target-inference

## Real-time 3D Target Inference via Biomechanical Simulation (CHI'24)

<img src="figs/simul_dense.gif" width="30%"> <img src="figs/simul_wide.gif" width="30%">

- ### Simulated user with human-like perceptual and motor skills
  - We created a rational agent with human-like perception and motor abilities, trained via reinforcement learning.
  - The simulation adapts to various target configurations and user features.
  - Our simulated user replicated the task speed, accuracy, and motor variability of human users (N=20).
  - Using only simulated data, our inference accuracy equaled that achieved with data from seven human users.
  - The trained inference model enhanced selection speed and accuracy for human users in VR.
- ### [Project page](https://hsmoon121.github.io/projects/chi24-target-inference/index.html)
- ### [Paper (Open access)](https://dl.acm.org/doi/10.1145/3613904.3642131)
- ### [Presentation video (YouTube link)](https://youtu.be/AIL9BGkmlXA?si=sElVnOeKDMSfxG9L)

## Datasets & Code Release Overview

- Datasets
  - We provide datasets from user studies conducted on raycasting selection trials. The datasets are as follows:
  - [data/study_1](data/study_1) - Raycasting selection trials (N=20)
  - [data/study_3a](data/study_3a) - Raycasting selection trials (Wide target grid) with assistance (N=12)
  - [data/study_3b](data/study_3b) - Raycasting selection trials (Dense target grid) with assistance (N=20)
- Key results (Figures) replication
  - We provide scripts to replicate the key results presented in our paper. The replication code for plots can be found in the [plots/](plots) directory.
- Code for inference model
  - The code for training the inference model is available in this repository.
- Code for simulator model
  - We are currently refactoring the code for the simulator model. The refactored code will be available soon.
- Tutorials
  - We are also working on adding more tutorial code to help users understand and utilize our models and datasets.

## Acknowledgements

The biomechanical agent training part of this code utilizes the previous code from Ikkala et al. (
Breathing Life Into Biomechanical User Models, UIST'22 paper). 

For more details and citation, please refer to the respective repository: https://github.com/aikkala/user-in-the-box

## Citation

- Please cite this paper as follows if you use this code in your research.

```
@inproceedings{moon2024real,
title={Real-time 3D Target Inference via Biomechanical Simulation},
author={Moon, Hee-Seung and Liao, Yi-Chi and Li, Chenyu and Lee, Byungjoo and Oulasvirta, Antti},
booktitle={Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems},
year={2024},
publisher = {Association for Computing Machinery},
url = {https://hsmoon121.github.io/projects/chi24-target-inference},
doi = {10.1145/3613904.3642131},
location = {Honolulu, Hi, USA},
series = {CHI '24}
}
```
