# Deep transfer learing-based bearing fault diagnosis

## :book: 1. Introduction
This respository contains popular deep transfer learning algorithms implemented for cross-load fault diagnosis transfer tasks via PyTorch, including: 

- [x] General classification task: traing and test apply the same machines, working conditions and faults.

- [x] *domain adaptation*: the distribution of the source domain data may be different from the target domain data, but the label set of the target domain is the same as the source domain, i.e., $\mathcal{D} _{s}=(X_s,Y_s)$, $\mathcal{D} _{t}=(X_t,Y_t)$, $X_s \ne X_t$, $Y_s = Y_t$.
  - [x] [Deep Domain Confusion (DDC)](https://arxiv.org/pdf/1412.3474.pdf) (including [DeepCORAL](https://arxiv.org/abs/1607.01719))
  - [x] [Unsupervised Domain Adaptation by Backpropagation(DANN)](http://proceedings.mlr.press/v37/ganin15.pdf)
  - [ ] TODO

- [x] *Open-set domain adaptation*: the distribution of the source domain data may be different from the target domain data. What's more, the target label set contains unknown categories, i.e., $\mathcal{D} _{s}=(X_s,Y_s)$, $\mathcal{D} _{t}=(X_t,Y_t)$, $X_s \ne X_t$, $Y_s \in Y_t$. We refer to their common categories $\mathcal{Y}_s\cap \mathcal{Y}_t$ as the *known classes*, and $\mathcal{Y}_s\setminus \mathcal{Y}_t$ (or $\mathcal{Y}_t\setminus \mathcal{Y}_s$) in the target domain as the *unknown class*.
  - [x] [Open Set Domain Adaptation by Backpropagation (OSDABP)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kuniaki_Saito_Adversarial_Open_Set_ECCV_2018_paper.pdf)
  - [ ] TODO

If you find this repository useful and apply it in your works, please cite the following reference, thanks~:
```
@ARTICLE{10042467,
  author={Chen, Xiaohan and Yang, Rui and Xue, Yihao and Huang, Mengjie and Ferrero, Roberto and Wang, Zidong},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Deep Transfer Learning for Bearing Fault Diagnosis: A Systematic Review Since 2016}, 
  year={2023},
  volume={72},
  number={},
  pages={1-21},
  doi={10.1109/TIM.2023.3244237}}
```


---
## :wrench: 2. Requirements
- python 3.9.12
- Numpy 1.23.1
- pytorch 1.12.0
- scikit-learn 1.1.1
- torchvision 0.13.0

---
## :handbag: 3. Datasets
Download the bearing dataset from [CWRU Bearing Dataset Centre](https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data) and place the `.mat' files in the `./dataset' folder according to the following structure:
```
dataset/
  └── CWRU/
      ├── Drive_end_0/
      │   └── 97.mat 109.mat 122.mat 135.mat 173.mat 189.mat 201.mat 213.mat 226.mat 238.mat
      ├── Drive_end_1/
      │   └── 98.mat 110.mat 123.mat 136.mat 175.mat 190.mat 202.mat 214.mat 227.mat  239.mat
      ├── Drive_end_2/
      │   └── 99.mat 111.mat 124.mat 137.mat 176.mat 191.mat 203.mat 215.mat 228.mat 240.mat
      └── Drive_end_3/
          └── 100.mat 112.mat 125.mat 138.mat 177.mat 192.mat 204.mat 217.mat 229.mat 241.mat
```

---
## :pencil: 4. Usage
> **NOTE**: When using pre-trained models to initialise the backbone and classifier in transfer learning tasks, run classification tasks first to generate corresponding checkpoints.

Four typical neural networks are implemented in this repository, including *MLP*, *1D CNN*, *1D ResNet18*, and *2D ResNet18* (torchvision package). More details can be found in the `./Backbone` folder.

**General Classification task:**
- Train and test the model on same machines, working conditions and faults. Using following commands:
```python
python3 classification.py --datadir './datasets' --max_epoch 100
```

**Transfer Learning:**
- If using the DDC transfer learning method, applying following commands:
```python
python3 DDC.py --datadir './datasets' -backbone "CNN1D" --pretrained False --kernel 'Linear'
```
- If using the DeepCORAL transfer learning method, applying following commands:
```python
python3 DDC.py --datadir './datasets' -backbone "CNN1D" --pretrained False --kernel 'CORAL'
```
- If using the DANN transfer learning method, applying following commands:
```python
python3 DANN.py --backbone "CNN1D"
```

**Open Set Domain Adaptation:**
- The target domain contains unknow classes, applying following commands:
```python
python OSDABP.py
```
---
## :flashlight: 5. Results
> The following results do not represent the best results.

**General Classification task:**
Dataset: CWRU
Load: 3  
Label set: [0,1,2,3,4,5,6,7,8,9]  

|                 | MLPNet | CNN1D | ResNet1D | ResNet2D |
|:---------------:|:------:|:-----:|:--------:|:--------:|
|acc (time domain)|  93.95 | 97.70 |   99.58  |   98.02  |
|acc (freq domain)|  99.95 | 99.44 |   100.0  |   99.96  |

**Transfer Learning:**
Dataset: CWRU  
Source load: 3  
Target Load: 2  
Label set: [0,1,2,3,4,5,6,7,8,9]  
Pre-trained model: True  

Time domain:  
|           | MLPNet | CNN1D | ResNet1D | ResNet2D |
|:---------:|:------:|:-----:|:--------:|:--------:|
|DDC (linear kernel)|  75.47 | 85.53 |   91.79  |   91.32  |
| DeepCORAL |  82.33 | 88.23 |   93.88  |   90.84  |
|    DANN   |  87.68 | 94.77 |   98.88  |   93.95  |

Frequency domain
|           | MLPNet | CNN1D | ResNet1D | ResNet2D |
|:---------:|:------:|:-----:|:--------:|:--------:|
| DeepCORAL |  98.65 | 98.22 |   99.75  |   99.31  |
|    DANN   |  99.38 | 98.74 |   99.89  |   99.47  |

**Open Set Domain Adaptation**
- *OSDABP*
Dataset: CWRU  
Source load: 3  
Target Load: 2  
Source label set: [0,1,2,3,4,5]  
Target label set: [0,1,2,3,4,5,6,7,8,9]  
Pre-trained model: True  

|   Label  |   0   |   1   |   2   |   3   |   4   |   5   |  unk  | All   | Only known |
|:--------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|-------|------------|
|  MLPNet  | 99.83 | 95.96 | 59.76 | 76.10 | 19.85 | 96.58 | 59.21 | 70.21 | 75.99      |
|   CNN1D  | 100.0 | 94.95 | 94.47 | 99.08 | 47.31 | 74.32 | 26.36 | 61.75 | 85.35      |
| ResNet1D | 100.0 | 100.0 | 80.14 | 100.0 | 43.32 | 93.49 | 45.22 | 70.04 | 86.58      |
| ResNet2D | 100.0 | 100.0 | 94.82 | 100.0 | 18.55 | 98.12 | 53.42 | 72.95 | 85.96      |


---
## :camping: 6. See also
- Multi-scale CNN and LSTM bearing fault diagnosis [[paper](https://link.springer.com/article/10.1007/s10845-020-01600-2)][[GitHub](https://github.com/Xiaohan-Chen/baer_fault_diagnosis)]

## :globe_with_meridians: 7. Acknowledgement

```
@article{zhao2021applications,
  title={Applications of Unsupervised Deep Transfer Learning to Intelligent Fault Diagnosis: A Survey and Comparative Study},
  author={Zhibin Zhao and Qiyang Zhang and Xiaolei Yu and Chuang Sun and Shibin Wang and Ruqiang Yan and Xuefeng Chen},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2021}
}
```