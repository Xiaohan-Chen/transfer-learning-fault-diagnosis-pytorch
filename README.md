## :book: 1. Introduction
This respository contains popular deep learning algorithms implemented for fault diagnosis tasks: general classification, domain adaptation, open-set domain adaptation, few-shot learning, self-supervised learning, knowledge distillation.

- [x] General classification: traing and test apply the same machines, working conditions and faults.

- [x] *domain adaptation*: the distribution of the source domain data may be different from the target domain data, but the label set of the target domain is the same as the source domain, i.e., $\mathcal{D} _{s}=(X_s,Y_s)$, $\mathcal{D} _{t}=(X_t,Y_t)$, $X_s \ne X_t$, $Y_s = Y_t$.
  - [x] [Deep Domain Confusion (DDC)](https://arxiv.org/pdf/1412.3474.pdf) (including [DeepCORAL](https://arxiv.org/abs/1607.01719))
  - [x] [Unsupervised Domain Adaptation by Backpropagation(DANN)](http://proceedings.mlr.press/v37/ganin15.pdf)

- [ ] *Open-set domain adaptation*: the distribution of the source domain data may be different from the target domain data. What's more, the target label set contains unknown categories, i.e., $\mathcal{D} _{s}=(X_s,Y_s)$, $\mathcal{D} _{t}=(X_t,Y_t)$, $X_s \ne X_t$, $Y_s \in Y_t$. We refer to their common categories $\mathcal{Y}_s\cap \mathcal{Y}_t$ as the *known classes*, and $\mathcal{Y}_s\setminus \mathcal{Y}_t$ (or $\mathcal{Y}_t\setminus \mathcal{Y}_s$) in the target domain as the *unknown class*.
  - [x] [Open Set Domain Adaptation by Backpropagation (OSDABP)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kuniaki_Saito_Adversarial_Open_Set_ECCV_2018_paper.pdf)

- [ ] *Few-shot learning*: Compared with standard classification, few-shot classification tasks require we have only few training samples of each class.
  - [ ] [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) [[*Code (unofficial)*]](https://github.com/fangpin/siamese-pytorch)

- [ ] *Self-supervised learning*: 

- [ ] *Universal domain adaptation*: the distribution of the source domain domain data may be different from the target domain. In addition, the target label set is unknwon, and we do not known the relationships between the source label set and the target label set.
  - [ ] TBD
  - [ ] TBD

---
## :wrench: 2. Requirements
- python 3.9.12
- Numpy 1.23.1
- pytorch 1.12.0
- scikit-learn 1.1.1
- torchvision 0.13.0

---
## :handbag: 3. Datasets
- [CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter/welcome)
- [PU Bearing Dataset](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/)
- [PHM09 Data Challenge Gearbox Dataset](https://phmsociety.org/data-analysis-competition/)(can not access)
- [FEMTO-ST Bearing Dataset (PHM12)](https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset)

---
## :pencil: 4. Usage
> If using pre-trained models to initialize the backbone and classifier in transfer learning tasks, run classification tasks firstly to generate corresponding checkpoints.

**General Classification task:**
- Train and test the model on same machines, working conditions and faults. Using following commands:
```python
python3 main.py --task "CLS" --backbone "MLPNet"
```

**Transfer Learning:**
- If using the DDC transfer learning method, applying following commands:
```python
python3 main.py --task "DDC" --backbone "CNN1D" --max_epoch 500
```
- If using the DANN transfer learning method, applying following commands:
```python
python3 main.py --task "DANN" --backbone "CNN1D" --max_epoch 500
```

**Open Set Domain Adaptation:**
- General Classification task:

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

Dataset: PU
Load: 2
Label set: [0,1,2,3,4,5,6,7,8,9,10,11,12]

|                 | MLPNet | CNN1D | ResNet1D | ResNet2D |
|:---------------:|:------:|:-----:|:--------:|:--------:|
|acc (time domain)|  83.10 | 86.71 |   99.95  |   97.57  |
|acc (freq domain)|  99.95 | 88.93 |   99.97  |   99.80  |

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

Dataset: PU
Source load: 3  
Target Load: 2  
Label set: [0,1,2,3,4,5,6,7,8,9,10,11,12]  
Pre-trained model: False

Time domain:  
|           | MLPNet | CNN1D | ResNet1D | ResNet2D |
|:---------:|:------:|:-----:|:--------:|:--------:|
|DDC (linear kernel)| 64.46  | 78.53 | 98.11  |  92.84 |
|    DANN   | 46.65  | 38.14 |  63.24  |  75.82   |

Frequency domain
|           | MLPNet | CNN1D | ResNet1D | ResNet2D |
|:---------:|:------:|:-----:|:--------:|:--------:|
| DDC (linear kernel) |  99.92 | 86.63 |   99.59  |   99.28  |
|    DANN   |  87.35 | 58.21 |  92.43  |   77.25  |

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
## :triangular_ruler: 6. Visualization:


---
## :bulb: 7. References:

> @misc{tllib,
    author = {Junguang Jiang, Baixu Chen, Bo Fu, Mingsheng Long},
    title = {Transfer-Learning-library},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/thuml/Transfer-Learning-Library}},
}

> @misc{Zhao2019,
author = {Zhibin Zhao and Qiyang Zhang and Xiaolei Yu and Chuang Sun and Shibin Wang and Ruqiang Yan and Xuefeng Chen},
title = {Unsupervised Deep Transfer Learning for Intelligent Fault Diagnosis},
year = {2019},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/ZhaoZhibin/UDTL}},
}


