# [AI Tech 3기 Level 2 P Stage] Data Annotation
<img width="809" alt="화면 캡처 2022-04-23 220932" src="https://user-images.githubusercontent.com/90603530/164895934-5d37ae03-51b5-402d-9e8c-ef0545634e95.png">


# ConVinsight 🧑‍💻
Convenience + Insight : 이용자의 편의를 찾는 통찰력
## Member
| 김나영 | 신규범 | 이정수 | 이현홍 | 전수민 |  
| :-: | :-: | :-: | :-: | :-: |  
|[Github](https://github.com/dudskrla) | [Github](https://github.com/KyubumShin) | [Github](https://github.com/sw930718) | [Github](https://github.com/Heruing) | [Github](https://github.com/Su-minn) |

## Wrap Up Report 📑
💻 [ConVinsight level2-data-annotation Notion](https://yummy-angle-b95.notion.site/CV-05-Wrap-Up-Report-25e5f0ad2ec84a00a6623acde379dfdf)

## Final Score 🏆
- Public f1 score 0.7089 → Private f1 score 0.6790
- Public 6위 → Private 7위
![그림1](https://user-images.githubusercontent.com/90603530/164896072-fc13babf-eb85-4690-88dd-5be5c9569240.jpg)



## Competition Process 🗓️
### Time Line
![2](https://user-images.githubusercontent.com/90603530/164896093-ca491d29-ddb0-4381-8a6d-39e5ade7b3fe.jpg)


### Project Outline 

> batch size 실험
> 
- [x]  batch size 12
- [x]  batch size 16
- [x]  batch size 24
- [x]  batch size 32

> Data Augmentation 실험
> 
- [x]  Blur
- [x]  Color
- [x]  Inverting
- [x]  Outline
- [x]  Rotate

> Optimizer 실험
> 
- [x]  SGD (lr = 1e-3)
- [x]  Adam (lr = 1e-3)
- [x]  AdamW (lr = 1e-3)
- [x]  SGD (lr = 1e-2)

> Learning Rate Scheduler 실험
> 
- [x]  MultiStepLR
- [x]  CosineAnnealingWarmupRestarts
- [x]  CosineAnnealingLR



### Folder Structure 📂
```
level2-data-annotation_cv-level2-cv-05/
│
├── 📂 baseline
│   ├── 📝 convert_mlt.py
│   ├── 📝 scheduler.py
│   ├── 📝 train.py
│   └── 📝 inference.py
│
└── 📂 utils
    ├── 📂 Convert_Custom
    │  	 ├── 📝 from_doc_to_ufo.ipynb
    │ 	 └── 📝 from_doc_to_ufo.py
    ├── 📂 UFO_visualization
    │ 	 └── 📝 data_visualization.ipynb
    └── 📂 multi_label_stratified_kfold
```
