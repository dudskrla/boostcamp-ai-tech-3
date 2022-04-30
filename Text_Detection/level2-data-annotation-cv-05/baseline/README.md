# Baseline Files

## 1. train.py
```
python train.py --name example 
```
### **Arguments**
```
--data_dir : 데이터 경로  
--model_dir : model checkpoint를 저장할 폴더 경로 (pth가 덮어씌워질 경우를 방지하기 위하여 폴더가 이미 존재할 경우 Error 발생)  
--device : cuda  
--num_workers : num_workers [Default 4]  
--image_size : 리사이즈할 이미지 크기 [Default 1024]  
--input_size : model input size [Default 512]  
--batch_size : batch_size [Default 12]  
--learning_rate : learning rate [Default 1e-3]  
--save_interval : Save 파일의 생성 주기  
--log_interval : wandb log를 남기는 빈도  
--no-val : val을 스킵하기위한 설정 (현재 validate 미구현)  
--name : wandb Name 설정을 위한 Argument, [Default None]  
--seed : seed 설정, [Default 2022]  
```

## 2. inference.py
inference 함수  
직접적으로 사용할 일은 없음  
제출시 argument 부분에 --model_dir [pth_파일경로] 입력 필요  

## 3. convert_mlt.py
ICDAR17로 만드는 Dataset  
현재 팀에서는 3가지 종류의 filter를 적용하여 Dataset을 생성하였음
```
Korean : 기존 aistage에서 제공한 Default Dataset
Korean2 : 전체 Dataset에서 Korean 이 포함된 모든 이미지 1063으로 이루어진 Dataset
Custom : Korean or English가 포함된 모든 이미지 약 7800장으로 이루어진 Dataset
```
50번째 line 수정을 통하여 만들어지는 Dataset을 조절할 수 있음
