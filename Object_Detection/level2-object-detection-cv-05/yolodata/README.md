# yolodata
## folder structure
```
π yolodata/
β
βββ π images 
β	βββ π train
β	βββ π test
βββ π labels
β
βββ π train.txt 
βββ π val.txt 
βββ π test.txt 
```

## labels ν΄λ μμ± 
- [convert2yolo](https://github.com/ssaru/convert2Yolo) μ¬μ©
```bash
git clone https://github.com/ssaru/convert2Yolo.git
cd convert2Yolo
```
- names.txt νμΌ μμ± 
```bash
# convert2Yolo/names.txt

General trash
Paper
Paper pack
Metal
Glass
Plastic
Styrofoam
Plastic bag
Battery
Clothing
```
- convert2yolo μ€ν μμ
```bash
python3 example.py --datasets COCO \
--img_path /opt/ml/detection/dataset/train/ \
--label /opt/ml/detection/dataset/train.json \
--convert_output_path /opt/ml/detection/yolodata/labels/ \
--img_type ".jpg" \
--manipast_path ./ \
--cls_list_file names.txt
```
- labels ν΄λ
  - μμ±ν labels ν΄λλ₯Ό yolodata ν΄λλ‘ μ΄λ

## txt νμΌ μμ±

- coco2yolo νμΌ μμ± ν μ€ν
```bash
python coco2yolo.py [train.json νμΌ κ²½λ‘]
python coco2yolo.py [val.json νμΌ κ²½λ‘]
python coco2yolo.py [test.json νμΌ κ²½λ‘]
```
- [train.json νμΌλͺ].txt -> train.txtλ‘ μ΄λ¦ λ³κ²½ 
- [val.json νμΌλͺ].txt -> val.txtλ‘ μ΄λ¦ λ³κ²½ 

- txt νμΌ
  - μΈ txt νμΌ (train.txt, val.txt, test.txt) λͺ¨λ yolodata ν΄λλ‘ μ΄λ

## images ν΄λ μμ± 
- image ν΄λ 
	- 1 ) (datasets ν΄λμ μλ) train, test μ΄λ―Έμ§ ν΄λλ₯Ό λ³΅μ¬ν΄μ
	- 2 ) yolodata ν΄λμ images ν΄λ λ΄λΆμ λΆμ¬λ£κΈ°
