import pandas as pd
import os, glob, json
from tqdm import tqdm

anno_root = '/opt/ml/input/data/documents/annotations'
img_root = '/opt/ml/input/data/documents/images'
file_list = sorted(glob.glob(os.path.join(anno_root, '*.json')))
jpg_file_name = glob.glob(os.path.join(img_root, '*.jpg'))
JPG_file_name = glob.glob(os.path.join(img_root, '*.JPG'))

image_file_dir = sorted(jpg_file_name + JPG_file_name)
image_file_name = [img.split("/")[-1] for img in image_file_dir]

def check_kor_eng(text : str):
    kor, eng = 0, 0

    for c in text:
        if ord('가') <= ord(c) <= ord('힣'):
            kor += 1
        elif ord('a') <= ord(c.lower()) <= ord('z'):
            eng += 1
    
    answer = []
    if kor > 0:
        answer.append('ko')
    if eng > 0:
        answer.append('en')
    
    return answer
    
# convert to UFO format
temp_images = {}

for i, file in enumerate(tqdm(file_list)):
    anno = {}

    with open(file, 'r') as f:
        train_json = json.load(f)
        images = train_json['images']
        annotations = train_json['annotations']

        images_df = pd.DataFrame.from_dict(images)
        annotations_df = pd.DataFrame.from_dict(annotations)

        temp_anno = {}

        temp_anno['img_h'] = int(images_df['image.height'].values)
        temp_anno['img_w'] = int(images_df['image.width'].values)

        word = {}
        for index in annotations_df.index:
            temp_word = {}

            row = annotations_df.loc[index]

            idx = int(row['id'])
            transcription = row['annotation.text']
            illegibility = False
            orientation = 'Horizontal'
            word_tags = 'null'

            language = check_kor_eng(transcription)

            x, y, w, h = row['annotation.bbox']

            points = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]

            temp_word['points'] = points
            temp_word['transcription'] = transcription
            temp_word['language'] = language
            temp_word['illegibility'] = illegibility
            temp_word['orientation'] = orientation
            temp_word['word_tags'] = word_tags

            word[idx] = temp_word


        temp_anno['words'] = word
        temp_anno['tags'] = None

        img_name = image_file_name[i]

        anno[f'{img_name}'] = temp_anno

    temp_images.update(anno)
    
UFO_ann = {}
UFO_ann['images'] = temp_images

file_path = "/opt/ml/input/data/documents/annotation.json"

with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(UFO_ann, file, indent="\t")

print("Done make files!")
