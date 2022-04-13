from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

#get data
nfold = 5
seed = 2022
data_root = '/opt/ml/input/data/ICDAR17_Korean/images'
anno_root = '/opt/ml/input/data/ICDAR17_Korean/ufo/train.json'

categories = {'ko': 0, 'en': 1}

with open(anno_root, 'r') as f:
    train_json = json.load(f)
    images = train_json['images']
    images_df = pd.DataFrame.from_dict(images)

languages = []
words_df = images_df.loc['words']

for index, img in enumerate(words_df):
    for key in img.keys():
        lang = {}
        lang['image_id'] = index
        lang['category_id'] = 0 if img[f'{key}']['language'] == ['ko'] else 1

        languages.append(lang)

x = images
y = [[0] * len(categories) for _ in range(len(images))]

for lang in languages:
    y[lang['image_id']][lang['category_id']] += 1

mskf = MultilabelStratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)

path = './multi_label_stratified_k_fold'

if not os.path.exists(path):
    os.mkdir(path)

for idx, (train_index, val_index) in tqdm(enumerate(mskf.split(x, y)), total=nfold):
    train_dict = dict()
    val_dict = dict()

    train_dict['images'] = train_json['images']
    val_dict['images'] = train_json['images']

    # train dict 
    temp_train = {}
    for index in train_index:
        temp = {}
        image = np.array(list(images.items()))[index]
        key = image[0]
        value = image[1] 

        temp[key] = value
        temp_train.update(temp)
    train_dict['images'] = temp_train

    # val dict 
    temp_val = {}
    for index in val_index:
        temp = {}
        image = np.array(list(images.items()))[index]
        key = image[0]
        value = image[1] 

        temp[key] = value
        temp_train.update(temp)
    val_dict['images'] = temp_val

    train_dir = os.path.join(path, f'cv_train_{idx + 1}.json')
    val_dir = os.path.join(path, f'cv_val_{idx + 1}.json')
    with open(train_dir, 'w') as train_file:
        json.dump(train_dict, train_file)

    with open(val_dir, 'w') as val_file:
        json.dump(val_dict, val_file)

print("Done Make files")