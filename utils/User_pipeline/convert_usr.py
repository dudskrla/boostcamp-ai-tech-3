import os.path as osp
import json
import os
from tqdm import tqdm
import copy


def maybe_mkdir(x):
    if not osp.exists(x):
        os.makedirs(x)


del_index = [7, 14, 19, 25, 34, 72, 73, 75, 107, 109, 110, 114, 146, 150, 157, 206, 222, 240, 260, 266, 271, 274, 276,
             317, 319, 326, 337, 351, 375, 388, 389, 392, 396, 399, 400, 402, 422, 460, 475, 488, 517, 525, 539, 541,
             552, 578, 597, 598, 604, 605, 617, 626, 629, 630, 631, 638, 645, 663, 694, 695, 701, 705, 711, 720, 759,
             784, 809, 819, 828, 835, 838, 846, 870, 888, 892, 901, 904, 951, 956, 960, 994, 1009, 1031, 1105, 1113,
             1141, 1145, 1149, 1150, 1151, 1152, 1155, 1159, 1161, 1234, 1284]
# del_index = [11, 30, 40, 49, 50, 56, 66, 87, 104, 137, 216, 229, 323, 366, 393, 413, 424, 629, 820, 869, 1151]

add_data_dir = '/opt/ml/input/data/Camper_dataset/dataset'

with open(osp.join(add_data_dir, 'ufo/{}.json'.format('train')), 'r') as f:
    anno = json.load(f)

anno = anno['images']

anno_temp = copy.deepcopy(anno)

count = 0
count_normal = 0
count_none_anno = 0

for i, (img_name, img_info) in enumerate(tqdm(anno.items())):
    if img_info['words'] == {} or i in del_index:
        del (anno_temp[img_name])
        count_none_anno += 1
        continue
    for obj, obj_info in img_info['words'].items():
        if len(img_info['words'][obj]['points']) == 4:
            count_normal += 1
            continue
        elif len(img_info['words'][obj]['points']) < 4:
            del (anno_temp[img_name]['words'][obj])
        else:
            over_polygon_temp = copy.deepcopy(anno_temp[img_name]['words'][obj])
            point_len = len(img_info['words'][obj]['points'])
            for index in range(point_len // 2 - 1):
                poly_region = []
                poly_region.append(img_info['words'][obj]['points'][index])
                poly_region.append(img_info['words'][obj]['points'][index + 1])
                poly_region.append(img_info['words'][obj]['points'][-index - 2])
                poly_region.append(img_info['words'][obj]['points'][-index - 1])
                over_polygon_temp['points'] = poly_region
                anno_temp[img_name]['words'][obj + f'{index + point_len}'] = copy.deepcopy(over_polygon_temp)
            del anno_temp[img_name]['words'][obj]
        if anno_temp[img_name]['words'] == {}:
            del (anno_temp[img_name])
            count_none_anno += 1
            continue
        count += 1

print(f'normal polygon count : {count_normal}')
print(f'deleted {count} over polygon')
print(count_none_anno)

anno = {'images': anno_temp}

ufo_dir = osp.join(add_data_dir, 'ufo')
maybe_mkdir(ufo_dir)
with open(osp.join(ufo_dir, 'train.json'), 'w') as f:
    json.dump(anno, f, indent=4)
