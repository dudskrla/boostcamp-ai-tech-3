import os
import cv2
import random
from glob import glob
from tqdm import tqdm
import albumentations as A


def paste_bg_patch(img_dir, x, y, bg_img):
    transform = A.Compose([
        A.RandomRotate90(p=0.2)
    ])
    img = cv2.imread(img_dir, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(img_dir.replace('image', 'mask'))
    output = transform(image=img, mask=mask)
    img, mask = output['image'], output['mask']
    roi = bg_img[y:y+512, x:x+512]
    cv2.copyTo(img, mask, roi)
    return bg_img


def create_rand_len(eps):
    wh = [[], []]
    for i in range(2):
        for _ in range(3):
            wh[i].append(0)
            f = random.randint(100+eps, 512)
            s = random.randint(512-f+eps, 512)
            wh[i].append(f)
            wh[i].append(f+s)
    ep_w = random.randint(0, eps)
    ep_h = random.randint(0, eps)
    return [wh[0][0], wh[0][3], wh[0][6], wh[0][1], wh[0][4], wh[0][7], wh[0][2], wh[0][5], wh[0][8]], wh[1], ep_w, ep_h


def main():
    default_dir = '/opt/ml/detection/dataset'
    os.makedirs(os.path.join(default_dir, 'bg'), exist_ok=True)
    image_list = glob(os.path.join(default_dir, 'bg_patch/image/*.jpg'))
    random.shuffle(image_list)
    for i in tqdm(range(len(image_list)//9), total=len(image_list)//9):
        files = []
        for _ in range(9):
            files.append(image_list.pop())
        bg_img = cv2.imread(os.path.join(default_dir, 'null.jpg'))
        w, h, ep_w, ep_h = create_rand_len(20)
        anchors = [[i, j] for (i, j) in zip(w, h)]
        random.shuffle(anchors)
        for file, anchor in zip(files[:9], anchors):
            paste_bg_patch(file, anchor[0], anchor[1], bg_img)
        img = bg_img[0 + ep_w:1024 + ep_w, 0 + ep_h:1024 + ep_h]
        cv2.imwrite(os.path.join(default_dir, 'bg', str(i) + '.jpg'), img)


if __name__=='__main__':
    main()
