#!/usr/bin/env bash

wget https://datasets.cvc.uab.es/rrc/ImagesPart1.zip --no-check-certificate
unzip ImagesPart1.zip -d /opt/ml/input/data/ICDAR19_MLT/raw/training_images
wget https://datasets.cvc.uab.es/rrc/ImagesPart2.zip --no-check-certificate
unzip ImagesPart2.zip -d /opt/ml/input/data/ICDAR19_MLT/raw/training_images
wget https://datasets.cvc.uab.es/rrc/train_gt_t13.zip --no-check-certificate
unzip train_gt_t13.zip -d /opt/ml/input/data/ICDAR19_MLT/raw/training_gt
rm *.zip