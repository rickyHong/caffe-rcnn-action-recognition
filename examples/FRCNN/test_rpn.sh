#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
BUILD=build/examples/FRCNN/test_rpn.bin

$BUILD --gpu 1 \
       --model models/FRCNN/vgg16_rpn/test.prototxt \
       --weights models/FRCNN/VGG16_faster_rcnn_final.caffemodel \
       --default_c examples/FRCNN/config/default_config.json \
       --override_c examples/FRCNN/config/rpn_config.json \
       --image_root VOCdevkit/VOC2007/JPEGImages/ \
       --image_list examples/FRCNN/dataset/voc2007_val.txt \
       --out_file examples/FRCNN/results/voc2007_val_predict_rpn.txt

CAL_RECALL=examples/FRCNN/calculate_recall.py

python $CAL_RECALL --gt examples/FRCNN/dataset/voc2007_val.txt \
            --answer examples/FRCNN/results/voc2007_val_predict_rpn.txt \
            --overlap 0.5
