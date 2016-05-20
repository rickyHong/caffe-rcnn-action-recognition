#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
BUILD=build/examples/FRCNN/test_rpn.bin

$BUILD --gpu 1 \
       --model models/FRCNN/zf_rpn/test.prototxt \
       --weights models/FRCNN/zf_faster_rcnn_final.caffemodel \
       --default_c examples/FRCNN/config/default_config.json \
       --override_c examples/FRCNN/config/voc_config.json \
       --image_root VOCdevkit/VOC2007/JPEGImages/ \
       --image_list examples/FRCNN/dataset/voc2007_test.txt \
       --out_file examples/FRCNN/results/voc2007_test.rpn

CAL_RECALL=examples/FRCNN/calculate_recall.py

python $CAL_RECALL --gt examples/FRCNN/dataset/voc2007_test.txt \
            --answer examples/FRCNN/results/voc2007_test.rpn \
            --overlap 0.5
