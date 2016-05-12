#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
BUILD=build/examples/FRCNN/demo_rpn_api.bin

$BUILD --gpu 0 \
       --model models/FRCNN/vgg16_rpn/test.prototxt \
       --weights models/FRCNN/VGG16_faster_rcnn_final.caffemodel \
       --default_c examples/FRCNN/config/default_config.json \
       --override_c examples/FRCNN/config/rpn_config.json \
       --image_dir examples/FRCNN/images/  \
       --out_dir examples/FRCNN/results/ 
