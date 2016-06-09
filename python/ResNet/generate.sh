# This shell give an example to show how to set params in cifar10_resnet_model.py
# cifar10_resnet_model.py is a python shell to generate resnet 
# Using the "Identity Mappings"
# n = {3,5,7,9,18} for layers = {20,32,44,56,110}
#
# Details in the following two papers.
# [Deep Residual Learning for Image Recognition](http://arxiv.org/pdf/1512.03385v1.pdf)
# [Identity Mappings in Deep Residual Networks](http://arxiv.org/pdf/1603.05027v2.pdf)
#
# Please execute this shell in the caffe root dir
python ./python/ResNet/cifar10_resnet_model.py \
    --lmdb_train examples/cifar10/cifar10_train_lmdb \
    --lmdb_test  examples/cifar10/cifar10_test_lmdb  \
    --mean_file  examples/cifar10/mean.binaryproto   \
    --N          9 \
    --batch_size_train 128 \
    --batch_size_test  100 \
    --model      examples/cifar10_resnet/cifar10_resnet56_train_val.proto
