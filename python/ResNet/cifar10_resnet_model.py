#!/usr/bin/env python
import _init_paths
from special_net_spec import layers as L, params as P, to_proto
import caffe
import sys, os, argparse
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate cifar 10 resnet prototxt')
    parser.add_argument('--lmdb_train', dest='lmdb_train',
                        help='train lmdb data',
                        default='examples/cifar10/cifar10_train_lmdb', type=str)
    parser.add_argument('--lmdb_test', dest='lmdb_test',
                        help='test lmdb data',
                        default='examples/cifar10/cifar10_test_lmdb', type=str)
    parser.add_argument('--mean_file', dest='mean_file',
                        help='overlap value',
                        default='examples/cifar10/mean.binaryproto', type=str)
    parser.add_argument('--N', dest='resnet_N',
                        help='resnet_N',
                        default=9, type=int)
    parser.add_argument('--batch_size_train', dest='batch_size_train',
                        help='batch_size_train',
                        default=128, type=int)
    parser.add_argument('--batch_size_test', dest='batch_size_test',
                        help='batch_size_test',
                        default=100, type=int)
    parser.add_argument('--model', dest='model',
                        help='model proto',
                        default='cifar10_resnet_train_val.proto', type=str)
    args = parser.parse_args()

    return args

class Resnet(object):

    def __init__(self, n):
        self.n = n
        self.layers = {}


    def Conv2D(sefl, name, input, num_output, kernal_size = 3, pad = 1, weight_filler = dict(type='xavier'), stride = 1, bias_filler = dict(type='constant',value=0)):

		return L.Convolution(input, name=name, ex_top = [name], kernel_size=kernal_size, pad = pad, stride = stride, num_output=num_output, weight_filler=weight_filler, bias_filler=bias_filler)

    def BatchNorm(sefl, name, input, use_global_stats = True, Has_Scale = True):
		temp = L.BatchNorm(input, name = name+'-bn', ex_top = [name], batch_norm_param = dict(use_global_stats=use_global_stats))
		if Has_Scale:
			return L.Scale(temp, name = name+'-scale', in_place = True, scale_param = dict(bias_term=True))
		else:
			return temp

    def AvePooling(self, name, input, kernel_size = 2, stride = 2):
        return L.Pooling(input, name = name+'-pool', kernel_size = kernel_size, ex_top = [name+'-pool'], stride = stride, pooling_param = dict(pool=P.Pooling.AVE))

    def ReLU(self, name, input):
        return L.ReLU(input, name = name+'-relu', in_place = True)

    def BN_ReLU(self, name, input, use_global_stats = True, Has_Scale = True):
        temp = self.BatchNorm(name, input, use_global_stats, Has_Scale)
        return self.ReLU(name = name, input = temp)

    def Eltwise(self, name, A, B):
        return L.Eltwise(A, B, name = name+'-sum', ex_top = [name], eltwise_param = dict(operation=P.Eltwise.MAX))

    def Residual(self, name, input, out_channel, stride = 1, first=False):
        temp = None
        if not first:
            temp = self.BatchNorm(input = input, name = name+'-B1-')
            temp = self.ReLU(input = temp, name = name+'-B1-')
        else:
            temp = input

        conv1 = self.Conv2D(input = temp, name = name+'-B1-conv', stride = stride, num_output = out_channel)
        bn2   = self.BN_ReLU(input = conv1, name = name+'-B2')
        conv2 = self.Conv2D(input = bn2, name = name+'-B2-conv', num_output = out_channel)

        if stride == 2:
            input = self.Conv2D(input = input, name = name+'-A1-conv', stride = stride, num_output = out_channel)
            #input = self.AvePooling(input = temp, name = name)

        return self.Eltwise(name = name, A = conv2, B = input, )


    def build_cifar_model(self, lmdb_train, lmdb_test, batch_size_train, batch_size_test, mean_file):

        data_, label_ = L.Data(source=lmdb_train, name = 'data', ex_top = ['data','label']
                                                           , backend=P.Data.LMDB, batch_size=batch_size_train, ntop=2, transform_param=dict(mean_file=mean_file)
                                                           , include={'phase':caffe.TRAIN})
        data, label = L.Data(source=lmdb_test, name = 'data', ex_top = ['data','label']
                                                           , backend=P.Data.LMDB, batch_size=batch_size_test, ntop=2, transform_param=dict(mean_file=mean_file)
                                                           , include={'phase':caffe.TEST})
        # Convolution 0
        conv_0 = self.Conv2D(input = data, name = 'conv-0', num_output = 16)
        bn_0   = self.BN_ReLU(input = conv_0, name = 'conv-0', )

        #32*32, c=16
        layer = res_1_0 = self.Residual(input = bn_0, name = 'res1.0', out_channel = 16, first = True)
        for k in range(1, self.n-1):
            layer = self.Residual(input = layer, name = 'res1.{}'.format(k), out_channel = 16)

        #16*16, c=32
        layer = res_2_0 = self.Residual(input = layer, name = 'res2.0', out_channel = 32, stride = 2)
        for k in range(1, self.n-1):
            layer = self.Residual(input = layer, name = 'res2.{}'.format(k), out_channel = 32)

        #8*8, c=64
        layer = res_3_0 = self.Residual(input = layer, name = 'res3.0', out_channel = 32, stride = 2)
        for k in range(1, self.n-1):
            layer = self.Residual(input = layer, name = 'res3.{}'.format(k), out_channel = 32)

        layer = res_post = self.BN_ReLU(input = layer, name = 'res3-post', )

        ## Ave Pooling
        global_pool = self.AvePooling(input = layer, name = 'global', kernel_size = 8, stride = 1)

        fc = L.InnerProduct(global_pool, num_output=10, name = 'fc', ex_top = ['fc'])
        loss = L.SoftmaxWithLoss(fc, label, name = 'softmaxloss', ex_top = ['loss'])
        acc = L.Accuracy(fc, label, name = 'accuracy', ex_top = ['accuracy'])
        return to_proto(data_,label_), to_proto(loss, acc)


if __name__ == '__main__':
    args   = parse_args()
    resnet = Resnet(args.resnet_N)
    train_data, proto  = resnet.build_cifar_model(args.lmdb_train,args.lmdb_test,args.batch_size_train,args.batch_size_test,args.mean_file)
    with open(args.model, 'w') as model:
        #print(proto, model)
        model.write('name: "CIFAR10_Resnet_%d"\n' % (args.resnet_N*6+2))
        model.write('%s\n' % train_data)
        model.write('%s\n' % proto)

    print 'Generate Done'
