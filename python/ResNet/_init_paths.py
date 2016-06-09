import os
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

caffe_path = osp.join(os.getcwd(), 'python')
print 'caffe/python path :%s' %caffe_path

add_path(caffe_path)
