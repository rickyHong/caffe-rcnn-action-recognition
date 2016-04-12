function active_caffe_mex()
% active_caffe_mex(gpu_id, caffe_version)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
    cur_dir = pwd;
    caffe_dir = fullfile('../../../matlab');
    addpath(genpath(caffe_dir));
    cd(caffe_dir);
    fprintf(['Current Dir :',pwd,'\n']);
    cd(cur_dir);
    disp('ADD Caffe Mex Done');
end
