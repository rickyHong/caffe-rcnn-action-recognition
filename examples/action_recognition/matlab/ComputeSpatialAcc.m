active_caffe_mex();

% test video and its optical flow field
% temporal prediction
% model_def_file = './Developy/ResNet-50-rgb-Test.prototxt';
% model_file = './ResNet-50-model-RGB-Split1_iter_10000.caffemodel'; 
model_def_file = '.prototxt';
model_file = '.caffemodel';
mean_file = 'rgb_mean.mat';
gpu_id = 7;

caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');

[videos , ~ , label] = textread('../dataset_file_examples/val_rgb_split1.txt','%s %d %d');
%Dataset = load('ucfList.mat');
label = label + 1;
%label = Dataset.test1_label;
%videos = Dataset.test1;
predict = zeros( size(label,1) , 1 );
fprintf('Total test videso : %d \n',size(label,1));

accur = 0.0;
for index = 1:size(videos,1)
    tic;
    rgb_video = ['../../../' , videos{index} ];
    spatial_prediction = VideoSpatialPrediction(rgb_video, mean_file, net);
    [ ~ , predict(index) ] = get_ID( spatial_prediction , 3 );
    accur = accur + ( predict(index) == label(index) );
    fprintf('%05d videos predict done in %.3f s :=%.4f || %03d vs %03d \n', index , toc , accur / index, predict(index),label(index));
end

fprintf('Accuracy : %.4f\n' , accur / size(videos,1) );
caffe.reset_all();
