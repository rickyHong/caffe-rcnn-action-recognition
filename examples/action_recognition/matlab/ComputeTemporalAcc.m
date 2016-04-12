active_caffe_mex();

% test video and its optical flow field
% temporal prediction
% model_def_file = './ResNet-50-rgb-Test.prototxt';
% model_file = './ResNet-50-model-RGB-Split1_iter_10000.caffemodel'; 
model_def_file = 'prototxt' ;
model_file = '.caffemodel';
mean_file = 'flow_mean.mat';
gpu_id = 6;

caffe.reset_all();
caffe.set_device(gpu_id);
caffe.set_mode_gpu();
net = caffe.Net(model_def_file, model_file, 'test');

[videos , ~ , label] = textread('../dataset_file_examples/val_flow_split1.txt','%s %d %d');
label = label + 1;
%Dataset = load('ucfList.mat');
%label = Dataset.test1_label;
%videos = Dataset.test1;
predict = zeros( size(label,1) , 1 );
fprintf('Total test videso : %d \n',size(videos,1));

accur = 0.0;
for index = 1:size(videos,1)
    tic;
    video_flow = ['../../../' , videos{index} ];
    temporal_prediction = VideoTemporalPrediction(video_flow, mean_file, net);
    [ ~ , predict(index) ] = get_ID( temporal_prediction , 3 );
    accur = accur + ( predict(index) == label(index) );
    fprintf('%05d videos predict done in %.3f s :=%.4f || %03d vs %03d \n', index , toc , accur / index, predict(index),label(index));
end

fprintf('Accuracy : %.4f\n' , accur / size(videos,1) );
caffe.reset_all();
