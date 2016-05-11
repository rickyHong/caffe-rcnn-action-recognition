cur_dir = pwd;
caffe_dir = fullfile('../../../matlab');
addpath(genpath(caffe_dir));
cd(caffe_dir);
fprintf(['Current Dir :',pwd,'\n']);
cd(cur_dir);
disp('ADD Caffe Mex Done');
% test video and its optical flow field
% temporal prediction
% model_def_file = './Developy/ResNet-50-rgb-Test.prototxt';
% model_file = './ResNet-50-model-RGB-Split1_iter_10000.caffemodel'; 
model_def_file = 'VGG16-RGB-Test.prototxt';
model_file = 'AC_VGG16_Activity_iter_10000.caffemodel';
mean_file = '../matlab/rgb_mean.mat';
gpu_id = 7;

caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');

[videos , ~ , label] = textread('../dataset_file_examples/Val_RGB_Full.txt','%s %d %d');
%Dataset = load('ucfList.mat');
label = label + 1;
%label = Dataset.test1_label;
%videos = Dataset.test1;
predict = zeros( size(label,1) , 200 );
fprintf('Total test videso : %d \n',size(label,1));

accur = 0.0;
assert(size(label,1) == size(videos,1));
for index = 1:size(videos,1)
    tic;
    rgb_video = ['../../../Val_Extract/' , videos{index} ];
    spatial_prediction = ActionSpatialPrediction(rgb_video, mean_file, net);
    %[ ~ , predict(index) ] = get_ID( spatial_prediction , 3 );
    spatial_prediction = sum(spatial_prediction, 2);
    predict(index, : ) = spatial_prediction;
    [~, predict_cur] = max(spatial_prediction);
    %[~, predict(index)] = max(spatial_prediction);
    accur = accur + ( predict_cur == label(index) );
    fprintf('%05d videos predict done in %.3f s :=%.4f || %03d vs %03d \n', index , toc , accur / index, predict_cur,label(index));
end

save('VGG-Val-Full-Predict.mat','predict','videos','label','-v7.3');
fprintf('Accuracy : %.4f\n' , accur / size(videos,1) );
caffe.reset_all();
