active_caffe_mex();
gpu_id = 7;
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);

Spatial.mean_file = 'rgb_mean.mat';
Spatial.model_def_file = '.prototxt';
Spatial.model_file = '.caffemodel';
Spatial.net = caffe.Net(Spatial.model_def_file, Spatial.model_file, 'test');
[Spatial.videos , ~ , Spatial.label] = textread('../dataset_file_examples/val_rgb_split1.txt','%s %d %d');
Spatial.label = Spatial.label + 1;

Temporal.model_def_file = '.prototxt' ;
Temporal.model_file = '.caffemodel';
Temporal.mean_file = 'flow_mean.mat';
Temporal.net = caffe.Net(Temporal.model_def_file, Temporal.model_file, 'test');
[Temporal.videos , ~ , Temporal.label] = textread('../dataset_file_examples/val_flow_split1.txt','%s %d %d');
Temporal.label = Temporal.label + 1;

%CHECK Temporal amd Spatial read file int the same order 
assert( length(Temporal.label) == length(Spatial.label) )
assert( all(Temporal.label==Spatial.label) );
label = Temporal.label;
%predict = zeros( size(Spatial.label,1) , 1 );
fprintf('Total test videso : %d \n',size(Spatial.label,1));

accur = [0.0,0.0,0.0];
total = size(Spatial.label,1);

for index = 1:total
    tic;
    rgb_video = ['../../../' , Spatial.videos{index} ];
    spatial_prediction = VideoSpatialPrediction(rgb_video, Spatial.mean_file, Spatial.net);
    video_flow = ['../../../' , Temporal.videos{index} ];
    temporal_prediction = VideoTemporalPrediction(video_flow, Temporal.mean_file, Temporal.net);
    [ sans , sL ] = get_ID( spatial_prediction , 3 );
    [ tans , tL ] = get_ID( temporal_prediction, 3 );
    ans = sans + 2*tans ;
    [~,predict] = max(ans);
    accur(1) = accur(1) + ( sL == label(index) ); 
    accur(2) = accur(2) + ( tL == label(index) ); 
    accur(3) = accur(3) + ( predict == label(index) ); 
    fprintf('Spatial  : %.4f || %03d vs %03d \n',accur(1) / index, sL , label(index));
    fprintf('Temporal : %.4f || %03d vs %03d \n',accur(2) / index, tL , label(index));
    fprintf('Fuse     : %.4f || %03d vs %03d \n',accur(3) / index, predict , label(index));
    fprintf('%05d videos predict done in %.3f s \n', index , toc );
end

caffe.reset_all();
