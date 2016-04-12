%%% Example 
%%% model_def_file and model_def_file only the first filter size is different
%%% 
%{

model_def_file = './Developy/ResNet-50-rgb-Test.prototxt';
model_file = '../../../models/action_recognition/ResNet-50-rgb-model.caffemodel';
Final_def_file = './Developy/ResNet-50-flow-Test-without-conv1.prototxt';
saved_model = '../../../models/action_recognition/ResNet-50-flow-model.caffemodel';
gpu_id = 7;
AveChannelToN(model_def_file, model_file, Final_def_file, saved_model, gpu_id);

%}

function Ans = AveChannelToN(model_def_file, model_file, Final_def_file, saved_model, gpu_id)
active_caffe_mex();
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
p_net = caffe.Net(model_def_file, model_file, 'test');
p_name = p_net.layer_names;
a_net = caffe.Net(Final_def_file, model_file, 'test');
a_name = a_net.layer_names;
%%CHECK NAME 
assert(length( a_name ) == length( p_name ));
for index = 1:length( a_name )
    if(strcmp(p_name{index},a_name{index})~=1)
        fprintf('FALT Model Prototxt %4dth Layer does not match\n', index);
    end
end



for index = 1:length( a_name )

    name  = a_name{index};    
    layer = p_net.layer_vec(index);
    if( size(layer.params,1) == 0 ),continue;end

    Aft_L = a_net.layer_vec(index); 
    %% Check Size of parameters
    assert( all( size(Aft_L.params) == size(layer.params) ) );

    if( strcmp(name,'conv1') == 1 ) % Conv1 Filter
        disp('=====First Conv Layer=====');
        assert( length(layer.params) == 2 );
        for idx = 1 : length(layer.params)
            StringG(['--Pre ',name], idx , layer.params(idx).shape);
            StringG(['--Aft ',name], idx , Aft_L.params(idx).shape);
            weights = layer.params(idx).get_data();
            next = single(zeros( Aft_L.params(idx).shape ));
            if( idx == 2 )
                a_net.layer_vec(index).params(idx).set_data( weights );
            else
                fprintf('Special Shape [1,2,4] : (%4d,%4d,%4d)\n',size(weights,1),size(weights,2),size(weights,4));
                for i = 1 : size(weights,1)
                    for j = 1 : size(weights,2)
                        for k = 1 : size(weights,4)
                            next(i,j,:,k) = single(zeros(size(next,3),1)) + mean(weights(i,j,:,k));
                        end
                    end
                end
                a_net.layer_vec(index).params(idx).set_data( next );
            end
        end
        disp('First Conv Layer Convert Done ******************');
        continue;
        assert( false );
    end

    disp('=========================');
    for idx = 1 : length(layer.params)
        %% Check Size of parameters
        assert( all( layer.params(idx).shape == Aft_L.params(idx).shape ) );
        StringG(['Pre ',name], idx , layer.params(idx).shape);
        StringG(['Aft ',name], idx , Aft_L.params(idx).shape);
        weights = layer.params(idx).get_data();
        a_net.layer_vec(index).params(idx).set_data( weights );
    end

end
%%% Parameters Done 
%%% Snapshot
a_net.save(saved_model);
fprintf('Saved as %s\n', saved_model);
end

function StringG( name, idx, sz )
    if( numel(sz) == 4 )
        fprintf('Layer %s  :(%d) Shape(%4d,%4d,%4d,%4d)\n',name,idx,sz);
    elseif( numel(sz) == 1 )
        fprintf('Layer %s  :(%d) Shape(%4d)\n',name,idx,sz);
    elseif( numel(sz) == 2 )
        fprintf('Layer %s  :(%d) Shape(%4d,%4d)\n',name,idx,sz);
    elseif( numel(sz) == 3 )
        fprintf('Layer %s  :(%d) Shape(%4d,%4d,%4d)\n',name,idx,sz);
    else
        fprintf('FUCK-------');
    end
end
