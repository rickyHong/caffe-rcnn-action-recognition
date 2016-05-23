
% this code is inspired by VOCevaldet in the PASVAL VOC devkit
% Note: this function has been significantly optimized since ILSVRC2013
% Make_Faster_RCNN_Train_Data('../../VOCdevkit/VOC2007/ImageSets/Main/train.txt', '../../VOCdevkit/VOCcode', '../../VOCdevkit/VOC2007/Annotations', '../../examples/FRCNN/dataset/voc2007_train.txt')
% Make_Faster_RCNN_Train_Data('../../VOCdevkit/VOC2007/ImageSets/Main/test.txt', '../../VOCdevkit/VOCcode', '../../VOCdevkit/VOC2007/Annotations', '../../examples/FRCNN/dataset/voc2007_test.txt')
% Make_Faster_RCNN_Train_Data('../../VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', '../../VOCdevkit/VOCcode', '../../VOCdevkit/VOC2007/Annotations', '../../examples/FRCNN/dataset/voc2007_trainval.test')
function Make_Faster_RCNN_Train_Data(ImageSet, devkit, XML_Dir, Save_Name)
% Format : 
% # image_index
% img_path (rel path)
% num_roi
% label x1 y1 x2 y2
assert( ~isempty(devkit) && exist(devkit,'dir') );
assert( ~isempty(XML_Dir) && exist(XML_Dir,'dir') );
addpath( fullfile(pwd, devkit) );
VOCinit;
[ pic ] = textread(ImageSet,'%s');
num_imgs = length(pic);
t = tic;
Fid = fopen(Save_Name,'w');
fprintf('total imgs : %d\n', num_imgs);
for i=1:num_imgs
    rec = VOCreadxml(fullfile(XML_Dir,[pic{i},'.xml']));
    fprintf(Fid,'# %d\n%s\n',i-1, rec.annotation.filename);
    if ~isfield(rec.annotation,'object')
        fprintf(Fid,'0\n');
    else
        %fprintf(Fid,'%-3d\n',length(rec.annotation.object));
        cobj = [];
        for j=1:length(rec.annotation.object)
            obj = rec.annotation.object(j);
            label = Get_Label(VOCopts.classes, obj.name);
            b = obj.bndbox;
            box = str2double({b.xmin b.ymin b.xmax b.ymax});
            %fprintf(Fid, '%-3d %-5.0f %-5.0f %-5.0f %-5.0f\n', label, box );
            if(str2double(getfield(obj,'difficult')) == 0) 
                cobj{end+1} = j;
            end
        end
        fprintf(Fid,'%-3d\n',length(cobj));
        for jj=1:length(cobj)
            j = cobj{jj};
            obj = rec.annotation.object(j);
            label = Get_Label(VOCopts.classes, obj.name);
            b = obj.bndbox;
            box = str2double({b.xmin b.ymin b.xmax b.ymax});
            fprintf(Fid, '%-3d %-5.2f %-5.2f %-5.2f %-5.2f\n', label, box );
        end
    end
end
fprintf('Process %6d xmls in %.2f min\n', num_imgs , toc(t)/60);

end

function label = Get_Label(classes, name)
    for i = 1:length(classes)
        if(strcmp(classes{i},name)==1)
            label = i;
            return;
        end
    end
    assert( false , ['Does not find matched class name : ', name]);
end

