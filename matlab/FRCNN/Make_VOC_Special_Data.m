% this code is inspired by VOCevaldet in the PASVAL VOC devkit
% Note: this function has been significantly optimized since ILSVRC2013
% Make_Faster_RCNN_Train_Data('../../VOCdevkit/VOC2007/ImageSets/Main/test.txt', '../../VOCdevkit/VOCcode', '../../VOCdevkit/VOC2007/Annotations', '../../examples/FRCNN/dataset/voc2007_test','difficult')
function Make_VOC_Special_Data(ImageSet, devkit, XML_Dir, Save_Name, attr)
% Format : 
% # image_index
% img_path (rel path)
% num_roi
% attribute 
assert( ~isempty(devkit) && exist(devkit,'dir') );
assert( ~isempty(XML_Dir) && exist(XML_Dir,'dir') );
addpath( fullfile(pwd, devkit) );
VOCinit;
[ pic ] = textread(ImageSet,'%s');
num_imgs = length(pic);
Save_Name = [Save_Name, '.', attr];
fprintf('Save to %s\n',Save_Name);
t = tic;
Fid = fopen(Save_Name,'w');
total_attr = 0;
for i=1:num_imgs
    rec = VOCreadxml(fullfile(XML_Dir,[pic{i},'.xml']));
    fprintf(Fid,'# %d\n%s\n',i-1, rec.annotation.filename);
    if ~isfield(rec.annotation,'object')
        fprintf(Fid,'0\n');
    else
        fprintf(Fid,'%-3d\n',length(rec.annotation.object));
        for j=1:length(rec.annotation.object)
            obj = rec.annotation.object(j);
            if ~isfield(obj, attr)
                fprintf(Fid, '0\n');
            else
                value = getfield(obj,attr);
                fprintf(Fid, '%s\n', value );
                total_attr = total_attr + 1;
            end
        end
    end
end
fprintf('Process %6d xmls in %.2f min\n', num_imgs , toc(t)/60);
fprintf('Extract %6d atrribute value\n',total_attr);
end
