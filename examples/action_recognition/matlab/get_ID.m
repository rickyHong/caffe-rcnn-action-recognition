function [ A , ans ] = get_ID(prediction , swit )
    num = size(prediction , 1);
    if( swit == 1 ) 
        A = sum(prediction,2);
    elseif( swit == 2 )
        cur = prediction .* prediction;
        cur = sum(cur);
        cur = repmat(cur , num , 1);
        cur = prediction ./ cur; 
        A = sum(cur,2);
    elseif( swit == 3 )
        prediction = exp( prediction );
        cur = sum(prediction);
        cur = repmat(cur , num , 1);
        cur = prediction ./ cur; 
        A = sum(cur,2);
    else
        disp(' Unkown swit !');
        exit(-1);
    end
    [~,ans] = max(A);
end
