function [ output_args ] = visualize_results(filenames, result, path_ann)

    if(nargin < 3)
        ann = 0;
    else
        ann = 1;
    end
    [a, fs] = audioread(filenames);
    s_total = round(length(a) / fs);
    
%     [s_es, e_es, ~] = textread(path_es, '%f %f %s', 'delimiter', '\t');
    if(ann) [~, e_an, ~] = textread( path_ann, '%f %f %s', 'delimiter', '\t'); end
    seg_es = zeros(1, s_total);
    
    if(ann) seg_an = zeros(1, s_total); end
    for i =1:length(result)
        temp = floor(result(i));
        if(temp == 0) temp = 1; end
        seg_es(temp) = 1;
    end
    if(ann)
        for i =1:length(e_an)
            temp = round(e_an(i));
            if(temp == 0) temp = 1; end
            seg_an(temp) = 1;
        end
    end
    
    
    
    n = 1:1:s_total;
    figure;
    subplot(2,1,1)
    if(ann) 
        if(length(seg_an) > s_total)
            n = 1:1:length(seg_an);
            seg_es(length(seg_an)) = 0;
        end
        stem(n, seg_an, 'r','Marker', 'none'); 
    end
    title('annotation');
    xlabel('sec')
    
    
    subplot(2,1,2)
%     if(ann) stem(n, seg_an, 'r','Marker', 'none'); end
    hold on
    stem(n, seg_es, 'b','Marker', 'none')
    title('estimation');
    xlabel('sec')
   
end

