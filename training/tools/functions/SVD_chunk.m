function chunk_index = SVD_chunk( path_audio, path_svd,threshold_trim, win_size, thres_density, win2_size, vis )
    % 0.65, 13, 5, 150, 1
    [a, fs] = audioread(path_audio);
    load(path_svd);
    svd = tag188(:, 20);
    svd_trim = svd;
    svd_trim(svd <= threshold_trim ) = 0;
    % Grouping
    svd_group = zeros(size(svd));
    for i = 1:length(svd_trim) - win_size
            tmp = svd_trim(i : i + win_size);
            if(sum(tmp > 0) > thres_density)
               svd_group(i:i + win_size)  = 1;
            end
    end
    for i = 1:length(svd_group) - win2_size -1
        if(svd_group(i) && svd_group(i + win2_size))
           svd_group(i:i + win2_size) = 1; 
        end
    end

    tmp = diff(svd_group);
    chunk_index = find(tmp ~= 0) + 1;
    chunk_index = chunk_index * ((length(a) / fs) / length(svd));
    
    if(vis)
        figure;
        subplot(3,1,1)
        stem(svd,'r', 'marker','none' )
        title('JY SVD')
        subplot(3,1,2)
        stem(svd_trim,'r', 'marker','none' )
        title('trim')
        subplot(3,1,3)
        stem(svd_group,'r', 'marker','none' )
        title('group')
    end
end

