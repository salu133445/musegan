function [b, c] = RtoPeak(R, vis, s_frame, sl, st, lamda, thres, variance)
%
% disp('Structure Feature (3/4)');

L = sm_to_time_lag(R);
tsl = round(sl*s_frame );
tst = round(st*s_frame );

if(vis)
    figure;
    colormapSet = generateColormapValue();
    colormap(colormapSet.colormap5); 
    imagesc(L);
end
P = kernel_density_estimation_2d(L, tst, tsl, variance);
N = length(P);
if(vis)
        figure;
        colormapSet = generateColormapValue();
        colormap(colormapSet.colormap5);
        imagesc(P);
end
%% Novelty Curve & Peak Selection
%     disp('Novelty Curve & Peak Selection (4/4)'); 
    
    c = [];
    for i =1:length(P)-1
        temp = norm(P(i+1,:) - P(i,:)) ^2;
        c = [c, temp];
    end
    
    % normalization
    c = (c - min(c(:)) + realmin) ./ (max(c(:))-min(c(:)));

    if(vis)
        figure;
        plot(c);
    end
%% Select Peak

    ispeak = ones(1,length(c));
    pwin = round(lamda * s_frame);
    for i = 1:length(c)
        if (c(i) < thres)
            ispeak(i) = 0;
            continue;
        end
        temp = [];
        for j = -pwin:pwin
            if((i + j) > 0 && (i+j) <= length(c) && (j ~= 0))  
                temp = [temp, c(i+j)];
            end
        end
        if(max(temp) > c(i))
            ispeak(i) = 0;
        end
    end
    
    if(vis)
        hold on;
        plot(ispeak, 'g');
    end
    
    b  = find(ispeak);
%     if(b(1) > (1 + pwin/3)) b = [1,b]; end
    if(b(end) < (N - pwin/3 )) b = [b, N]; 
    else
        b(end) = N;
    end

end

