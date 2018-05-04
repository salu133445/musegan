function [seg_output, label] = audio_segmenter_sf( filename,chroma,vis, lab, winLenSTMSP, m, k ,st)
    if(nargin < 6)
           m = 2.5;
           k = 0.04;
           st = 30;
    end
    if(nargin < 5) winLenSTMSP = 4410; end
    if(nargin < 4) lab = 0; end
    if(nargin < 3) vis = 0; end
    if(nargin < 2) chroma = 'cens'; end
    if(nargin < 1)
        error('not enough inputs');
    end
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    disp('Feature & Pre-processing (1/4)');
    feature = feature_generator(filename, chroma, winLenSTMSP, vis);
    [a, fs] = audioread(filename);    % audio
    X = feature;                        
    [~, No] =size(X);                 % 12 x #frames
    s_total = floor(length(a) / fs);  % total length in seconds
    s_frame = No / s_total;           % #frames per seconds
    mf = round(m*s_frame);            % embbed dimension
    tau = 1;                             
    w = (mf - 1) * tau;                
    N = No - w;                       % N'
    sl = 0.3;
    thres = 0.05;
    lamda = 6;
%% Accounting for the past

    Xd = [];
    for i = w+1:No
        xd = [];
        for j = 0:w
            xd = [xd,X(:,i-j)'];
        end
        xd = xd';
        Xd = [Xd, xd];  
    end
%% Recurrence Plot
    disp('Recurrence Plot (2/4)');
    
    R = recurrence_plot(k, Xd);
    
    if(vis)
        figure;
        colormapSet = generateColormapValue();
        colormap(colormapSet.colormap5);
        imagesc(R);
    end
%% Structure Feature
    disp('Structure Feature (3/4)');

    L = sm_to_time_lag(R);
    
%     if(vis)
%         figure;
%         colormapSet = generateColormapValue();
%         colormap(colormapSet.colormap5);
%         imagesc(L);
%     end
    
    tsl = round(sl*s_frame );
    tst = round(st*s_frame );
    variance = 0.4;
    
    P = kernel_density_estimation_2d(L, tst, tsl, variance);
    
    if(vis)
        figure;
        colormapSet = generateColormapValue();
        colormap(colormapSet.colormap5);
        imagesc(P);
    end
%% Novelty Curve & Peak Selection
    disp('Novelty Curve & Peak Selection (4/4)'); 
    
    c = [];
    for i =1:N-1
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
    
%% Post-Processing & Set Boundaries
   
    b  = find(ispeak);
    if(b(1) > (1 + pwin/3)) b = [1,b]; end
    if(b(end) < (N - pwin/3 )) b = [b, N]; 
    else
        b(end) = N;
    end

    w_shift = ceil(1+w/2);
    seg_output= 0.0;
    for i = 1:length(b)
        if(i == 1) continue; end
        seg_output = [seg_output, (b(i) +  w_shift)/s_frame];
    end

%% Labeling

    if(lab)
        disp('labeling...')
        label = segment_labeling( b, R, vis);
    else
        label = [];
    end
    
    disp('done!!')
end

