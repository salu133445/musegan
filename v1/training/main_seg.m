clc; clear all; close all;
%%

addpath (genpath('./tools/functions'));

path_data_root = 'structure/';
path_chroma ='tracks/Piano_Roll/';
path_R = 'tracks/R/';
path_images = [path_data_root, 'images/'];
path_segs = [path_data_root, 'segs/'];
filesL = dir([path_chroma, '*.mat']);

%%
for fidx = 1500:length(filesL)
disp(fidx)
disp(filesL(fidx).name)
image_dir = [path_images, filesL(fidx).name];
mkdir(image_dir)
filenmae = [filesL(fidx).name];
load([path_chroma,filenmae]);

C = sum(piano_roll(1:12:128, :));
C_ = sum(piano_roll(2:12:128, :));
D = sum(piano_roll(3:12:128, :));
D_ = sum(piano_roll(4:12:128, :));
E = sum(piano_roll(5:12:128, :));
F = sum(piano_roll(6:12:128, :));
F_ = sum(piano_roll(7:12:128, :));
G = sum(piano_roll(8:12:128, :));
G_ = sum(piano_roll(9:12:128, :));
A = sum(piano_roll(10:12:128, :));
A_ = sum(piano_roll(11:12:128, :));
B = sum(piano_roll(12:12:128, :));
%%

% win_size = 20;
% hop_size = 10;
% 
% re_chroma = [];
% chroma = chroma / max(max(chroma)) - min(min(chroma)); 
% for i = 1:hop_size:length(chroma) -  win_size 
%     tmp = sum(chroma(:, i:i+win_size-1)')';
%     re_chroma = [re_chroma, tmp];
% end
% re_chroma = re_chroma / max(max(re_chroma)) - min(min(re_chroma));
% visualizeChroma(chroma);
chroma_hi = [C;C_;D;D_;E;F;F_;G;G_;A;A_;B];
chroma = [];
win_size = 6;
for i = 1:win_size:length(chroma_hi)
    tmp = sum(chroma_hi(:, i:i+5), 2);
    chroma = [chroma, tmp];
end
chroma = chroma / max(max(chroma)) - min(min(chroma));
ic = figure;
visualizeChroma(chroma);
saveas(gcf, [image_dir, '/chroma.png'])
close all
%%
vis=1;
m = 2.5;
k = 0.04;
st = 30;

 X = chroma  ;                      
    [~, No] =size(X);                 % 12 x #frames
    s_total = length(chroma) /10;  % total length in seconds
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
    
   save( [path_R,num2str(fidx), '.mat'],'R')
   %% 
    if(vis)
        figure;
        colormapSet = generateColormapValue();
        colormap(colormapSet.colormap5);
        imagesc(R);
        saveas(gcf,[image_dir, '/R.png'])
        
        close all;
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
    
%     if(vis)
%         figure;
%         colormapSet = generateColormapValue();
%         colormap(colormapSet.colormap5);
%         imagesc(P);
%     end
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
    saveas(gcf,[image_dir, '/boundary.png'])
    close all
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
    seg_output = round(seg_output * 60 / 96);
    
    % adjust the end
    %%
    nb = length(B) / 96;
    if(seg_output(end) > nb - 4)
        seg_output(end) = nb; 
    else
        seg_output = [seg_output, nb];
    end
%% Labeling
 lab = 1;
    if(lab)
        disp('labeling...')
        label = segment_labeling( b, R, 0);
    else
        label = [];
    end
    
    disp('done!!')
    
%%

final_b = (b +  w_shift) * win_size;
filename_output = [path_segs, filenmae(1:end-4), '.lab'];
write_results(filename_output,  seg_output, label);
end