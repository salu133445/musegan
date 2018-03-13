clc; clear all; close all;
%%

path_data_root = 'structure/';
addpath (genpath('./tools/functions'));
path_lab = [path_data_root, 'labs/'];
path_seg = [path_data_root, 'segs/'];
files = dir([path_seg, '*.lab']);

%%
clc;

for fidx = 1:length(files)
path_seg_file = [path_seg, files(fidx).name]; % 2 140 63
[st, ed, lab] = textread(path_seg_file, '%f %f %s', 'delimiter', '\t');



lab = cell2mat(lab) - 64;
%%

num_thres = 2;
lab_u = unique(lab) ;
lab_u_num = zeros(length(lab_u), 1);
lab_arr = [];
lab_s = zeros(length(lab_u),1);
for i = 1:length(lab_u)
    N = numel(find(lab==lab_u(i)));
    lab_u_num(i) = N;
    if(N >= num_thres)
        lab_s(i) = 3;
    else
        lab_s(i) = 2;
    end
end

[v, p] = max(flip(lab_u_num));
rp = length(lab_u_num) - p + 1;
lab_s(lab_u(rp)) = 4; 

if(lab_u_num(lab(end)) < num_thres)
    lab_s(lab(end) ) = 1;
end
if(lab_u_num(lab(1)) < num_thres)
    lab_s(lab(1) ) = 0;
end
clc
% 0      1      2       3      4  (new_label)
% intro  outro  bridge  verse  chorus
% for i = 1:length(st)
%     fprintf('%f %f %s %d %d\n',st(i), ed(i), lab(i) + 64, lab_u_num(lab(i) ),  lab_s(lab(i) ))
% end

filename_output = [path_lab,files(fidx).name];
fid = fopen(filename_output, 'w');
for i = 1:length(st)
    fprintf(fid,'%f %f %s %d %d\n',st(i), ed(i), lab(i) + 64, lab_u_num(lab(i) ),  lab_s(lab(i) ));
end
% label number new_lab
fclose(fid);

end