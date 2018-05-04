function [ colormapSet ] = generateColormapValue()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: generateColormapValue.m
% Date of Revision: 2013-06
% Programmer: Nanzhu Jiang, Peter Grosche, Meinard Müller
% http://www.audiolabs-erlangen.de/resources/MIR/SMtoolbox/
%
%
% Description:
%   This function caculates a forward smoothing similarity matrix and a  
%   backward smoothing similrity matrix with tempo enhancement. 
%
%   Here, the tempo-invariance is included by computing different similarity 
%   matrices with diffrent tempo changes. We assume the first feature sequence 
%   having different tempo compared to the second feature sequence. This is 
%   implemented by resampling the first feature sequence into different time 
%   positions. A similarity matrix is then computed by the resampled first 
%   feature sequence and the original second sequence. In the end, we take 
%   the element-wise maximum among all similarity matriices since simulate 
%   different time positions might have different tempo changes.
%
% Input:

%
%
% Output:
%       colormapSet: a set of colormap values.

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Reference:
%   If you use the 'SM toobox' please refer to:
%   [MJG13] Meinard Müller, Nanzhu Jiang, Harald Grohganz
%   SM Toolbox: MATLAB Implementations for Computing and Enhancing Similarity Matrices
%   Proceedings of the 53rd Audio Engineering Society Conference on Semantic Audio, London, 2014.
%
% License:
%     This file is part of 'SM Toolbox'.
%
%     'SM Toolbox' is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 2 of the License, or
%     (at your option) any later version.
%
%     'SM Toolbox' is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with 'SM Toolbox'. If not, see
%     <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure;
colormap1 = flipud([gray(64) ;ones(128,3) ]);
colormap2 = flipud(gray);
colormap3 = flipud([gray(128) ;ones(128,3) ]);

% enhanced with pentalty
delta = -2;
g_max = 0.01;
g_min = 0.005;
h_min = 0.05;
h_max = 1;
nH = h_max-h_min;
nG = g_max-g_min;
range = abs(delta)/1;
relsize = range*nH/nG;
a = flipud(gray(floor(256./(range+1))));
sa = size(a,1);
b = flipud(gray(floor(sa*relsize)));
sb = size(b,1);
colormap4 = [b(round(g_min*sb):round(g_max*sb),:); a(round(h_min*sa):round(h_max*sa),:)];

% exponential
x = 50;
y = (x.^(0:0.01:1)'-1)./(x-1);
colormap5 = 1-([y,y,y]);

%    colormap6 = bsxfun(@times,gray(64),10.^((0:63))');
%    colormap7 = parameter.colormap./max(parameter.colormap(:));

close;

colormapSet = [];
colormapSet.colormap1 = colormap1;
colormapSet.colormap2 = colormap2;
colormapSet.colormap3 = colormap3;
colormapSet.colormap4 = colormap4;
colormapSet.colormap5 = colormap5;




end
