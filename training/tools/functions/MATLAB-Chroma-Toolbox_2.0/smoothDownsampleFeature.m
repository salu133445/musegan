function [f_feature_stat,newFeatureRate] = smoothDownsampleFeature(f_feature,parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: smoothDownsampleFeature
% Date of Revision: 2011-03
% Programmer: Meinard Mueller, Sebastian Ewert
%
% Description:
% - Temporal smoothing and downsampling of a feature sequence
%
% Remark:
% - parameter.featureRate specifies the input feature rate. This value is
% used to derive the output feature rate.
%
% Input:
%         f_feature
%         parameter.winLenSmooth = 1;
%         parameter.downsampSmooth = 1;
%         parameter.inputFeatureRate = 0;
%
% Output:
%         f_feature
%         newFeatureRate
%
% License:
%     This file is part of 'Chroma Toolbox'.
% 
%     'Chroma Toolbox' is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 2 of the License, or
%     (at your option) any later version.
% 
%     'Chroma Toolbox' is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with 'Chroma Toolbox'. If not, see <http://www.gnu.org/licenses/>.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<2
    parameter=[];
end
if nargin<1
    error('Please specify input data');
end
if isfield(parameter,'winLenSmooth')==0
    parameter.winLenSmooth = 1;
end
if isfield(parameter,'downsampSmooth')==0
    parameter.downsampSmooth = 1;
end
if isfield(parameter,'inputFeatureRate')==0
    parameter.inputFeatureRate = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Temporal Smoothing
if (parameter.winLenSmooth ~= 1) || (parameter.downsampSmooth ~= 1)
    winLenSmooth = parameter.winLenSmooth;
    downsampSmooth = parameter.downsampSmooth;
    stat_window = hanning(winLenSmooth);
    stat_window = stat_window/sum(stat_window);
    
    % upfirdn filters and downsamples each column of f_stat_help
    f_feature_stat = zeros(size(f_feature));
    f_feature_stat = (upfirdn(f_feature',stat_window,1,downsampSmooth))';
    seg_num = size(f_feature,2);
    stat_num = ceil(seg_num/downsampSmooth);
    cut = floor((winLenSmooth-1)/(2*downsampSmooth));
    f_feature_stat = f_feature_stat(:,(1+cut:stat_num+cut));            %adjust group delay
else
    f_feature_stat = f_feature;
end

newFeatureRate = parameter.inputFeatureRate / parameter.downsampSmooth;

end

