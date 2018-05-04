function [f_CRP,sideinfo] = pitch_to_CRP(f_pitch,parameter,sideinfo)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: pitch_to_CRP
% Date of Revision: 2011-03
% Programmer: Meinard Mueller, Sebastian Ewert
%
% Description:
% - Calculates CRP (Chroma DCT-reduced Log Pitch) Features
% (see "Towards Timbre-Invariant Audio Features for Harmony-Based Music" by
%  Meinard Mueller and Sebastian Ewert)
%
% Remark:
% - parameter.inputFeatureRate specifies the input feature rate. This value is
% used to derive the output feature rate.
%
% Input:
%         f_pitch
%         parameter.coeffsToKeep = [55:120];
%         parameter.applyLogCompr = 1;
%         parameter.factorLogCompr = 1000;
%         parameter.addTermLogCompr = 1;
%         parameter.normP = 2;
%         parameter.winLenSmooth = 1;
%         parameter.downsampSmooth = 1;
%         parameter.normThresh = 10^-6;
%         parameter.inputFeatureRate = 0;
%         parameter.save = 0;
%           parameter.saveDir = '';
%           parameter.saveFilename = '';
%         parameter.visualize = 0;
%         sideinfo
%
% Output:
%         f_CRP
%         sideinfo
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

if nargin<3
    sideinfo=[];
end
if nargin<2
    parameter=[];
end
if nargin<1
    error('Please specify input data f_pitch');
end

if isfield(parameter,'coeffsToKeep')==0
    parameter.coeffsToKeep = [55:120];
end
if isfield(parameter,'applyLogCompr')==0
    parameter.applyLogCompr = 1;
end
if isfield(parameter,'factorLogCompr')==0
    parameter.factorLogCompr = 1000;
end
if isfield(parameter,'addTermLogCompr')==0
    parameter.addTermLogCompr = 1;
end
if isfield(parameter,'normP')==0
    parameter.normP = 2;
end
if isfield(parameter,'winLenSmooth')==0
    parameter.winLenSmooth = 1;
end
if isfield(parameter,'downsampSmooth')==0
    parameter.downsampSmooth = 1;
end
if isfield(parameter,'normThresh')==0
    parameter.normThresh = 10^-6;
end
if isfield(parameter,'inputFeatureRate')==0
    parameter.inputFeatureRate = 0;
end
if isfield(parameter,'save')==0
    parameter.save = 0;
end
if isfield(parameter,'saveDir')==0
    parameter.saveDir = '';
end
if isfield(parameter,'saveFilename')==0
    parameter.saveFilename = '';
end
if isfield(parameter,'visualize')==0
    parameter.visualize = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

seg_num = size(f_pitch,2);

% log compression
if parameter.applyLogCompr
    f_pitch_log = log10(parameter.addTermLogCompr+f_pitch*parameter.factorLogCompr);
else
    f_pitch_log = f_pitch;
end

% DCT based reduction
DCT = internal_DCT(size(f_pitch_log,1));
DCTcut = DCT;
DCTcut(setdiff([1:120],parameter.coeffsToKeep),:) = 0;
DCT_filter = DCT'*DCTcut;
f_pitch_log_DCT = DCT_filter*f_pitch_log;

% calculate energy for each chroma band
f_CRP = zeros(12,seg_num);
for p=1:120
    chroma = mod(p,12)+1;
    f_CRP(chroma,:) = f_CRP(chroma,:)+f_pitch_log_DCT(p,:);
end

% normalize the vectors according to the norm l^p
f_CRP = normalizeFeature(f_CRP,parameter.normP, parameter.normThresh);

if (parameter.winLenSmooth ~= 1) || (parameter.downsampSmooth ~= 1)
    % Temporal smoothing and downsampling
    [f_CRP,CrpFeatureRate] = smoothDownsampleFeature(f_CRP,parameter);
    
    % re-normalize the vectors according to the norm l^p
    f_CRP = normalizeFeature(f_CRP,parameter.normP, parameter.normThresh);
else
    CrpFeatureRate = parameter.inputFeatureRate;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update sideinfo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sideinfo.CRP.coeffsToKeep = parameter.coeffsToKeep;
sideinfo.CRP.applyLogCompr = parameter.applyLogCompr;
sideinfo.CRP.factorLogCompr = parameter.factorLogCompr;
sideinfo.CRP.addTermLogCompr = parameter.addTermLogCompr;
sideinfo.CRP.normP = parameter.normP;
sideinfo.CRP.winLenSmooth = parameter.winLenSmooth;
sideinfo.CRP.downsampSmooth = parameter.downsampSmooth;
sideinfo.CRP.normThresh = parameter.normThresh;
sideinfo.CRP.featureRate = CrpFeatureRate;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Saving to file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if parameter.save
    filename = strcat(parameter.saveFilename,'_CRP_',num2str(parameter.winLenSmooth),'_',num2str(parameter.downsampSmooth));
    save(strcat(parameter.saveDir,filename),'f_CRP','sideinfo');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if parameter.visualize
    parameterVis.title = 'CRP chromagram';
    parameterVis.featureRate = CrpFeatureRate;
    visualizeCRP(f_CRP,parameterVis);
end

end

function matrix = internal_DCT(l)

matrix = zeros(l,l);

for m = 0:l-1
    for n = 0:l-1
        matrix(m+1,n+1) = sqrt(2/l)*cos((m*(n+0.5)*pi)/l);
    end
end

matrix(1,:) = matrix(1,:)/sqrt(2);

end
