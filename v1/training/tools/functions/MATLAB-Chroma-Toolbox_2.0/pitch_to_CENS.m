function [f_CENS,sideinfo] = pitch_to_CENS(f_pitch,parameter,sideinfo)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: pitch_to_CENS
% Date of Revision: 2011-03
% Programmer: Meinard Mueller, Sebastian Ewert
%
% Description:
% Normalized statistical chroma-based energy distribution feature (CENS).
% The following is computed:
% * energy for each chroma band
% * normalisation of the chroma vectors
% * local statistics:
%   - component-wise quantisation of the normalized chroma vectors
%   - upfirdn filters and downsamples each column of f_stat_help
%   - normalize each vector with its l^2 norm
%
% Remark:
% * parameter.inputFeatureRate specifies the feature rate of f_pitch. The value
% is used to derive the output feature rate given via sideinfo.
%
% Input:
%         f_pitch
%         parameter.quantSteps = [40 20 10 5] / 100;
%         parameter.quantWeights = [ 1 1 1 1]/4;
%         parameter.normThresh = 0.001;
%         parameter.winLenSmooth = 41;
%         parameter.downsampSmooth = 10;
%         parameter.midiMin = 21;
%         parameter.midiMax = 108;
%         parameter.inputFeatureRate = 0;
%         parameter.save = 0;
%         parameter.saveDir = '';
%         parameter.saveFilename = '';
%         parameter.visualize = 0;
%         sideinfo
%
% Output:
%         f_CENS
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

if isfield(parameter,'quantSteps')==0
    parameter.quantSteps = [40 20 10 5] / 100;
end
if isfield(parameter,'quantWeights')==0
    parameter.quantWeights = [ 1 1 1 1]/4;
end
if isfield(parameter,'normThresh')==0
    parameter.normThresh = 0.001;
end
if isfield(parameter,'winLenSmooth')==0
    parameter.winLenSmooth = 41;
end
if isfield(parameter,'downsampSmooth')==0
    parameter.downsampSmooth = 10;
end
if isfield(parameter,'midiMin')==0
    parameter.midiMin = 21;
end
if isfield(parameter,'midiMax')==0
    parameter.midiMax = 108;
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

% calculate energy for each chroma band
f_chroma_energy = zeros(12,seg_num);
for p=parameter.midiMin:parameter.midiMax
    chroma = mod(p,12)+1;
    f_chroma_energy(chroma,:) = f_chroma_energy(chroma,:)+f_pitch(p,:);
end

% normalize the chroma vectors
f_chroma_energy_distr = zeros(12,seg_num);
for k=1:seg_num
    if sum(f_chroma_energy(:,k)>parameter.normThresh)>0
        seg_energy_square = sum(f_chroma_energy(:,k));
        f_chroma_energy_distr(:,k) = ((f_chroma_energy(:,k))/seg_energy_square);
    end
end

% calculate a CENS feature

% component-wise quantisation of the normalized chroma vectors
f_stat_help = zeros(12,seg_num);
for n=1:length(parameter.quantSteps)
    f_stat_help = f_stat_help + (f_chroma_energy_distr>parameter.quantSteps(n))*parameter.quantWeights(n);
end

% Temporal smoothing and downsampling
[f_chroma_energy_stat,CENSfeatureRate] = smoothDownsampleFeature(f_stat_help,parameter);

% last step: normalize each vector with its l^2 norm
f_CENS = normalizeFeature(f_chroma_energy_stat,2, parameter.normThresh);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update sideinfo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sideinfo.CENS.version = 1;
sideinfo.CENS.midiMin = parameter.midiMin;
sideinfo.CENS.midiMax = parameter.midiMax;
sideinfo.CENS.featureRate = CENSfeatureRate;
sideinfo.CENS.quantSteps = parameter.quantSteps;
sideinfo.CENS.quantWeights = parameter.quantWeights;
sideinfo.CENS.normThresh = parameter.normThresh;
sideinfo.CENS.winLenSmooth = parameter.winLenSmooth;
sideinfo.CENS.downsampSmooth = parameter.downsampSmooth;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Saving to file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if parameter.save
    filename = strcat(parameter.saveFilename,'_CENS_',num2str(parameter.winLenSmooth),'_',num2str(parameter.downsampSmooth));
    save(strcat(parameter.saveDir,filename),'f_CENS','sideinfo');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if parameter.visualize
    parameterVis.featureRate = CENSfeatureRate;
    parameterVis.title = 'CENS chromagram';
    visualizeChroma(f_CENS,parameterVis)
end

end
