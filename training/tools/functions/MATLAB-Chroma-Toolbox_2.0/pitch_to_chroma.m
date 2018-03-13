function [f_chroma_norm,sideinfo] = pitch_to_chroma(f_pitch,parameter,sideinfo)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: pitch_to_chroma 
% Date of Revision: 2011-03
% Programmer: Meinard Mueller, Sebastian Ewert
%
% Description:
% Computes normalized chroma vectors from pitch features
%
% Input:  
%         f_pitch
%         parameter.applyLogCompr = 0;
%         parameter.factorLogCompr = 100;
%         parameter.addTermLogCompr = 1;
%         parameter.winLenSmooth = 1;
%         parameter.downsampSmooth = 1;
%         parameter.applyNormalization = 1;
%         parameter.normP = 2;
%         parameter.normThresh = 0.001;
%         parameter.midiMin = 1;
%         parameter.midiMax = 120;
%         parameter.inputFeatureRate = 0;
%         parameter.save = 0;
%            parameter.save_dir = '';
%            parameter.save_filename = '';
%         parameter.visualize = 0;
%         sideinfo
%
% Output: 
%         f_chroma_norm
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

if isfield(parameter,'applyLogCompr')==0
    parameter.applyLogCompr = 0;
end
if isfield(parameter,'factorLogCompr')==0
    parameter.factorLogCompr = 100;
end
if isfield(parameter,'addTermLogCompr')==0
    parameter.addTermLogCompr = 1;
end
if isfield(parameter,'winLenSmooth')==0
    parameter.winLenSmooth = 1;
end
if isfield(parameter,'downsampSmooth')==0
    parameter.downsampSmooth = 1;
end
if isfield(parameter,'applyNormalization')==0
    parameter.applyNormalization = 1;
end
if isfield(parameter,'normP')==0
    parameter.normP = 2;
end
if isfield(parameter,'normThresh')==0
    parameter.normThresh = 0.001;
end
if isfield(parameter,'midiMin')==0
    parameter.midiMin = 1;
end
if isfield(parameter,'midiMax')==0
    parameter.midiMax = 120;
end
if isfield(parameter,'inputFeatureRate')==0
    parameter.inputFeatureRate = 0;
end
if isfield(parameter,'save')==0
    parameter.save = 0;
end
if isfield(parameter,'save_dir')==0
    parameter.save_dir = '';
end
if isfield(parameter,'save_filename')==0
    parameter.save_filename = '';
end
if isfield(parameter,'visualize')==0
    parameter.visualize = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

seg_num = size(f_pitch,2);

if parameter.applyLogCompr
    f_pitch = log10(parameter.addTermLogCompr+f_pitch*parameter.factorLogCompr);
end

% calculate energy for each chroma band
f_chroma =  zeros(12,seg_num);
for p=parameter.midiMin:parameter.midiMax
    chroma = mod(p,12)+1;
    f_chroma(chroma,:) = f_chroma(chroma,:)+f_pitch(p,:);
end

% Temporal smoothing and downsampling
[f_chroma,chromaFeatureRate] = smoothDownsampleFeature(f_chroma,parameter);

if parameter.applyNormalization
    % normalise the chroma vectors according the norm l^p
    f_chroma_norm = normalizeFeature(f_chroma,parameter.normP, parameter.normThresh);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update sideinfo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sideinfo.chroma.applyLogCompr = parameter.applyLogCompr;
sideinfo.chroma.factorLogCompr = parameter.factorLogCompr;
sideinfo.chroma.addTermLogCompr = parameter.addTermLogCompr;
sideinfo.chroma.winLenSmooth = parameter.winLenSmooth;
sideinfo.chroma.downsampSmooth = parameter.downsampSmooth;
sideinfo.chroma.applyNormalization = parameter.applyNormalization;
sideinfo.chroma.normP = parameter.normP;
sideinfo.chroma.normThresh = parameter.normThresh;
sideinfo.chroma.midiMin = parameter.midiMin;
sideinfo.chroma.midiMax = parameter.midiMax;
sideinfo.chroma.chromaFeatureRate = chromaFeatureRate;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Saving to file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if parameter.save == 1   
    filename = strcat(parameter.save_filename,'_chroma');
    save(strcat(parameter.save_dir,filename),'f_chroma_norm','f_chroma','sideinfo');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if parameter.visualize
    titleString = 'Chromagram';
    imagerange = [0 1];
    if parameter.applyLogCompr
        titleString = ['Log ', titleString];
    end
    if parameter.applyNormalization
        imagerange = 0;
        titleString = ['Normalized ', titleString];
    end
    parameterVis.imagerange = imagerange;
    parameterVis.featureRate = chromaFeatureRate;
    parameterVis.title = titleString;
    visualizeChroma(f_chroma_norm,parameterVis)
end

end