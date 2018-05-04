function [shiftFB,centerA4,tuningSemitones,sideinfo] = estimateTuning(f_input,parameter,sideinfo)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: estimateTuning
% Date of Revision: 2011-03
% Programmer: Sebastian Ewert
%
% Description:
% - input is either a mono audio signal or a real valued spectrogram (magnitude or power)
% - if input is a spectrogram you have to set parameter.timeinfo and
% parameter.freqinfo (vectors defining the time and freq centers for each spectrogram bin)
% - if input is an audio signal, a sampling freq of 22050 Hz is assumed
% - guesses the tuning according to a simple energy maximizing criterion
% - output is either: what shiftFB is best to use (shiftFB \in [0:5]).
%   Alternatively, the center freq for A4 is given which can be used to
%   specify a filterbank on your own. The second option is more fine
%   grained.
%   Alternatively, it gives a tuning in semitones, which can
%   easily be shifted cyclicly. For example: a tuning of -19/20 is more likely to be
%   +1/20 Tuning difference.
% - parameter.numAdditionalTunings: how many tunings besides the fixed shiftFB ones
% to test. For example: If set to 3, than three additional tuning settings are
% tested for, located at 1/4, 2/4 and 3/4 semitones below the reference
% tuning. If set to 5, then at 1/6, 2/6,..., 5/6 semitones.
% - parameter.pitchRange specifies which pitches are considered for the
% tuning estimation. 
% - parameter.pitchWeights: each pitch is considered according to a weight
% - Middle pitches are considered as being more important per default because
% here the frequency resolution is high enough. Additionally the piano has
% a consistent tuning only for middle pitches.
%
% Input:
%        f_input
%        parameter.numAdditionalTunings = 0;
%        parameter.pitchRange = [21:108];
%        parameter.pitchWeights = gausswin(length(parameter.pitchRange)).^2;  
%        parameter.fftWindowLength = 8192;
%        parameter.windowFunction = @hanning;
%        sideinfo
%
% Output:
%        shiftFB
%        centerA4
%        tuningSemitones
%        sideinfo
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

if isfield(parameter,'numAdditionalTunings')==0
    parameter.numAdditionalTunings = 0;
end
if isfield(parameter,'pitchRange')==0
    % Which pitches to consider during the estimation
    parameter.pitchRange = [21:108];
end
if isfield(parameter,'pitchWeights')==0
    % assign a weight to each pitch specified in parameter.pitchRange to
    % specify it's importance
    parameter.pitchWeights = gausswin(length(parameter.pitchRange)).^2;  
end

% the following parameters are only for audio signal input
if isfield(parameter,'fftWindowLength')==0
    parameter.fftWindowLength = 8192;
end
if isfield(parameter,'windowFunction')==0
    parameter.windowFunction = @hanning;  % only tested with hanning. 
end

if min(size(f_input)) == 1
    inputIsAudioSignal = 1;
else
    inputIsAudioSignal = 0;
end

if ~inputIsAudioSignal
    if isfield(parameter,'freqinfo')==0
        error('When using a spectrogram input you have to set parameter.freqinfo');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numTunings = 6 + parameter.numAdditionalTunings;
referenceFreqsA4 = zeros(numTunings,1);
tunings = zeros(numTunings,1);

tunings(1) = 0;
tunings(2) = -1/4;
tunings(3) = -1/3;
tunings(4) = -1/2;
tunings(5) = -2/3;
tunings(6) = -3/4;
for k=1:parameter.numAdditionalTunings
    tunings(k+6) = -k/(parameter.numAdditionalTunings+1);
end
referenceFreqsA4 = 2.^((69-69+tunings)/12) * 440;

if inputIsAudioSignal
    [s,f,t] = spectrogram(f_input, parameter.windowFunction(parameter.fftWindowLength), parameter.fftWindowLength/2, parameter.fftWindowLength, 22050);
else
    s = f_input;
    f = parameter.freqinfo;
end
s = abs(s);

directFreqBinSearch = 0;
if all( (f(2:end)-f(1:end-1)) - (f(2)-f(1)) < eps )
    directFreqBinSearch = 1;
end

averagedPowerSpectrogram = sum(s.^2,2);
totalPitchEnergyViaSpec = zeros(numTunings,1);
for tu=1:numTunings
    centerfreqs = 2.^((parameter.pitchRange-69)/12) * referenceFreqsA4(tu);
    upperborderfreqs = 2.^((parameter.pitchRange-68.5)/12) * referenceFreqsA4(tu);
    lowerborderfreqs = 2.^((parameter.pitchRange-69.5)/12) * referenceFreqsA4(tu);
    
    % build triangular filterbank for magnitude spectrogram
    spectrogramFilter = zeros(length(f),1);
    for k=1:length(parameter.pitchRange)        
        c = getCorrespondingBin(f,centerfreqs(k),directFreqBinSearch);
        u = getCorrespondingBin(f,upperborderfreqs(k),directFreqBinSearch);
        l = getCorrespondingBin(f,lowerborderfreqs(k),directFreqBinSearch);
        
        % order is important here. If third parameter is < 2, then linspace
        % returns the second parameter
        spectrogramFilter(c:u) = parameter.pitchWeights(k) * linspace(1,0,u-c+1);
        spectrogramFilter(l:c) = parameter.pitchWeights(k) * linspace(0,1,c-l+1);
    end

    totalPitchEnergyViaSpec(tu) = sum(spectrogramFilter.^2 .* averagedPowerSpectrogram);    
end

[ignoreMe, maxIndex] = max(totalPitchEnergyViaSpec(1:6));
shiftFB = maxIndex-1;

[ignoreMe, maxIndex] = max(totalPitchEnergyViaSpec);
centerA4 = referenceFreqsA4(maxIndex);
tuningSemitones = tunings(maxIndex);

sideinfo.tuning.shiftFB = shiftFB;
sideinfo.tuning.centerA4 = centerA4;
sideinfo.tuning.tuningSemitones = tuningSemitones;
sideinfo.tuning.method = 'estimateTuningV1';
sideinfo.tuning.numAdditionalTunings = parameter.numAdditionalTunings;
sideinfo.tuning.pitchRange = parameter.pitchRange;
sideinfo.tuning.pitchWeights = parameter.pitchWeights;
sideinfo.tuning.fftWindowLength = parameter.fftWindowLength;
sideinfo.tuning.windowFunction = parameter.windowFunction;
sideinfo.tuning.inputWasAudioSignal = inputIsAudioSignal;

end

function index = getCorrespondingBin(x,sval,directSearch)
% - Finds the entry in x with the smallest absolute distance to sval.
% - x is assumed to be sorted (ascending)
% - 'directSearch' means that all values in x are equally spaced
% - x is assumed to be at least of length 2.
% - If directSearch==0 then we use binary seach to find the entry
%
% You can test the correctness of this procedure by comparing it against
% the result of:  [ignoreMe index] = min(abs(x-sval))
%
% Author: Sebastian Ewert

if sval >= x(end)
    index = length(x);
    return;
elseif sval <= x(1)
    index = 1;
    return;
end


if directSearch
    index = round( (sval-x(1)) / (x(2)-x(1))) + 1;
else
    from=1;
    to=length(x);
    
    while from<=to
        mid = round((from + to)/2);
        diff = x(mid)-sval;
        if diff<0   % x(mid) < sval
            from=mid;
        else              % x(mid) => sval
            to=mid;
        end
        
        if to-from==1
            break;
        end
    end
    if abs(x(from)-sval) < abs(x(to)-sval)
        index = from;
    else
        index = to;
    end
end

end

