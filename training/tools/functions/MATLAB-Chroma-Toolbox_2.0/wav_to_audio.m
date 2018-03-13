function [f_audio,sideinfo] = wav_to_audio(dirAbs,dirRel,wavfilename,parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: wav_to_audio
% Date of Revision: 2011-03
% Programmer: Meinard Mueller, Sebastian Ewert
%
% Description:
% Loads a Wav file and fills a sideinfo variable according to AGtoolbox
% specifications. Resampling and single channel conversion is default, but
% optional.
%
% Input:
%        dirAbs
%        dirRel
%        wavfilename
%        parameter.useResampling = 1;
%        parameter.destSamplerate = 22050;
%        parameter.convertToMono = 1;
%        parameter.monoConvertMode = 'downmix';
%        parameter.message = 0;
%        parameter.vis = 0;
%        parameter.save = 0;
%        parameter.saveDir = [dirAbs,dirRel];
%        parameter.saveFilename = wavfilename;
%
% Output:
%        f_audio
%        sideinfo.wav.version
%        sideinfo.wav.filename
%        sideinfo.wav.dirRel
%        sideinfo.wav.size
%        sideinfo.wav.duration
%        sideinfo.wav.energy
%        sideinfo.wav.fs
%        sideinfo.wav.nbits
%        sideinfo.wav.channels
%        sideinfo.wav.resampled
%        sideinfo.wav.monoConverted
%        sideinfo.wav.monoConvertMode
%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<4
    parameter=[];
end
if nargin<3
    error('Please specify at least the path and filename of the wav file')
end

if isfield(parameter,'useResampling')==0
    parameter.useResampling = 1;
end
if isfield(parameter,'destSamplerate')==0
    parameter.destSamplerate = 22050;
end
if isfield(parameter,'convertToMono')==0
    parameter.convertToMono = 1;
end
if isfield(parameter,'monoConvertMode')==0
    parameter.monoConvertMode = 'downmix';
end
if isfield(parameter,'message')==0
    parameter.message = 0;
end
if isfield(parameter,'vis')==0
    parameter.vis = 0;
end
if isfield(parameter,'save')==0
    parameter.save = 0;
end
if isfield(parameter,'saveDir')==0
    parameter.saveDir = [dirAbs,dirRel];
end
if isfield(parameter,'saveFilename')==0
    parameter.saveFilename = wavfilename;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if parameter.message == 1
    fprintf('wav_to_audio: processing %s, ',wavfilename);
end

[pathstr,name,ext] = fileparts(wavfilename);
[f_audio,fs] = audioread(strcat(dirAbs,dirRel,wavfilename));
    

bConverted_to_mono = 0;
if parameter.convertToMono
    if size(f_audio,2)>1
        bConverted_to_mono = 1;
        if parameter.message == 1
            fprintf('converting to mono, ');
        end
        switch parameter.monoConvertMode
            case 'leftmost_channel'
                f_audio= f_audio(:,1);
            case 'rightmost_channel'
                f_audio= f_audio(:,size(f_audio,2));
            case 'downmix'
                % pay attention to energy loss due to differences in phase
                % when using this method. This is often the case for bad
                % stereo mixes
                nChannels = size(f_audio,2);
                
                f_audio = sum(f_audio,2);
                f_audio = f_audio / nChannels;
            otherwise
                disp('wav_to_audio: monoConvertMode : Unknown method')
        end
    end
end

bResampled = 0;
if parameter.useResampling
    if (fs ~= parameter.destSamplerate)
        bResampled = 1;
        if parameter.message == 1
            fprintf('Resampling to %d, ', parameter.destSamplerate);
        end
        f_audio = resample (f_audio,parameter.destSamplerate,fs,100);
        fs = parameter.destSamplerate;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update sideinfo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sideinfo.wav.version    = 1;
sideinfo.wav.filename   = wavfilename;
sideinfo.wav.dirRel    = dirRel;
sideinfo.wav.size       = size(f_audio,1);
sideinfo.wav.duration   = (sideinfo.wav.size-1)/fs;
sideinfo.wav.energy     = sum(f_audio.^2);
sideinfo.wav.fs         = fs;
% sideinfo.wav.nbits      = nbits;
sideinfo.wav.channels   = size(f_audio,2);
sideinfo.wav.resampled  = bResampled;
sideinfo.wav.monoConverted = bConverted_to_mono;
if bConverted_to_mono
    sideinfo.wav.monoConvertMode = parameter.monoConvertMode;
else
    sideinfo.wav.monoConvertMode = 'none';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Saving data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if parameter.save == 1
    if parameter.message == 1
        fprintf('Saving to file, ');
    end
    filename = strcat(parameter.saveFilename,'_audio');
    save(strcat(parameter.saveDir,filename),'f_audio','sideinfo');
end

if parameter.message == 1
    fprintf('Done\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if parameter.vis
    figure;
    for k=1:sideinfo.wav.channels
        if sideinfo.wav.channels > 1
            subplot(sideinfo.wav.channels,1,k);
        end
        plot( [0:sideinfo.wav.size-1] / sideinfo.wav.fs , f_audio(:,k));
        axis tight;
    end
end

end