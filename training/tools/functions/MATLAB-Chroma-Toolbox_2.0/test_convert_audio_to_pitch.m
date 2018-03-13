%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: test_convert_audio_to_pitchSTMSP.m
% Date of Revision: 2011-03
% Programmer: Meinard Mueller, Sebastian Ewert
%
% Description: 
% * Computes pitch subband decomposition of WAV file
%   (default: MIDI pitches 21 to 108) 
% * each pitch subband contains short time mean-square power (STMSP) 
% * Features are computed in a batch modus
% * Features are stored in folder 'data_feature/'
%
% Reference: 
% Details on the feature computation can be found in the following book:
%
% Meinard Mueller: Information Retrieval for Music and Motion,
%                  Springer 2007
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
clear;
close all hidden;

dirFileNames = {
    'data_WAV/','Bach_BWV988-Aria-Measures1-4_Meinard_fast.wav';
    'data_WAV/','Burgmueller_Op100-02-FirstPart_Meinard_SE.wav';
    'data_WAV/','Systematic_Cadence-C-Major_Meinard_portato.wav';
    'data_WAV/','Systematic_Cadence-C-Major_Meinard_staccato.wav';
    'data_WAV/','Systematic_Scale-C-Major_Meinard_fast.wav';
    'data_WAV/','Systematic_Scale-C-Major_Meinard_middle.wav';
    'data_WAV/','Systematic_Chord-C-Major_Eight-Instruments.wav';
    };

for n=1:size(dirFileNames,1)
    clear parameter;
    parameter.message = 1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Convert WAV to expected audio format (mono, 22050 Hz)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [f_audio,sideinfo] = wav_to_audio('', dirFileNames{n,1}, dirFileNames{n,2},parameter);
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Estimation of the global tuning of the recording and selection of
    % an appropriate filterbank for use in the next step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clear parameter
    shiftFB = estimateTuning(f_audio);
    fprintf('Using filterbank number: %d\n',shiftFB);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute pitch features
    % 
    % Input: audio file of format: mono, 22050 Hz
    %
    % Output: sequence of pitch vectors 
    %         (specified by N x 120 matrix f_pitch)
    %         Only subband for MIDI pitches 21 to 108 are computed, the
    %         other subbands are set to zero.
    %
    % Parameter: parameter.win_len specifies window length (in samples)
    %            with window overlap of half size  
    %            Example: audio sampling rate: 22050 Hz
    %                     parameter.win_len = 4410
    %                     Resulting feature rate: 10 Hz
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    clear parameter
    parameter.winLenSTMSP = 4410;
    parameter.fs = sideinfo.wav.fs;
    parameter.save = 1;
    parameter.saveDir = 'data_feature/';
    parameter.saveFilename = dirFileNames{n,2}(1:end-4);
    parameter.shiftFB = shiftFB;
    parameter.saveAsTuned = 1;
    [f_pitch,sideinfo] = audio_to_pitch_via_FB(f_audio,parameter,sideinfo);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Visualization of pitch decomposition (f_pitch)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    parameter.usePitchNameLabels = 1;
    parameter.title = 'Logarithmic compression of amplitude';
    parameter.featureRate = sideinfo.pitch.featureRate;
    parameter.xlabel = 'Time [Seconds]';
    parameter.ylabel = 'Pitch';
    visualizePitch(log(5*f_pitch+1),parameter);
end
