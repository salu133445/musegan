%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: test_convert_pitch_to_CRP.m
% Date of Revision: 2011-03
% Programmer: Meinard Mueller, Sebastian Ewert
%
% Description: 
% * Computes CRP features (f_crp) from pitch features (f_pitch) 
% * CRP is a chroma-like feature tuned for timbre-invariance
%
% Reference: 
% Details on the feature computation can be found in the following articles:
%
% Meinard Mueller, Sebastian Ewert, and Sebastian Kreuzer
% Making chroma features more robust to timbre changes.
% Proceedings of IEEE International Conference on Acoustics, Speech, and
% Signal Processing (ICASSP), Taipei, Taiwan, pp. 1869-1872, 2009.
%
% Meinard Mueller, and Sebastian Ewert
% Towards Timbre-Invariant Audio Features for Harmony-Based Music.
% IEEE Transactions on Audio, Speach, and Language Processing.
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

directory = 'data_feature/';


%filename = 'Bach_BWV988-Aria-Measures1-4_Meinard_fast.wav';
%filename = 'Burgmueller_Op100-02-FirstPart_Meinard_SE.wav';
%filename = 'Systematic_Cadence-C-Major_Meinard_portato.wav';
%filename = 'Systematic_Cadence-C-Major_Meinard_staccato.wav';
%filename = 'Systematic_Scale-C-Major_Meinard_fast.wav';
%filename = 'Systematic_Scale-C-Major_Meinard_middle.wav';
filename = 'Systematic_Chord-C-Major_Eight-Instruments.wav';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loads pitch features (f_pitch) and computes CRP features (f_crp)
%
% Note: feature filename is specified by WAV filename
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

win_len = 4410;
filename_pitch = strcat(filename(1:end-4),'_pitch_',num2str(win_len));
load(strcat(directory,filename_pitch)); % load f_pitch and sideinfo;

parameter.coeffsToKeep = [55:120];
parameter.applyLogCompr = 1;
parameter.factorLogCompr = 1000;
parameter.featureRate = sideinfo.pitch.featureRate;
[f_crp,sideinfo] = pitch_to_CRP(f_pitch,parameter,sideinfo);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualization of CRP chromagram 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

parameter.featureRate = sideinfo.CRP.featureRate;
parameter.xlabel = 'Time [Seconds]';
parameter.title = 'CRP chromagram';
visualizeCRP(f_crp,parameter);

