%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: test_convert_pitch_to_CENS.m
% Date of Revision: 2011-03
% Programmer: Meinard Mueller, Sebastian Ewert
%
% Description: 
% * Computes CENS features (f_CENS) from pitch features (f_pitch) 
% * CENS is a chroma-like feature 
%   (Chroma Energy Normalized Statistics)
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
%close all hidden;

directory = 'data_feature/';


%filename = 'Bach_BWV988-Aria-Measures1-4_Meinard_fast.wav';
%filename = 'Burgmueller_Op100-02-FirstPart_Meinard_SE.wav';
%filename = 'Systematic_Cadence-C-Major_Meinard_portato.wav';
%filename = 'Systematic_Cadence-C-Major_Meinard_staccato.wav';
%filename = 'Systematic_Scale-C-Major_Meinard_fast.wav';
%filename = 'Systematic_Scale-C-Major_Meinard_middle.wav';
filename = 'Systematic_Chord-C-Major_Eight-Instruments.wav';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loads pitch features (f_pitch) and computes CENS features (f_CENS)
%
% Note: feature filename is specified by WAV filename
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

win_len = 4410;
filename_pitch = strcat(filename(1:end-4),'_pitch_',num2str(win_len));
load(strcat(directory,filename_pitch)); % load f_pitch and sideinfo;

parameter.winLenSmooth = 21;
parameter.downsampSmooth = 5;
parameter.featureRate = sideinfo.pitch.featureRate;
[f_CENS,sideinfo] = pitch_to_CENS(f_pitch,parameter,sideinfo);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualization of chromagrams (f_CENS,f_chroma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

parameter.featureRate = sideinfo.CENS.featureRate;
parameter.xlabel = 'Time [Seconds]';
parameter.title = sprintf('CENS %d %d chromagram',parameter.winLenSmooth,parameter.downsampSmooth);
visualizeChroma(f_CENS,parameter);


