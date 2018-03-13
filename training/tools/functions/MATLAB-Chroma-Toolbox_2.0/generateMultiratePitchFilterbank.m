%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generating filter bank of filters corresponding MIDI pitches
%
% Pitches 21-59, fs = 882
% Pitches 60-95, fs = 4410
% Pitches 96-120, fs =22050
%
% Q                  (center frequency) / bandwidt, Q > 30 separates notes
% stop:              pass_rel = 1/(2*Q); stop_rel = stop*pass_rel;
% Rp                 loses no more than Rp dB in the passband
% Rs                 attenuation in the stopband in dB
%
% For details to filter desgin use MATLAB help function
%                    e.g., "help ellipord" and "help ellip"
%
% Attention: Construction of [b,a] may fail if the
%                   filter specification are too restrictive
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;

semitoneOffsets = [0, -0.25, -1/3, -0.5, -2/3, -0.75];
nameSuffixes = {''; '_minusQuarter'; '_minusThird'; '_minusHalf'; '_minusTwoThird'; '_minusThreeQuarters'};

for k=1:length(semitoneOffsets)
    
    
    midi = (1:128);                     % midi notes
    midi_freq = 2.^((midi-69+semitoneOffsets(k))/12)*440;  % computing frequencies of midi notes
    nameSuffix = nameSuffixes{k};
    h(120)=struct('a',[],'b',[]);
    
    disp(['Generating Filterbank: ',nameSuffix]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % fs = 22005, pitches 96-120
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fs = 22050;
    nyq = fs/2;
    midi_min = 96;
    midi_max = 120;
    Q = 25; stop = 2; Rp = 1; Rs = 50;
    pass_rel = 1/(2*Q);
    stop_rel = pass_rel*stop;
    
    for k = midi_min:midi_max;
        pitch = midi_freq(k);
        Wp = [pitch-pass_rel*pitch pitch+pass_rel*pitch]/nyq;
        Ws = [pitch-stop_rel*pitch pitch+stop_rel*pitch]/nyq;
        [n Wn]=ellipord(Wp,Ws,Rp,Rs);
        [h(k).b,h(k).a]=ellip(n,Rp,Rs,Wn);
    end
    num = midi_max-midi_min+1;
    h_fvtool = cell(2*num,1);
    for i = 1:num
        h_fvtool{2*i-1}=h(midi_min+i-1).b;
        h_fvtool{2*i}=h(midi_min+i-1).a;
    end
    fvtool(h_fvtool{:});
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % fs = 4410, pitches 60-95
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fs = 4410;
    nyq = fs/2;
    midi_min = 60;
    midi_max = 95;
    Q = 25; stop = 2; Rp = 1; Rs = 50;
    pass_rel = 1/(2*Q);
    stop_rel = pass_rel*stop;
    
    for k = midi_min:midi_max;
        pitch = midi_freq(k);
        Wp = [pitch-pass_rel*pitch pitch+pass_rel*pitch]/nyq;
        Ws = [pitch-stop_rel*pitch pitch+stop_rel*pitch]/nyq;
        [n Wn]=ellipord(Wp,Ws,Rp,Rs);
        [h(k).b,h(k).a]=ellip(n,Rp,Rs,Wn);
    end
    num = midi_max-midi_min+1;
    h_fvtool = cell(2*num,1);
    for i = 1:num
        h_fvtool{2*i-1}=h(midi_min+i-1).b;
        h_fvtool{2*i}=h(midi_min+i-1).a;
    end
    fvtool(h_fvtool{:});
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % fs = 882, pitches 21-59
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fs = 882;
    nyq = fs/2;
    midi_min = 21;
    midi_max = 59;
    Q = 25; stop = 2; Rp = 1; Rs = 50;
    pass_rel = 1/(2*Q);
    stop_rel = pass_rel*stop;
    
    for k = midi_min:midi_max;
        pitch = midi_freq(k);
        Wp = [pitch-pass_rel*pitch pitch+pass_rel*pitch]/nyq;
        Ws = [pitch-stop_rel*pitch pitch+stop_rel*pitch]/nyq;
        [n Wn]=ellipord(Wp,Ws,Rp,Rs);
        [h(k).b,h(k).a]=ellip(n,Rp,Rs,Wn);
    end
    num = midi_max-midi_min+1;
    h_fvtool = cell(2*num,1);
    for i = 1:num
        h_fvtool{2*i-1}=h(midi_min+i-1).b;
        h_fvtool{2*i}=h(midi_min+i-1).a;
    end
    fvtool(h_fvtool{:});
    
    
    save(['MIDI_FB_ellip_pitch_60_96_22050_Q25',nameSuffix],'h','-V6');
    
end

