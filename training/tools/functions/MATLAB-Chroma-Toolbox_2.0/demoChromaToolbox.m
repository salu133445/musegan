clear
close all

filename = 'Systematic_Chord-C-Major_Eight-Instruments.wav';
[f_audio,sideinfo] = wav_to_audio('', 'data_WAV/', filename);
shiftFB = estimateTuning(f_audio);

paramPitch.winLenSTMSP = 4410;
paramPitch.shiftFB = shiftFB;
paramPitch.visualize = 1;
[f_pitch,sideinfo] = ...
    audio_to_pitch_via_FB(f_audio,paramPitch,sideinfo);

paramCP.applyLogCompr = 0;
paramCP.visualize = 1;
paramCP.inputFeatureRate = sideinfo.pitch.featureRate;
[f_CP,sideinfo] = pitch_to_chroma(f_pitch,paramCP,sideinfo);

paramCLP.applyLogCompr = 1;
paramCLP.factorLogCompr = 100;
paramCLP.visualize = 1;
paramCLP.inputFeatureRate = sideinfo.pitch.featureRate;
[f_CLP,sideinfo] = pitch_to_chroma(f_pitch,paramCLP,sideinfo);

paramCENS.winLenSmooth = 21;
paramCENS.downsampSmooth = 5;
paramCENS.visualize = 1;
paramCENS.inputFeatureRate = sideinfo.pitch.featureRate;
[f_CENS,sideinfo] = pitch_to_CENS(f_pitch,paramCENS,sideinfo);

paramCRP.coeffsToKeep = [55:120];
paramCRP.visualize = 1;
paramCRP.inputFeatureRate = sideinfo.pitch.featureRate;
[f_CRP,sideinfo] = pitch_to_CRP(f_pitch,paramCRP,sideinfo);

paramSmooth.winLenSmooth = 21;
paramSmooth.downsampSmooth = 5;
paramSmooth.inputFeatureRate = sideinfo.CRP.featureRate;
[f_CRPSmoothed, featureRateSmoothed] = ...
    smoothDownsampleFeature(f_CRP,paramSmooth);
parameterVis.featureRate = featureRateSmoothed;
visualizeCRP(f_CRPSmoothed,parameterVis);


% filename = 'Systematic_Chord-C-Major_Eight-Instruments.wav';
% [f_audio,sideinfo] = wav_to_audio('', 'data_WAV/', filename);
% shiftFB = estimateTuning(f_audio);
%
% paramPitch.win_len = 4410;
% paramPitch.shiftFB = shiftFB;
% paramPitch.visualize = 1;
% [f_pitch,sideinfo] = ...
%     audio_to_pitch_via_FB(f_audio,paramPitch,sideinfo);
%
% paramCP.applyLogCompr = 0;
% paramCP.visualize = 1;
% paramCP.inputFeatureRate = sideinfo.pitch.featureRate;
% [f_CP,sideinfo] = pitch_to_chroma(f_pitch,paramCP,sideinfo);
%
% paramCLP.applyLogCompr = 1;
% paramCLP.logParamMult = 100;
% paramCLP.visualize = 1;
% paramCLP.inputFeatureRate = sideinfo.pitch.featureRate;
% [f_CLP,sideinfo] = pitch_to_chroma(f_pitch,paramCLP,sideinfo);
%
% paramCENS.stat_window_length = 21;
% paramCENS.stat_downsample = 5;
% paramCENS.visualize = 1;
% paramCENS.inputFeatureRate = sideinfo.pitch.featureRate;
% [f_CENS,sideinfo] = pitch_to_CENS(f_pitch,paramCENS,sideinfo);
%
% paramCRP.coeffsToKeep = [55:120];
% paramCRP.visualize = 1;
% paramCRP.inputFeatureRate = sideinfo.pitch.featureRate;
% [f_CRP,sideinfo] = pitch_to_CRP(f_pitch,paramCRP,sideinfo);
%
% paramSmooth.stat_window_length = 21;
% paramSmooth.stat_downsample = 5;
% paramSmooth.inputFeatureRate = sideinfo.CRP.featureRate;
% [f_CRPSmoothed, featureRateSmoothed] = ...
%     smoothDownsampleFeature(f_CRP,paramSmooth);
% parameterVis.featureRate = featureRateSmoothed;
% visualize_CRP(f_CRPSmoothed,parameterVis);


