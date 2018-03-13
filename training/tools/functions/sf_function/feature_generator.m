function feature = feature_generator( filename, chroma_type, winLenSTMSP, vis)
    
    if(nargin < 3) 
        winLenSTMSP = 4410;
        vis = 0;
    end
    if(nargin < 2)  chroma_type = 'clp'; end
    
    if(strcmp(chroma_type, 'cens'))
        [f_audio,sideinfo] = wav_to_audio('', '', filename);
        paramPitch.winLenSTMSP = winLenSTMSP;
        paramPitch.visualize = 0;
        [f_pitch,sideinfo] = audio_to_pitch_via_FB(f_audio,paramPitch,sideinfo);
        paramCENS.winLenSmooth = 21;
        paramCENS.downsampSmooth = 1;
        paramCENS.visualize = vis;
        paramCENS.inputFeatureRate = sideinfo.pitch.featureRate;
        [f_CENS,sideinfo] = pitch_to_CENS(f_pitch,paramCENS,sideinfo);
        feature = f_CENS;
    elseif(strcmp(chroma_type, 'clp'))
        [f_audio,sideinfo] = wav_to_audio('', '', filename);
        paramPitch.winLenSTMSP = winLenSTMSP;
        paramPitch.visualize = 0;
        [f_pitch,sideinfo] = audio_to_pitch_via_FB(f_audio,paramPitch,sideinfo);
        paramCLP.applyLogCompr = 1;
        paramCLP.factorLogCompr = 100;
        paramCLP.visualize = vis;
        paramCLP.inputFeatureRate = sideinfo.pitch.featureRate;
        [f_CLP,sideinfo] = pitch_to_chroma(f_pitch,paramCLP,sideinfo);
        feature = f_CLP; 
    elseif(strcmp(chroma_type, 'crp'))
        [f_audio,sideinfo] = wav_to_audio('', '', filename);
        paramPitch.winLenSTMSP = winLenSTMSP;
        paramPitch.visualize = 0;
        [f_pitch,sideinfo] = audio_to_pitch_via_FB(f_audio,paramPitch,sideinfo);
        paramCRP.coeffsToKeep = [55:120];
        paramCRP.inputFeatureRate = sideinfo.pitch.featureRate;
        paramCRP.visualize = vis;
        [f_CRP,sideinfo] = pitch_to_CRP(f_pitch,paramCRP,sideinfo);
        feature = f_CRP;
    else
        load(chroma_type);
    end
end

