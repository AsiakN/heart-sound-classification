function [feature_table_all,N] = extractFeatures(fds, window_length, window_overlap, reference_table)
%Function to extract only features for heart sound classification demo.
%Copyright (c) 2017, MathWorks, Inc. 

warning off

overlap_length = window_length * window_overlap / 100;
step_length = window_length - overlap_length;

feature_table_all = table();
% 
 labelMap = containers.Map('KeyType','int32','ValueType','char');
 keySet = {-1, 1};
 valueSet = {'Normal','Abnormal'};
 labelMap = containers.Map(keySet,valueSet);

while hasdata(fds)
    PCG = read(fds);
    signal = PCG.data;
    fs = PCG.fs;
    
    springer_options   = default_Springer_HSMM_options;
    % resample to 1000 Hz
    PCG_resampled = resample(signal,springer_options.audio_Fs,fs); % resample to springer_options.audio_Fs (1000 Hz)
   
% filter the signal between 25 to 400 Hz
    PCG_resampled = butterworth_low_pass_filter(PCG_resampled,2,400,springer_options.audio_Fs, false);
    %PCG_resampled = butterworth_high_pass_filter(PCG_resampled,2,25,springer_options.audio_Fs);

    % remove spikes
    PCG_resampled = schmidt_spike_removal(PCG_resampled,springer_options.audio_Fs);
    
    current_class = reference_table(strcmp(reference_table.record_name, PCG.filename), :).record_label(1);
    N = length(PCG_resampled);        % Bernhard: to keep track of #samples in files we process
    number_of_windows = floor( (N - overlap_length*springer_options.audio_Fs) / (springer_options.audio_Fs * step_length));
    
    feature_table = table();
    for iwin = 1:number_of_windows
        current_start_sample = (iwin - 1) * springer_options.audio_Fs * step_length + 1;
        current_end_sample = current_start_sample + window_length *springer_options.audio_Fs- 1;
        current_signal = PCG_resampled(current_start_sample:current_end_sample);

        % Calculate mean value feature
        feature_table.meanValue(iwin, 1) = mean(current_signal);
        
        % Calculate median value feature
        feature_table.medianValue(iwin, 1) = median(current_signal);

        % Calculate standard deviation feature
        feature_table.standardDeviation(iwin, 1) = std(current_signal);
        
        % Calculate mean absolute deviation feature
        feature_table.meanAbsoluteDeviation(iwin, 1) = mad(current_signal);
        
        % Calculate signal 25th percentile feature
        feature_table.quantile25(iwin, 1) = quantile(current_signal, 0.25);
        
        % Calculate signal 75th percentile feature
        feature_table.quantile75(iwin, 1) = quantile(current_signal, 0.75);
        
        % Calculate signal inter quartile range feature
        feature_table.signalIQR(iwin, 1) = iqr(current_signal);
        
        % Calculate skewness of the signal values
        feature_table.sampleSkewness(iwin, 1) = skewness(current_signal);

        % Calculate kurtosis of the signal values
        feature_table.sampleKurtosis(iwin, 1) = kurtosis(current_signal);

        % Calculate Shannon's entropy value of the signal
        feature_table.signalEntropy(iwin, 1) = signal_entropy(current_signal');

        % Calculate spectral entropy of the signal
        feature_table.spectralEntropy(iwin, 1) = spectral_entropy(current_signal, springer_options.audio_Fs, 256);

        % Extract features from the power spectrum
        [maxfreq, maxval, maxratio] = dominant_frequency_features(current_signal, springer_options.audio_Fs, 256, 0);
        feature_table.dominantFrequencyValue(iwin, 1) = maxfreq;
        feature_table.dominantFrequencyMagnitude(iwin, 1) = maxval;
        feature_table.dominantFrequencyRatio(iwin, 1) = maxratio;
        
        % Extract wavelet features
        % REMOVED because didn't contribute much to final model (only 1 of
        % them was selected by NCA among the "important" features)
        %feature_table.spectralflux(iwin,,1) = spectralFlux(current_signal,springer_options.audio_Fs) 
        % Extract Mel-frequency cepstral coefficients
        Tw = window_length*1000;% analysis frame duration (ms)
        Ts = 10;                % analysis frame shift (ms)
        alpha = 0.97;           % preemphasis coefficient
        M = 20;                 % number of filterbank channels 
        C = 12;                 % number of cepstral coefficients
        L = 22;                 % cepstral sine lifter parameter
        LF = 20;                 % lower frequency limit (Hz)
        HF = 500;               % upper frequency limit (Hz)

        [MFCCs, ~, ~] = mfcc(current_signal,  springer_options.audio_Fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L);
        feature_table.MFCC1(iwin, 1) = MFCCs(1);
        feature_table.MFCC2(iwin, 1) = MFCCs(2);
        feature_table.MFCC3(iwin, 1) = MFCCs(3);
        feature_table.MFCC4(iwin, 1) = MFCCs(4);
        feature_table.MFCC5(iwin, 1) = MFCCs(5);
        feature_table.MFCC6(iwin, 1) = MFCCs(6);
        feature_table.MFCC7(iwin, 1) = MFCCs(7);
        feature_table.MFCC8(iwin, 1) = MFCCs(8);
        feature_table.MFCC9(iwin, 1) = MFCCs(9);
        feature_table.MFCC10(iwin, 1) = MFCCs(10);
        feature_table.MFCC11(iwin, 1) = MFCCs(11);
        feature_table.MFCC12(iwin, 1) = MFCCs(12);
        feature_table.MFCC13(iwin, 1) = MFCCs(13);
        %featurex =  spectralFlux(current_signal,springer_options.audio_Fs);
        %feature_table.spec_rollof(iwin, 1)= spectralRolloffPoint(current_signal,springer_options.audio_Fs);
        %feature_table.spec_centroid(iwin, 1) = spectralCentroid(current_signal, springer_options.audio_Fs, "Window",hamming(256), "OverlapLength", overlap_length);

        % Assign class label to the observation
          if iwin == 1
             feature_table.class = {labelMap(current_class)};
          else
              feature_table.class{iwin, :} = labelMap(current_class);
          end
        
    end
    
    feature_table_all = [feature_table_all; feature_table];
end