function label = classifyHeartSounds(sound_signal, sampling_frequency) %#codegen
%Label new observations using trained SVM model Mdl. The function takes 
%sound signal and sampling frequency as input and produces a classification
%of 'Normal' or 'Abnormal'
%Copyright (c) 2016, MathWorks, Inc. 

% Window length for feature extraction in seconds
win_len = 5;

% Overlap between adjacent windows in percentage
win_overlap = 0;

% Extract features
%features = extractFeaturesCodegen(sound_signal, sampling_frequency, win_len, win_overlap);
features = extractFeatures_prod(sound_signal, sampling_frequency, win_len, win_overlap);

% Load saved model
Mdl = loadLearnerForCoder('HeartSoundClassificationModel_SVM');
%Mdl_ensemble  = loadLearnerForCoder('HeartSoundClassificationModel_Ensemble');
% Predict classification for all windows
predicted_labels = predict(Mdl,features);

% Predict abnormal if even one window sounds abnormal

% Display Prediction
if  sum(strcmp(predicted_labels,'Abnormal')) >= numel(predicted_labels)/2
    label = 'Abnormal';
    disp 'Heart Sound  is Abnormal'
else
    label = 'Normal';
    disp 'Heart Sound is Normal'
end


end