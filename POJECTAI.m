Animaldatasetpath = fullfile("SET IMAGE");
imds = imageDatastore(Animaldatasetpath, ...
'IncludeSubfolders',true,'LabelSource','foldernames');


figure;
perm = randperm(5000,20);
for i = 1:20
subplot(4,5,i);
imshow(imds.Files{perm(i)});

end

labelCount = countEachLabel(imds);

img = readimage(imds,1);
size(img)

numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,250);

layers = [
  imageInputLayer([64 64 3])

  convolution2dLayer(5, 32, 'Padding', 'same', 'Stride', 1)
  batchNormalizationLayer
  reluLayer
  maxPooling2dLayer(2, 'Stride', 1)
 
 convolution2dLayer(3, 64, 'Padding', 'same')
  batchNormalizationLayer
  reluLayer
  maxPooling2dLayer(2, 'Stride', 2)

convolution2dLayer(3, 128, 'Padding', 'same', 'Stride', 1)
  batchNormalizationLayer
  reluLayer
  maxPooling2dLayer(2, 'Stride', 2)

convolution2dLayer(3, 256, 'Padding', 'same', 'Stride', 1)
  batchNormalizationLayer
  reluLayer
  maxPooling2dLayer(2, 'Stride', 2)

convolution2dLayer(3, 512, 'Padding', 'same', 'Stride', 1)
  batchNormalizationLayer
  reluLayer
  maxPooling2dLayer(2, 'Stride', 2)
  fullyConnectedLayer(5)
  softmaxLayer
  classificationLayer];

options = trainingOptions('sgdm', ...
  'InitialLearnRate', 0.001, ...
  'MaxEpochs', 83, ...
  'MiniBatchSize', 32, ...
  'Shuffle', 'every-epoch', ...
  'ValidationData', imdsValidation, ...
  'ValidationFrequency', 30, ...
  'Verbose', false, ...
  'Plots', 'training-progress');

net = trainNetwork(imdsTrain,layers,options);

%% 




%% 
inputSize = net.Layers(1).InputSize;

I = imread('11_output.jpg');
figure
imshow(I)
[label,scores] = classify(net,I);


figure
imshow(I)
classNames = net.Layers(end).ClassNames;
title(string(label) + ", " + num2str(100*scores(classNames == label),3) +"%");

[~,idx] = sort(scores,'descend');
idx = idx(5:-1:1);
classNamesTop = net.Layers(end).ClassNames(idx);
scoresTop = scores(idx);
figure
barh(scoresTop)
xlim([0 1])
title('Top 5 Predictions')
xlabel('Probability')
yticklabels(classNamesTop)



