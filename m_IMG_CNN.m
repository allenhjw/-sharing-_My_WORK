%% Description of the main program - CNN
% % 
% % using CNN to train and classify the image
% % 
% % 
%% Authorship and revision 
% %Author: Allen Hum J.W
% %
% %Revision: 0
% %Data:14 June 2019
% %Purpose: Created newly
%% Main code start here...
clc;close all;clear;
%% create the path to extract the data set into the database
%PATH='/users/allenhum/Documents/MATLAB/Experiment/COC/IMG/ALS/ML IMAGE (500x500)'; 
 PATH='/Volumes/My Book/4_PhD Research/IR Data/18-09-03-DATA-SAVE/Feat_for_ML/200x200/OP_img/RGB/'; 
% % the path include the foldername that is used as label
D_DATAPATH=fullfile(PATH); 
% % extract the data and store
imds=imageDatastore(D_DATAPATH,'IncludeSubfolders',true,'FileExtensions','.jpg'...
    ,'LabelSource','foldernames');

% % display the number of data in the database
imgCNT=countEachLabel(imds);
%% Preparing the training and validation data
numTrainFiles=0.6;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
NN_ARC=NNArchitecture200();
%% set up the CNN archetiture
% filtersize=[5 5];
% filtersize2=[7,7];
% filtersize3=[9,9];
% filternum=10;
% NN_ARC =[
%   
% % % input layer
%     imageInputLayer([500 500 1]) % must be the same size at the image size
%     
% % % 1st layer - 50 x 50 x 10
% % % A CNN - input size is 100 x 100 -> output size is 50 x 50 
% % % Input image is reduce to half. eg. 500 x 500 -> 250 x 250
%     convolution2dLayer(filtersize,filternum*3,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[2,2],'WeightL2Factor',0.01,'name','g_conv1_L1') %C1 
%     batchNormalizationLayer
%     %reluLayer % non-linear activation
%     convolution2dLayer(filtersize,filternum*3,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.01,'name','g_conv1_L2') %C1 
%     batchNormalizationLayer
%     convolution2dLayer(filtersize,filternum*3,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.01,'name','g_conv1_L3') %C1 
%     %convolution2dLayer(filtersize,filternum*3,'WeightsInitializer','he',...
%     %'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.01,'name','g_conv1_L4') %C1 
%     batchNormalizationLayer
%     eluLayer % 
%     maxPooling2dLayer(2,'stride',1) 
%     
% % % 2nd Layer - input 50 x 50  -> output size 25 x 25
%     convolution2dLayer(filtersize,filternum*3,'WeightsInitializer','he',...
%     'Padding',[3,3],'Stride',[2,2],'WeightL2Factor',0.05,'name','g_conv2_L1') %C2
%     batchNormalizationLayer
%     convolution2dLayer(filtersize,filternum*3,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.05,'name','g_conv2_L2') %C2
%     batchNormalizationLayer
%     convolution2dLayer(filtersize,filternum*4,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.05,'name','g_conv2_L3') %C2
%     batchNormalizationLayer
%     reluLayer % non-linear activation
%     %eluLayer 
%     maxPooling2dLayer(2,'stride',1) %S2 - reduce dimension by half, 0.5
% % % 3rd Layer input 25 x 25 -> output size 12 x 12
%     convolution2dLayer(filtersize,filternum*5,'WeightsInitializer','he',...
%     'Padding',[6,6],'Stride',[2,2],'WeightL2Factor',0.01,'name','g_conv3_L1') %C3
%     batchNormalizationLayer
%      convolution2dLayer(filtersize,filternum*5,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.01,'name','g_conv3_L2') %C3
%     batchNormalizationLayer
%     convolution2dLayer(filtersize,filternum*5,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.01,'name','g_conv3_L3') %C3
%     batchNormalizationLayer
%     reluLayer % non-linear activation
%     %eluLayer
%     maxPooling2dLayer(2,'stride',1) %C3 - reduce dimension by half, 0.5
% % % 4th Layer input size - 12x 12 ->output size 6 x6 
%     convolution2dLayer(filtersize,filternum*6,'WeightsInitializer','he',...
%     'Padding',[3,3],'Stride',[2,2],'WeightL2Factor',0.05,'name','g_conv4_L1') %C3
%     batchNormalizationLayer
%     convolution2dLayer(filtersize,filternum*6,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.05,'name','g_conv4_L2') %C3
%     batchNormalizationLayer
%     convolution2dLayer(filtersize,filternum*6,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.05,'name','g_conv4_L3') %C3
%     batchNormalizationLayer
%     reluLayer % non-linear activation
%     %eluLayer
%     maxPooling2dLayer(2,'stride',1) %C3 - reduce dimension by half, 0.5
%  % % 5th Layer input size 6 x 6 x 50 - > output size 1 x 1 
%     convolution2dLayer(filtersize2,filternum*7,'WeightsInitializer','he',...
%     'Padding',[3,3],'Stride',[2,2],'WeightL2Factor',0.01,'name','g_conv5_L1') %C3
%     batchNormalizationLayer
%     convolution2dLayer(filtersize,filternum*7,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.01,'name','g_conv5_L2') %C3
%     batchNormalizationLayer
%     convolution2dLayer(filtersize,filternum*7,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.01,'name','g_conv5_L3') %C3
%     batchNormalizationLayer
%     reluLayer % non-linear activation
%     %eluLayer
%     maxPooling2dLayer(2,'stride',1) %C3 - reduce dimension by half, 0.5   
% % % 6th Layer 1 x 1 x 60
%     convolution2dLayer(filtersize2,filternum*8,'WeightsInitializer','he',...
%     'Padding',[3,3],'Stride',[2,2],'WeightL2Factor',0.05,'name','g_conv6_L1') %C3
%     batchNormalizationLayer
%     convolution2dLayer(filtersize,filternum*8,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.05,'name','g_conv6_L2') %C3
%     batchNormalizationLayer
%     convolution2dLayer(filtersize,filternum*8,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.05,'name','g_conv6_L3') %C3
%     reluLayer
%     maxPooling2dLayer(2,'stride',1) %C3 - reduce dimension by half, 0.5  
% % % 7th Layer 
%     convolution2dLayer(filtersize2,filternum*9,'WeightsInitializer','he',...
%     'Padding',[3,3],'Stride',[2,2],'WeightL2Factor',0.01,'name','g_conv7_L1') %C3
%     batchNormalizationLayer
%     convolution2dLayer(filtersize,filternum*9,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.01,'name','g_conv7_L2') %C3
%     batchNormalizationLayer
%     convolution2dLayer(filtersize,filternum*9,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.01,'name','g_conv7_L3') %C3
%     reluLayer
%     maxPooling2dLayer(2,'stride',1) %C3 - reduce dimension by half, 0.5      
% % % 8th Layer 
%     convolution2dLayer(filtersize2,filternum*10,'WeightsInitializer','he',...
%     'Padding',[3,3],'Stride',[2,2],'WeightL2Factor',0.01,'name','g_conv8_L1') %C3
%     batchNormalizationLayer
%     convolution2dLayer(filtersize,filternum*10,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.01,'name','g_conv8_L2') %C3
%     batchNormalizationLayer
%     convolution2dLayer(filtersize,filternum*10,'WeightsInitializer','he',...
%     'Padding',[2,2],'Stride',[1,1],'WeightL2Factor',0.01,'name','g_conv8_L3') %C3
%     reluLayer
%     maxPooling2dLayer(2,'stride',1) %C3 - reduce dimension by half, 0.5          
%     
% % % Final layer
%     fullyConnectedLayer(filternum*10,'name','f_connect_9')
%     
%     reluLayer % non-linear activation
%     %eluLayer
%     dropoutLayer(0.5)
%     
%     fullyConnectedLayer(2,'name','f_connect_10')
%     softmaxLayer
%     classificationLayer
% ];

%% set up the training system
options = trainingOptions('adam', ...
    'InitialLearnRate',0.5e-3, ...
    'L2Regularization',0.00005,...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'Plots','training-progress');
% 'LearnRateSchedule','piecewise','LearnRateDropPeriod',10,'LearnRateDropFactor',0.92
%% Train the network using training data
net = trainNetwork(imdsTrain,NN_ARC,options);
save_NET=net;
% act1 = activations(net,imds,'g_conv1_L1'); % check the activation layer
%% Predict and classify
[YPred,score] = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);
[CONMAT,order]=confusionmat(YValidation,YPred);
fig=figure;
cm=confusionchart(YValidation,YPred);

fig_Position = fig.Position;
fig_Position(3) = fig_Position(3)*1.5;
fig.Position = fig_Position;

%cm.Normalization = 'column-normalized';
%sortClasses(cm,'descending-diagonal')
%cm.Normalization = 'absolute';  
%% SAVE the trained NN
save save_NET;

%% Create 200x200 NN functoin
function NN200=NNArchitecture200()

 F3CONV_SZ1=[3 3]; %filter size
F3CONV_num=8; %filter number
F3CONV_Pad1=[2 2]; %padding size
F3CONV_STD=[1 1]; %stride size
F3MP_STD=[2 2]; %pooling stride size
F3MPool_SZ=[2 2]; %pooling padding size
NN200 =[
        imageInputLayer([200 200 3]) % must be the same size at the image size
    %Conv1
        convolution2dLayer(F3CONV_SZ1,F3CONV_num*4,'WeightsInitializer','he',...
        'Padding',F3CONV_Pad1,'Stride',F3CONV_STD,'WeightL2Factor',0.01,'name','CH3conv_L2')
        batchNormalizationLayer('name','CH3BN_L3')
        convolution2dLayer(F3CONV_SZ1,F3CONV_num*4,'WeightsInitializer','he',...
        'Padding',F3CONV_Pad1,'Stride',F3CONV_STD,'WeightL2Factor',0.01,'name','CH3conv_L4')
        batchNormalizationLayer('name','CH3BN_L5')
        convolution2dLayer(F3CONV_SZ1,F3CONV_num*4,'WeightsInitializer','he',...
        'Padding',F3CONV_Pad1,'Stride',F3CONV_STD,'WeightL2Factor',0.01,'name','CH3conv_L4-1')
        batchNormalizationLayer('name','CH3BN_L5-1')
        reluLayer('name','CH3elu_L6') % 
        maxPooling2dLayer(F3MPool_SZ,'stride',F3MP_STD,'name','CH3maxpool_L7')
%         averagePooling2dLayer(F3MPool_SZ,'stride',F3MP_STD,'name','CH3avgpool_L4')

        %Conv2
        convolution2dLayer(F3CONV_SZ1,F3CONV_num*4,'WeightsInitializer','he',...
        'Padding',F3CONV_Pad1,'Stride',F3CONV_STD,'WeightL2Factor',0.01,'name','CH3conv_L8')
        batchNormalizationLayer('name','CH3BN_L9')
        convolution2dLayer(F3CONV_SZ1,F3CONV_num*4,'WeightsInitializer','he',...
        'Padding',F3CONV_Pad1,'Stride',F3CONV_STD,'WeightL2Factor',0.01,'name','CH3conv_L10')
        batchNormalizationLayer('name','CH3BN_L11')
        convolution2dLayer(F3CONV_SZ1,F3CONV_num*4,'WeightsInitializer','he',...
        'Padding',F3CONV_Pad1,'Stride',F3CONV_STD,'WeightL2Factor',0.01,'name','CH3conv_L10-1')
        batchNormalizationLayer('name','CH3BN_L11-1')
        reluLayer('name','CH3elu_L12') % 
        maxPooling2dLayer(F3MPool_SZ,'stride',F3MP_STD,'name','CH3maxpool_L13')
%         averagePooling2dLayer(F3MPool_SZ,'stride',F3MP_STD,'name','CH3avgpool_L8')

        %Conv3
        convolution2dLayer(F3CONV_SZ1,F3CONV_num*8,'WeightsInitializer','he',...
        'Padding',F3CONV_Pad1,'Stride',F3CONV_STD,'WeightL2Factor',0.01,'name','CH3conv_L14')
        batchNormalizationLayer('name','CH3BN_L15')
        convolution2dLayer(F3CONV_SZ1,F3CONV_num*8,'WeightsInitializer','he',...
        'Padding',F3CONV_Pad1,'Stride',F3CONV_STD,'WeightL2Factor',0.01,'name','CH3conv_L16')
        batchNormalizationLayer('name','CH3BN_L17')
        convolution2dLayer(F3CONV_SZ1,F3CONV_num*8,'WeightsInitializer','he',...
        'Padding',F3CONV_Pad1,'Stride',F3CONV_STD,'WeightL2Factor',0.01,'name','CH3conv_L16-1')
        batchNormalizationLayer('name','CH3BN_L17-1')
        reluLayer('name','CH3elu_L18') % 
        maxPooling2dLayer(F3MPool_SZ,'stride',F3MP_STD,'name','CH3maxpool_L19')
%         averagePooling2dLayer(F3MPool_SZ,'stride',F3MP_STD,'name','CH3avgpool_L12')

        %Conv4
        convolution2dLayer(F3CONV_SZ1,F3CONV_num*8,'WeightsInitializer','he',...
        'Padding',F3CONV_Pad1,'Stride',F3CONV_STD,'WeightL2Factor',0.01,'name','CH3conv_L20')
        batchNormalizationLayer('name','CH3BN_L21')
        convolution2dLayer(F3CONV_SZ1,F3CONV_num*8,'WeightsInitializer','he',...
        'Padding',F3CONV_Pad1,'Stride',F3CONV_STD,'WeightL2Factor',0.01,'name','CH3conv_L22')
        batchNormalizationLayer('name','CH3BN_L23')
        convolution2dLayer(F3CONV_SZ1,F3CONV_num*8,'WeightsInitializer','he',...
        'Padding',F3CONV_Pad1,'Stride',F3CONV_STD,'WeightL2Factor',0.01,'name','CH3conv_L22-1')
        batchNormalizationLayer('name','CH3BN_L23-1')
        reluLayer('name','CH3elu_L24') % 
        maxPooling2dLayer(F3MPool_SZ,'stride',F3MP_STD,'name','CH3maxpool_L25')
%         averagePooling2dLayer(F3MPool_SZ,'stride',F3MP_STD,'name','CH3avgpool_L16')
        %Conv5
        convolution2dLayer(F3CONV_SZ1,F3CONV_num*16,'WeightsInitializer','he',...
        'Padding',F3CONV_Pad1,'Stride',F3CONV_STD,'WeightL2Factor',0.01,'name','CH3conv_L26')
        batchNormalizationLayer('name','CH3BN_L27')
        reluLayer('name','CH3elu_L28') %
%         dropoutLayer(0.8,'name','CH3DOL_L20');
        maxPooling2dLayer(F3MPool_SZ,'stride',F3MP_STD,'name','CH3maxpool_L29')
%         averagePooling2dLayer(F3MPool_SZ,'stride',F3MP_STD,'name','CH3avgpool_L21')
        %Conv6
        convolution2dLayer(F3CONV_SZ1,F3CONV_num*16,'WeightsInitializer','he',...
        'Padding',F3CONV_Pad1,'Stride',F3CONV_STD,'WeightL2Factor',0.01,'name','CH3conv_L30')
        batchNormalizationLayer('name','CH3BN_L31')
        reluLayer('name','CH3elu_L32') %
%         dropoutLayer(0.8,'name','CH3DOL_L25');
        maxPooling2dLayer(F3MPool_SZ,'stride',F3MP_STD,'name','CH3maxpool_L33')
%         averagePooling2dLayer(F3MPool_SZ,'stride',F3MP_STD,'name','CH3avgpool_L26')
        %Conv7
        convolution2dLayer(F3CONV_SZ1,F3CONV_num*32,'WeightsInitializer','he',...
        'Padding',F3CONV_Pad1,'Stride',F3CONV_STD,'WeightL2Factor',0.01,'name','CH3conv_L34')
        batchNormalizationLayer('name','CH3BN_L35')
        reluLayer('name','CH3elu_L36') %
%         dropoutLayer(0.8,'name','CH3DOL_L30');
        maxPooling2dLayer(F3MPool_SZ,'stride',F3MP_STD,'name','CH3maxpool_L37')
%         averagePooling2dLayer(F3MPool_SZ,'stride',F3MP_STD,'name','CH3avgpool_L31')
%         %Conv8
%         convolution2dLayer(F3CONV_SZ1,F3CONV_num*9,'WeightsInitializer','he',...
%         'Padding',F3CONV_Pad1,'Stride',F3CONV_STD,'WeightL2Factor',0.01,'name','CH3conv_L32')
%         batchNormalizationLayer('name','CH3BN_L33')
%         eluLayer('name','CH3elu_L34') %
%         dropoutLayer(0.5,'name','CH3DOL_L35');
% %         maxPooling2dLayer(F3MPool_SZ,'stride',F3MP_STD,'name','CH3maxpool_L36')
%         averagePooling2dLayer(F3MPool_SZ,'stride',F3MP_STD,'name','CH3avgpool_L36')
        fullyConnectedLayer(F3CONV_num*32, 'name','CH3FC_L38')
        dropoutLayer(0.5,'name','CH3DOL_L39');
        reluLayer % non-linear activation
        %eluLayer
        fullyConnectedLayer(3,'name','f_connect_10')
        softmaxLayer
        classificationLayer
];

end