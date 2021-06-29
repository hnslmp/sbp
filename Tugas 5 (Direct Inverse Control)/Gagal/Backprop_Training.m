clear;
close all;
%% Data Input
load('data.mat')
feature = dataset(1:7000,1:12);
feature = feature{:,:};
class = dataset(1:7000,13);
class = class{:,:};

[feature_row,feature_col] = size(feature);
[class_row,class_col] = size(class);
%% Normalisasi
% %Tanpa Normalisasi
% feature_norm = feature;

%Normalisasi Min-Max
feature_norm = zeros(size(feature));
for m = 1 : feature_row
    for n_train = 1 : feature_col
       feature_norm(m,n_train) = ((feature(m,n_train) - min(feature(:,n_train)))/(max(feature(:,n_train)) - min(feature(:,n_train))));
    end
end

% %Normalisasi Z-Score
% feature_norm = zscore(feature,[],1);
%% Hyperparameter & Declare Variable

%Hyperparameter
x_size  = feature_col; %Input layer 
z_size = 7;  %Hidden layer
y_size = class_col; %Output layer

alpha = 0.1;
epoch = 1000;
miu = 0;

%Variable
stop = 0;
error_target = 0.00001;
data_count = feature_row;
epoch_count = 1;

delta_zy_old = 0;
delta_xz_old = 0;
delta_zy_bias_old = 0;
delta_xz_bias_old = 0;
%% Initialisasi

%Random Init
rng(2) %Seed 
epsilon_init = 0.5; %Range random number

%Randomize weight and bias
weight_xz = rand(x_size, z_size) * 2 * epsilon_init - epsilon_init; 
weight_zy = rand(z_size, y_size) * 2 * epsilon_init - epsilon_init;
bias_xz = rand(1,z_size) * 2 * epsilon_init - epsilon_init;
bias_zy = rand(1,y_size) * 2 * epsilon_init - epsilon_init;

%Nguyen Widrow
beta = 0.7 * z_size^(1/x_size);

for i = 1:z_size
    norm(i) = sqrt(sum(weight_xz(:,i).^2));
    weight_xz(:,i) = beta*((weight_xz(:,i))/norm(i));
end

bias_xz = rand(1, z_size) * 2 * beta - beta;
bias_zy = rand(1, y_size) * 2 * beta - beta;
%% Backpropagation
while stop == 0 && epoch_count <= epoch
    true_count_training = 0;
    for n_train = 1:data_count  
        %Forward Pass
        %Input -> Hidden
        x_train = feature_norm(n_train,:);
        t_train = class(n_train,:);
        
        z_in_train = bias_xz + x_train*weight_xz;
        
        for m=1:z_size
            z_train(1,m) = 1/(1+exp(-z_in_train(1,m)));
        end
        
        %Hidden -> Output
        y_in_train = bias_zy + z_train*weight_zy;
        
        for l=1:y_size
            y_train(n_train,l) = 1/(1+exp(-y_in_train(1,l)));
        end
             
        %Threshold
        for s=1:1
            if y_train(n_train,s) >= 0.7
                y_train(n_train,s) = 1;
            end
            if y_train(n_train,s) <= 0.3
                y_train(n_train,s) = 0;
            end 
        end
     
        %Backward Pass
        %Output->Hidden
        for l=1:y_size
            do_k(1,l) = (y_train(n_train,l) - t_train(1,l)) * (y_train(n_train,l)*(1-y_train(n_train,l)));
        end
             
        delta_zy = (alpha .* z_train' * do_k);
        delta_zy_bias = alpha .* do_k;
        
        %Hidden->Input
        sigma_j = do_k * weight_zy';
        for m=1:z_size
            do_j(1,m) = (sigma_j(1,m)) .* (z_train(1,m)*(1-z_train(1,m)));
        end
        
        delta_xz = (alpha .* x_train' * do_j);
        delta_xz_bias = alpha .* do_j;
        
        %Momentum calculation
        momentum_zy = miu*delta_zy_old;
        momentum_xz = miu*delta_xz_old;
        momentum_bias_zy = miu*delta_zy_bias_old;
        momentum_bias_xz = miu*delta_xz_bias_old;
        
        %Weight Update
        weight_zy = weight_zy - delta_zy - momentum_zy;
        weight_xz = weight_xz - delta_xz - momentum_xz;
        bias_zy = bias_zy - delta_zy_bias - momentum_bias_zy;
        bias_xz = bias_xz - delta_xz_bias - momentum_bias_xz;
        
        %Cost function    
        error(1,n_train) = 0.5.*sum((t_train-y_train(n_train,:)).^2);
        
        %Momentum update
        delta_zy_old = delta_zy;
        delta_xz_old = delta_xz;
        delta_zy_bias_old = delta_zy_bias;
        delta_xz_bias_old = delta_xz_bias;
        
        %Recognition Rate
        if y_train(n_train,:) == t_train
           true_count_training = true_count_training + 1;
        end
    end 
    error_per_epoch(1,epoch_count) = sum(error)/feature_row;
    acc_training = true_count_training/data_count*100;
    acc_training_per_epoch(1,epoch_count) = acc_training;
    
    fprintf('\n\nEPOCH              :  %d \n',epoch_count);
    fprintf('TRAIN SCORE \n');
    fprintf('MSE Train           :  %.4f \n', sum(error)/feature_row);
    fprintf('Accuracy Train   :  %.2f \n', acc_training);

    if error_per_epoch(1,epoch_count) < error_target
        stop = 1;
    end
    
    Backprop_testing_1Layer
    epoch_count = epoch_count+1;
end

figure;
plot(error_per_epoch)
hold on;
plot(error_test_per_epoch)

figure;
plot(acc_training_per_epoch)