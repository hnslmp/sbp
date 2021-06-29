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
feature_norm = zeros(size(feature));
for m = 1 : feature_row
    for n = 1 : feature_col
       feature_norm(m,n) = ((feature(m,n) - min(feature(:,n)))/(max(feature(:,n)) - min(feature(:,n))));
    end
end
%% Hyperparameter & Declare Variable

%Hyperparameter
x_size  = feature_col; %Input layer 
z_size = 7;  %Hidden layer
y_size = class_col; %Output layer

alpha = 0.15;
epoch = 2000;
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
    for n = 1:data_count  
        %Forward Pass
        %Input -> Hidden
        x = feature_norm(n,:);
        t = class(n,:);
        
        z_in = bias_xz + x*weight_xz;
        
        for m=1:z_size
            z(1,m) = 1/(1+exp(-z_in(1,m)));
        end
        
        %Hidden -> Output
        y_in = bias_zy + z*weight_zy;
        
        for l=1:y_size
            y(n,l) = 1/(1+exp(-y_in(1,l)));
        end
             
        %Threshold
        for s=1:1
            if y(n,s) >= 0.7
                y(n,s) = 1;
            end
            if y(n,s) <= 0.3
                y(n,s) = 0;
            end 
        end
     
        %Backward Pass
        %Output->Hidden
        for l=1:y_size
            do_k(1,l) = (y(n,l) - t(1,l)) * (y(n,l)*(1-y(n,l)));
        end
             
        delta_zy = (alpha .* z' * do_k);
        delta_zy_bias = alpha .* do_k;
        
        %Hidden->Input
        sigma_j = do_k * weight_zy';
        for m=1:z_size
            do_j(1,m) = (sigma_j(1,m)) .* (z(1,m)*(1-z(1,m)));
        end
        
        delta_xz = (alpha .* x' * do_j);
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
        error(1,n) = 0.5.*sum((t-y(n,:)).^2);
        
        %Momentum update
        delta_zy_old = delta_zy;
        delta_xz_old = delta_xz;
        delta_zy_bias_old = delta_zy_bias;
        delta_xz_bias_old = delta_xz_bias;
    end 
    error_per_epoch(1,epoch_count) = sum(error)/feature_row;
    
    disp(epoch_count)
    disp(sum(error)/feature_row)
    
    if error_per_epoch(1,epoch_count) < error_target
            stop = 1;
    end
    
    epoch_count = epoch_count+1;
    
end

plot(error_per_epoch)
Backprop_testing