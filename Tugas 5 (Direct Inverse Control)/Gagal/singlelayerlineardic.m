clear; clc
%% 
% *Import dataset.*

load("databasekufinal.mat");
load("parameteridenlin1.mat");
%% 
% *Split data*

x_train = [u(1:1500,1) u(2:1501, 1) out(4:1503,1) out(3:1502,1)];%[u(k-3) u(k-2) y(k) y(k-1)]
%
t_train = u(3:1502, 1);

x_test = [zeros(1500,1) zeros(1500,1) out(1507:3006,1) out(1506:3005,1)];%[u(k-3) u(k-2) y(k) y(k-1)]
t_test = u(1506:3005, 1);
yfin_test=out(1507:3006,1);
%x_test = [zeros(1500,1) zeros(1500,1) zeros(1500,1)+0.3 zeros(1500,1)+0.3];
%yfin_test=zeros(1500,1)+0.3;
%
%% 
% *Normalizations.*

[row_train, column_train] = size(x_train);
[row_test, column_test] = size(x_test);
[row_target, column_target] = size(t_train);
[row_target_test, column_target_test] = size(t_test);
[row_wijiden, column_wijiden] = size(w_ij1);
[row_wjkiden, column_wjkiden] = size(w_jk1);


%% 
% *Parameters of Neural Networks.*

input_neurons = column_train;
output_neurons = column_target;
hidden_layer = 15;

alpha = 0.01;
epoch = 250;
mu = 0.4;

epoch_iter = 1;
error_min = 0.001;
data_count = row_target;
%% 
% *Initializations.*

w_ij = rand(input_neurons,hidden_layer) - 0.5;
w_jk = rand(hidden_layer, output_neurons) - 0.5;

bias_j = rand(1, hidden_layer) - 0.5;
bias_k = rand(1, output_neurons) - 0.5;

norm = zeros(hidden_layer, 1);

beta = 0.7*hidden_layer^(1/input_neurons);

for j = 1:hidden_layer
    norm(j) = sqrt(sum(w_ij(:,j).^2)); 
    w_ij(:,j) = beta*((w_ij(:,j))/norm(j));
end
%% 
% *Backpropagation.*

delta_wij_old = zeros(input_neurons,hidden_layer);
delta_wjk_old = zeros(hidden_layer,output_neurons);

f_zin = zeros(1, hidden_layer);
y = zeros(row_train, output_neurons);

do_k = zeros(1, output_neurons);
do_j = zeros(1, hidden_layer);

error = zeros(1,row_train);
epoch_error = zeros(1, epoch);
%% 
% Train the model.

flag = 0;

while epoch_iter <= epoch && flag == 0
    
    for n = 1:row_train
        
        x = x_train(n, :);
        t = t_train(n, :);
        
        %% Feedforward
        % Input layer to hidden layer
        z_in = x * w_ij + bias_j;
        
        for m = 1:hidden_layer
            f_zin(1,m) = ((exp(z_in(1,m))-exp(-z_in(1,m)))/(exp(z_in(1,m))+exp(-z_in(1,m))));
        end
        
        % Hidden layer to output layer
        y_in = f_zin * w_jk + bias_k;
        
        for k = 1:output_neurons
            y(n,k) = y_in;
        end
        
        
    %% Backpropagate
        % Output layer to hidden layer
        for k = 1:output_neurons
             do_k(1,k) = (t(1,k) - y(n,k));
        end
        
        delta_wjk = alpha .* f_zin' * do_k;
        delta_bias_k = alpha .* do_k;
        
        % Hidden layer to input layer
        do_in = do_k * w_jk';
        
        for j = 1:hidden_layer
            do_j(1,j) = (do_in(1,j)) .* (1-(((exp(z_in(1,j))-exp(-z_in(1,j)))/(exp(z_in(1,j))+exp(-z_in(1,j)))).^2));
        end
        
        delta_wij = alpha .* x' * do_j;
        delta_bias_j = alpha .* do_j;
        
        % Weighing Update
        momentum_j = mu * delta_wij_old;
        momentum_k = mu * delta_wjk_old;
        
        w_ij = w_ij + delta_wij + momentum_j;
        w_jk = w_jk + delta_wjk + momentum_k;
        bias_j = bias_j + delta_bias_j;
        bias_k = bias_k + delta_bias_k;
        
        delta_wij_old = delta_wij;
        delta_wjk_old = delta_wjk;
        
        % Error Calculation
        error(1,n) = sum((t-y(n,:)).^2);
    end
    
    epoch_error(1, epoch_iter) = sum(error)/row_train;
    if epoch_error(1, epoch_iter) < error_min
        flag = 1;
    end
    
    epoch_iter = epoch_iter + 1;
    
end
%% 
% 
sum(error)/row_train
plot(epoch_error)
title("training");
%% 
% *Testing the Model*

input_neurons_test = column_test;
output_neurons_test = column_target_test;
hidden_layer_test = hidden_layer;

f_zin = zeros(1, hidden_layer_test);
f_zini = zeros(1, column_wijiden);
y = zeros(row_test, output_neurons_test);
yi = zeros(row_test, column_wjkiden);
%% 
% Feedforward

errortest=0;
SAE=0;
u1=0;
u2=0;
u3=0;
y1i=0;
y2i=0;
for n = 1:row_test
    x_test(n,1) = u2;
    x_test(n,2) = u1;
    x = x_test(n,:);
    t = t_test(n,:);
    u3 = u2;
    u2 = u1;
    %% Feedforward
    % Input layer to hidden layer
    z_in = x * w_ij + bias_j;
        
    for m = 1:hidden_layer_test
        f_zin(1,m) = ((exp(z_in(1,m))-exp(-z_in(1,m)))/(exp(z_in(1,m))+exp(-z_in(1,m))));
    end
        
    % Hidden layer to output layer
    y_in = f_zin * w_jk + bias_k;
        
    for k = 1:output_neurons_test
        y(n,k) = y_in;
        u1=y(n,k);
    end
    %% Feedforward iden
    % Input layer to hidden layer
    xiden = [u2 u1 y1i y2i];
    z_ini = xiden * w_ij1 + bias_j1;
    y2i=y1i;
    for m = 1:column_wijiden
        f_zini(1,m) = ((exp(z_ini(1,m))-exp(-z_ini(1,m)))/(exp(z_ini(1,m))+exp(-z_ini(1,m))));
    end
        
    % Hidden layer to output layer
    y_ini = f_zini * w_jk1 + bias_k1;
        
    for k = 1:column_wjkiden
        yi(n,k) = y_ini;
    end
    y1i=yi(n);
    errortest = sum((yfin_test(n,:)-yi(n,:)).^2)+errortest;
    SAE=sum(abs(yfin_test(n,:)-yi(n,:)))+SAE;
end

figure()
plot(1:1:100,yi(1:100,1))
title("Hasil komparasi")
hold on
plot(1:1:100,yfin_test(1:100,1))
legend("Output","Setpoint");
hold off
SAE
SSE=errortest
MSE = SSE/row_test