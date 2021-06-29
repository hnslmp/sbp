clc; 
clear; 
close all;

%% output random
rng(0,'twister'); %Mulai seed dan fungsi randomizer
u = (1-(-1)).*rand(3010,1)-1; %nilai random

y=0;
for i = 2:length(u)
    if i == 1 then
        y(i)=0;
    end
y(i)=(1/(1+y(i-1)^2)) + 0.25*u(i) - 0.300*u(i-1);
end

%% output sinusoid
u_sinusoid = sind(0:(length(u)-1)); %nilai random sinusoid

y_sinusoid = 0;
for i = 2:length(u_sinusoid)
    if i == 1 then
        y_sinusoid(i)=0;
    end
y_sinusoid(i)=(1/(1+y_sinusoid(i-1)^2))+ 0.250*u_sinusoid(i)- 0.300*u_sinusoid(i-1);
end

%% ANN Input sinusoid
input = [u_sinusoid(1:2100)' u_sinusoid(2:2101)' y_sinusoid(4:2103)' y_sinusoid(3:2102)'];
target = u_sinusoid(3:2102)';

[input_row,input_col] = size(input);
[target_row,target_col] = size(target);
n_train = input_row;

%% Normalisasi
input = normalize(input);
%% Hyperparameter & Declare Variable

%Hyperparameter
x_size = input_col; %Input layer 
z_size = 7;  %Hidden layer
y_size = target_col; %Output layer

alpha = 0.1;
epoch = 2000;
miu   = 0.3;

%Variable
stop = 0;
error_target = 0.00000001;
data_count = input_row;
epoch_count = 1;

delta_zy_old = 0;
delta_zy_bias_old = 0;
delta_xz_old = 0;
delta_xz_bias_old = 0;

%% Nguyen Widrow Initialization
%Random Init
rng(0) %Seed 
epsilon_init = 0.5; %Range random number

beta = 0.7 * z_size^(1/x_size);
weight_xz = rand(x_size, z_size) - epsilon_init;
weight_zy = rand(z_size, y_size) - epsilon_init;

for i = 1:z_size
    norma(i) = sqrt(sum(weight_xz(:,i).^2));
    weight_xz(:,i) = beta*((weight_xz(:,i))/norma(i));
end

bias_xz = (rand(1, z_size) - epsilon_init) * beta;
bias_zy = rand(1, y_size) - epsilon_init;

%% Backpropagation
while stop == 0 && epoch_count <= epoch
    for n = 1:data_count  
        %Forward Pass
        %Input -> Hidden
        xi = input(n,:);
        ti = target(n,:);
        
        z_in = bias_xz + xi*weight_xz;
        
        for m=1:z_size
            z(1,m) = 1/(1+exp(-z_in(1,m)));
        end
        
        %Hidden -> Output
        y_in = bias_zy + z*weight_zy;
        
        for l=1:y_size
           yk(n,l) = y_in(1,l);
        end
        
        %Sigmoid
        %for l=1:y_size
        %   yk(n,l) = 1/(1+exp(-y_in(1,l)));
        %end
     
        %Backward Pass
        
        %Momentum calculation
        momentum_zy = miu*delta_zy_old;
        momentum_xz = miu*delta_xz_old;
        
        %Output->Hidden
        %Sigmoid
        %do_k = (yk(n,1) - ti).*(1+yk(n,1)).*(1-yk(n,1)).*0.5;
        
        %Linear
        do_k = (yk(n,1) - ti);

             
        delta_zy = alpha.*z'.*do_k + momentum_zy;
        delta_zy_old = delta_zy;
        delta_zy_bias = alpha.*do_k + miu*delta_zy_bias_old;
        delta_zy_bias_old = delta_zy_bias;
        
        %Hidden->Input
        sigma_j = do_k * weight_zy';
        do_j = sigma_j.* z.*(1-z).*0.5;
        
        delta_xz = alpha.* xi' .* do_j + momentum_xz;
        delta_xz_old = delta_xz;
        delta_xz_bias = alpha .* do_j + miu*delta_xz_bias_old;
        delta_xz_bias_old = delta_xz_bias;
        
        %Weight Update
        weight_zy = weight_zy - delta_zy;
        weight_xz = weight_xz - delta_xz;
        bias_zy = bias_zy - delta_zy_bias;
        bias_xz = bias_xz - delta_xz_bias;
        
        %Cost function    
        error(n,1) = 0.5*sum((yk(n,1)-ti).^2);
        
        %Momentum update
        delta_zy_old = delta_zy;
        delta_xz_old = delta_xz;
    end 
    error_per_epoch(1,epoch_count) = sum(error)/input_row;
    
    if error_per_epoch(1,epoch_count) < error_target
            stop = 1;
    end
    
    epoch_count = epoch_count+1;
    
end
avgerrortrain = sum(error)/n_train;
plot(error_per_epoch);
%% ANN Input Acak (testing)

%% ANN Input sinusoid (testing)

% Data Input
input_testing = [u_sinusoid(2101:3000)' u_sinusoid(2102:3001)' y_sinusoid(2104:3003)' y_sinusoid(2103:3002)'];
target_testing = u_sinusoid(2103:3002)';
[input_row_test,input_col_test] = size(input_testing);
[target_row_test,target_col_test] = size(target_testing);
n_test = input_row_test;
%% Normalisasi
input = normalize(input);

%% Hyperparameter & Declare Variable

%Variable
data_count_test = input_row_test;
true_count = 0;

for n = 1:data_count_test  
    %Forward Pass
    %Input -> Hidden
    x_test = input_testing(n,:);
    t_test = target_testing(n,:);

    z_in_test = bias_xz + x_test*weight_xz;

    for m=1:z_size
        z_test(1,m) = 1/(1+exp(-z_in_test(1,m)));
    end

    %Hidden -> Output
    y_in_test = bias_zy + z_test*weight_zy;

%     %Sigmoid
%     for l=1:y_size
%         y_test(n,l) = 1/(1+exp(-y_in_test(1,l)));
%     end
    
    %Linear
    for l=1:y_size
        y_test(n,l) = y_in_test(1,l);
    end

end

%% Plotting
error_test(1,n) = 0.5*sum((y_test(n,1) - t_test).^2);    
avgerrortest = sum(error_test)/n_test;
MSE_test = (sum((t_test-y_test).^2))/n_test;
disp("Error average test = "+ avgerrortest *100 +" %");
disp("MSE testing = "+ MSE_test*100 +" %"); 

figure;
plot(y_test,'o');
hold on
plot(target_testing,'x');
xlim([0 n_test]); xlabel('k'); ylabel('y(k)'); legend('Output ANN','Output Plant');
hold off