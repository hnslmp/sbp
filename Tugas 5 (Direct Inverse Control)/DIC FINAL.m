clear;
clc;
close all;

y(1) = 0;

%% Generate Data
rng(42,'twister');    

% Input random
u = 2*rand(3010,1) - 1;

for k = 2:length(u)
    y(k,1) = (1/(1+y(k-1,1)^2)) + 0.25*u(k) - 0.3*u(k-1);
end

%% Data Input
in = [y(2:2101) y(1:2100) u(3:2102) u(2:2101) u(1:2100)];   
target = [y(3:2102)];                                       
[in_row,in_col] = size(in);
[target_row,target_col] = size(target);
n_data = in_row;

%% Normalisasi
in = normalize(in);

target = normalize(target);

%% Hyperparameter & Variabel
n_in = in_col;          %jumlah unit input layer
n_hidden = 30;           %jumlah unit hidden layer
n_out = target_col;     %jumlah unit output layer

alpha = 0.001;
miu = 0.9;

rng(25,'twister')

%% Inisialisasi

% Input layer ke hidden layer
beta = 0.7*n_hidden^(1/n_in);
v_ij = rand(n_in,n_hidden) - 0.5;
for i = 1:n_hidden
    norma(i) = sqrt(sum(v_ij(:,i).^2));
    v_ij(:,i) = (beta*v_ij(:,i))/norma(i);
end
v_0j = (rand(1,n_hidden) - 0.5)*beta*2;

% Hidden layer ke output layer
w_jk = rand(n_hidden,n_out) - 0.5;
w_0k = rand(1,n_out) - 0.5;

%% TRAINING

maxepoch = 20000;
targeterror = 0.00001;

stop = 0;
epoch = 1;
delta_wjk_old = 0;
delta_w0k_old = 0;
delta_vij_old = 0;
delta_v0j_old = 0;

%% NN

while stop == 0 && epoch <= maxepoch
    for n=1:n_data
        % Feedforward 
        xi = in(n,:);
        ti = target(n,:);
        
        % Komputasi input layer ke hidden layer
        z_inj = xi*v_ij + v_0j;      
        for j=1:n_hidden
            zj(1,j) = (1-exp(-z_inj(1,j))) / (1+exp(-z_inj(1,j)));
        end
        
        % Komputasi hidden layer ke output layer
        y_ink = zj*w_jk + w_0k;
        for k=1:n_out
            yk(n,k) = y_ink(k);
        end
        
        % Store error
        error(n,1) = 0.5*sum((yk(n,1) - ti).^2);
        
        % Backpropagation of error %
        
        % Komputasi dari output layer ke hidden layer
        d_ok = (yk(n,1) - ti);
        delta_wjk = alpha*zj'*d_ok + miu*delta_wjk_old;
        delta_w0k = alpha*d_ok + miu*delta_w0k_old;
        delta_wjk_old = delta_wjk;
        delta_w0k_old = delta_w0k;
        
        % Komputasi dari hidden layer ke input layer
        doinj = d_ok*w_jk';
        d_oj = doinj.*(1-zj.^2)/0.5;
        delta_vij = alpha*xi'*d_oj + miu*delta_vij_old;
        delta_v0j = alpha*d_oj + miu*delta_v0j_old;
        delta_vij_old = delta_vij;
        delta_v0j_old = delta_v0j;
        
        % Memperbarui bobot dan bias
        w_jk = w_jk - delta_wjk;
        w_0k = w_0k - delta_w0k;
        v_ij = v_ij - delta_vij;
        v_0j = v_0j - delta_v0j;
    end
    errorperepoch(1,epoch) = sum(error)/n_data;
    
    if errorperepoch(1,epoch) < targeterror
        stop = 1;
    end
    
    epoch = epoch+1; 
end

%% Plot

epoch = epoch - 1;
figure;
plot(errorperepoch);
ylabel('Error per epoch'); xlabel('Epoch')
disp("Error per epoch minimum = "+ min(errorperepoch) *100 +" %"); disp("Error akhir = "+ errorperepoch(1,epoch)*100+" %");

%% Plot MSE & RMSE
MSE_train = (sum((target-yk).^2))/n_data;
RMSE_train = sqrt(MSE_train);
disp("MSE training = "+ MSE_train); disp("RMSE training = "+ RMSE_train);

figure;
plot(yk,'o');
hold on
plot(target,'x');
xlim([0 n_data]); xlabel('k'); ylabel('y(k)'); legend('Output ANN','Output Plant');
hold off

%% TESTING

% Data Input

input_test = [y(2102:3001) y(2101:3000) u(2103:3002) u(2102:3001) u(2101:3000)];  
target_test = [y(2103:3002)];                                                   
n_test = length(input_test);
input_test = normalize(input_test);
target_test = normalize(target_test);

%% NN

test_true = 0;
test_false = 0;
for n=1:n_test
    xi_test = input_test(n,:);
    ti_test = target_test(n,:);
    
    % Komputasi input layer ke hidden layer
    z_inj_test = xi_test*v_ij + v_0j;      
    for j=1:n_hidden
        zj_test(1,j) = (1-exp(-z_inj_test(1,j))) / (1+exp(-z_inj_test(1,j)));
    end
    
    % Komputasi hidden layer ke output layer
    y_ink_test = zj_test*w_jk + w_0k;
    for k=1:n_out
        yk_test(n,k) = y_ink_test(1,k);
    end
    
    % Store error
    error_test(1,n) = 0.5*sum((yk_test(n,1) - ti_test).^2);
    
end

%% Error

avg_error = sum(error_test)/n_test;
disp("Error average test = "+ avg_error *100 +" %");

n_test = length(target_test);
MSE_test = (sum((target_test-yk_test).^2))/n_test;
RMSE_test = sqrt(MSE_test);
disp("MSE testing = "+ MSE_test); disp("RMSE testing = "+ RMSE_test);

%% Plot

figure;
plot(yk_test,'o');
hold on
plot(target_test,'x');
xlim([0 n_test]); xlabel('k'); ylabel('y(k)'); legend('Output ANN','Output Plant');
hold off

clearvars -except v_0j v_ij w_0k w_jk u y n_hidden n_out

input_control = [u(2:2100) u(1:2099) y(4:2102) y(3:2101) y(2:2100)];
target_control = [u(3:2101)];                             
[input_con_row,input_con_col] = size(input_control);
[target_con_row,target_con_col] = size(target_control);
n_con_data = input_con_row;

%% normalisasi
input_control = normalize(input_control);

target_control = normalize(target_control);

%% parameter
n_in_con = input_con_col;          %jumlah unit input layer
n_hidden_con = 30;           %jumlah unit hidden layer
n_out_con = target_con_col;     %jumlah unit output layer

alpha_con = 0.001;
miu_con = 0.3;

rng(30,'twister');

%% inisialisasi bobot

% Input layer ke hidden layer
beta = 0.7*n_hidden_con^(1/n_in_con);
v_ij_con = rand(n_in_con,n_hidden_con) - 0.5;
for i = 1:n_hidden_con
    norma(i) = sqrt(sum(v_ij_con(:,i).^2));
    v_ij_con(:,i) = (beta*v_ij_con(:,i))/norma(i);
end
v_0j_con = (rand(1,n_hidden_con) - 0.5)*beta*2;

% Hidden layer ke output layer
w_jk_con = rand(n_hidden_con,n_out_con) - 0.5;
w_0k_con = rand(1,n_out_con) - 0.5;

%% Training

maxepoch_con = 2000;
targeterror_con = 0.00001;

stop = 0;
epoch = 1;
delta_con_wjk_old = 0;
delta_con_w0k_old = 0;
delta_con_vij_old = 0;
delta_con_v0j_old = 0;

while stop == 0 && epoch <= maxepoch_con
    for n=1:n_con_data
        % %
        % Feedforward 
        xi = input_control(n,:);
        ti = target_control(n,:);
        
        % Komputasi input layer ke hidden layer
        z_inj = xi*v_ij_con + v_0j_con;      
        for j=1:n_hidden_con
            %zj(1,j) = 2/(1+exp(-z_inj(1,j))) - 1; 
            zj(1,j) = (1-exp(-z_inj(1,j))) / (1+exp(-z_inj(1,j)));
        end
        
        % Komputasi hidden layer ke output layer
        y_ink = zj*w_jk_con + w_0k_con;
        for k=1:n_out_con
            yk(n,k) = y_ink(k);
        end
        
        % Store error
        error(n,1) = 0.5*sum((yk(n,1) - ti).^2);
        
        % Backpropagation of error %
        
        % Komputasi dari output layer ke hidden layer
        d_ok = (yk(n,1) - ti);
        delta_wjk = alpha_con*zj'*d_ok + miu_con*delta_con_wjk_old;
        delta_w0k = alpha_con*d_ok + miu_con*delta_con_w0k_old;
        delta_con_wjk_old = delta_wjk;
        delta_con_w0k_old = delta_w0k;
        
        % Komputasi dari hidden layer ke input layer
        doinj = d_ok*w_jk_con';
        d_oj = doinj.*(1-zj.^2)/0.5;
        delta_vij = alpha_con*xi'*d_oj + miu_con*delta_con_vij_old;
        delta_v0j = alpha_con*d_oj + miu_con*delta_con_v0j_old;
        delta_con_vij_old = delta_vij;
        delta_con_v0j_old = delta_v0j;
        
        % Memperbarui bobot dan bias
        w_jk_con = w_jk_con - delta_wjk;
        w_0k_con = w_0k_con - delta_w0k;
        v_ij_con = v_ij_con - delta_vij;
        v_0j_con = v_0j_con - delta_v0j; 
    end
    errorperepoch(1,epoch) = sum(error)/n_con_data;    
    
    if errorperepoch(1,epoch) < targeterror_con
        stop = 1;
    end
    
    epoch = epoch+1; 
end

%% plot error

epoch = epoch - 1;
figure;
plot(errorperepoch);
ylabel('Error per epoch'); xlabel('Epoch')
disp("Error per epoch minimum = "+ min(errorperepoch) *100 +" %"); disp("Error akhir = "+ errorperepoch(1,epoch)*100+" %");

MSE_train_con = (sum((target_control-yk).^2))/n_con_data;
RMSE_train_con = sqrt(MSE_train_con);
disp("MSE training = "+ MSE_train_con); disp("RMSE training = "+ RMSE_train_con);

%% plot perbandingan output plant dan ANN

figure;
plot(yk,'o');
hold on
plot(target_control,'x');
xlim([0 n_con_data]); xlabel('k'); ylabel('y(k)'); legend('Output ANN','Output Plant');
hold off

%Data Inserting

input_test_control = [u(2101:3000) u(2100:2999) y(2103:3002) y(2102:3001) y(2101:3000)];   
target_test_control = [u(2102:3001)];                                                   
n_test_con = length(input_test_control);
input_test_control = normalize(input_test_control);
target_test_control = normalize(target_test_control); 

%Feedforward
test_true = 0;
test_false = 0;
for n=1:n_test_con
    xi_test = input_test_control(n,:);
    ti_test = target_test_control(n,:);
    
    % Komputasi input layer ke hidden layer
    z_inj_con_test = xi_test*v_ij_con + v_0j_con;      
    for j=1:n_hidden_con
        zj_con_test(1,j) = (1-exp(-z_inj_con_test(1,j))) / (1+exp(-z_inj_con_test(1,j)));
    end
    
    % Komputasi hidden layer ke output layer
    y_ink_con_test = zj_con_test*w_jk_con + w_0k_con;
    for k=1:n_out_con
        yk_con_test(n,k) = y_ink_con_test(1,k);
    end
    
    % Store error
    error_test_con(1,n) = 0.5*sum((yk_con_test(n,1) - ti_test).^2);
    
end

avg_error = sum(error_test_con)/n_test_con;
disp("Error average test = "+ avg_error *100 +" %"); 

%Test MSE & RMSE
n_test_con = length(target_test_control);
MSE_test_con = (sum((target_test_control-yk_con_test).^2))/n_test_con;
RMSE_test_con = sqrt(MSE_test_con);
disp("MSE testing = "+ MSE_test_con); 
disp("RMSE testing = "+ RMSE_test_con);


%Plot actual output and ANN output
figure ();
plot(yk_con_test,'o');
hold on 
plot(target_test_control,'x');
xlim([0 n_test_con]); xlabel('k'); ylabel('y(k)'); legend('Output ANN','Output Plant');
hold off

norm_ref = y(1:3010); %r(k) = y(k)
norm_ref = normalize(norm_ref);

[n_in_row,n_in_col] = size(norm_ref);
[n_out_row,n_out_col] = size(norm_ref);

for n = 1:n_in_row
    %-------------INV CONTROL---------------
    if n==1
        in = [0 0 norm_ref(n,1) 0 0];
    elseif n==2
        in = [yk_con_inv(n-1,:) 0 norm_ref(n,1) yk_id_inv(n-1,:) 0];
    else
        in = [yk_con_inv(n-1,:) yk_con_inv(n-2,:) norm_ref(n,1) yk_id_inv(n-1,:) yk_id_inv(n-2,:)];
    end
    
    in = normalize(in);
    
    target_inv = norm_ref(n,:);
    
    %Input to Hidden
    z_inj_con_inv = in*v_ij_con + v_0j_con;
    for j=1:n_hidden_con
        %zj_test(1,j) = 2/(1+exp(-z_inj_test(1,j))) - 1;
        zj_con_inv(1,j) = (1-exp(-z_inj_con_inv(1,j))) / (1+exp(-z_inj_con_inv(1,j)));
    end
    
    %Hidden to Output
    y_ink_con_inv = zj_con_inv*w_jk_con + w_0k_con;
    for k=1:n_out_con
        %yk_test(n,k) = 2/(1+exp(-y_ink_test(1,k))) - 1;
        %yk_test(n,k) = (1-exp(-y_ink_test(1,k))) / (1+exp(-y_ink_test(1,k)));
        yk_con_inv(n,k) = y_ink_con_inv(1,k);
    end

    %-------------SYSTEM ID---------------
    
    if n==1
        %in_id = [yk_con_inv(n,:) 0 0 0 0];
        in_id = [0 0 yk_con_inv(n,:) 0 0];
    elseif n==2
        %in_id = [yk_con_inv(n,:) yk_con_inv(n-1,:) 0 yk_id_inv(n-1,:) 0];
        in_id = [yk_id_inv(n-1,:) 0 yk_con_inv(n,:) yk_con_inv(n-1,:) 0];
    else
        %in_id = [yk_con_inv(n,:) yk_con_inv(n-1,:) yk_con_inv(n-2,:) yk_id_inv(n-1,:) yk_id_inv(n-2,:)];
        in_id = [yk_id_inv(n-1,:) yk_id_inv(n-2,:) yk_con_inv(n,:) yk_con_inv(n-1,:) yk_con_inv(n-2,:)];
    end
    
    in_id = normalize(in_id);

    z_inj_id_inv = in_id*v_ij + v_0j;
    for j=1:n_hidden
        %zj_test(1,j) = 2/(1+exp(-z_inj_test(1,j))) - 1;
        zj_id_inv(1,j) = (1-exp(-z_inj_id_inv(1,j))) / (1+exp(-z_inj_id_inv(1,j)));
    end
    
    y_ink_id_inv = zj_id_inv*w_jk + w_0k;
    for k=1:n_out
        %yk_test(n,k) = 2/(1+exp(-y_ink_test(1,k))) - 1;
        %yk_test(n,k) = (1-exp(-y_ink_test(1,k))) / (1+exp(-y_ink_test(1,k)));
        yk_id_inv(n,k) = y_ink_id_inv(1,k);
    end
    
    error_inv(1,n) = 0.5*sum((target_inv - yk_id_inv(n,:)).^2);
    %--------------------
    

end

MSE_DIC = sum(error_inv)/n_in_row

figure ();
plot(yk_id_inv,'o');
hold on 
plot(norm_ref,'x');
xlabel('k'); ylabel('y(k)'); legend('Output ANN','Reference');
hold off