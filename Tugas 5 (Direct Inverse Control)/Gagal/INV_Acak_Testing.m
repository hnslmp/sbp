%% ANN Input Acak (testing)

% Data Input
load('INV_Acak_Training.mat');
input_testing = [u(2101:3000) u(2102:3001) y(2104:3003)' y(2103:3002)'];
target_testing = u(2103:3002);
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

u_topi_k = y_test;
save("INV_Acak_Testing.mat",'u_topi_k');
ID_Acak_Testing;