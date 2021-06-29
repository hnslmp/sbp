
%% Data Input
load('data.mat')
feature_test = dataset(7001:10000,1:12);
feature_test = feature_test{:,:};
class_test = dataset(7001:10000,13);
class_test = class_test{:,:};

[feature_row_test,feature_col_test] = size(feature_test);
[class_row_test,class_col_test] = size(class_test);

%% Normalisasi
% %Tanpa Normalisasi
% feature_norm_test = feature_test;

%Normalisasi Min-Max
feature_norm_test = zeros(size(feature_test));
for m = 1 : feature_row_test
    for n_test = 1 : feature_col_test
       feature_norm_test(m,n_test) = ((feature_test(m,n_test) - min(feature_test(:,n_test)))/(max(feature_test(:,n_test)) - min(feature_test(:,n_test))));
    end
end

% %Normalisasi Z-Score
% feature_norm_test = zscore(feature_test,[],1);
%% Hyperparameter & Declare Variable

%Variable
data_count_test = feature_row_test;
true_count_test = 0;

for n_test = 1:data_count_test
    %Forward Pass
    %Input -> Hidden
    x_test = feature_norm_test(n_test,:);
    t_test = class_test(n_test);

    z_in_test = bias_xz + x_test*weight_xz;

    for m=1:z_size
        z_test(1,m) = 1/(1+exp(-z_in_test(1,m)));
    end

    %Hidden -> Output
    y_in_test = bias_zy + z_test*weight_zy;

    for l=1:y_size
        y_test(n_test,l) = 1/(1+exp(-y_in_test(1,l)));
    end
    
    %Threshold
    for s=1:1
        if y_test(n_test,s) >= 0.5
            y_test(n_test,s) = 1;
        end
        if y_test(n_test,s) < 0.5
            y_test(n_test,s) = 0;
        end 
    end
    
    %Cost function    
    error_test(1,n_test) = 0.5*((t_test-y_test(n_test,:)).^2);
    
    %Recognition Rate
    if y_test(n_test,:) == t_test
       true_count_test = true_count_test + 1;
    end
   
end

error_test_per_epoch(1,epoch_count) = sum(error_test)/feature_row_test;

fprintf('TEST SCORE \n');
mse_testing = sum(error_test)/feature_row_test;
fprintf('MSE Testing        :  %.4f \n', mse_testing);
acc_testing = true_count_test/data_count_test*100;
fprintf('Accuracy Testing   :  %.2f \n', acc_testing);