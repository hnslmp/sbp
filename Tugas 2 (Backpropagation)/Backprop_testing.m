
%% Data Input
load('data.mat')
feature = dataset(7001:10000,1:12);
feature = feature{:,:};
class = dataset(7001:10000,13);
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

%Variable
data_count = feature_row;
true_count = 0;

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
    
    %Recognition Rate
    [val, idx] = max(y(n,:));
    y(n,:) = zeros(size(y(n,:)));
    y(n,idx) = 1;
    
    if y(n,:) == t
       true_count = true_count + 1;
    end
   
end

recog_rate = true_count/data_count*100;
disp(recog_rate);