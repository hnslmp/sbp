dataset = readtable('dataset.csv');
feature = dataset(:,1:5);
feature = feature{:,:};
class = dataset(:,6:8);
class = class{:,:};


[feature_row,feature_col] = size(feature);
[class_row,class_col] = size(class);
feature_norm = zeros(size(feature));

for m = 1 : feature_row
    for n = 1 : feature_col
       feature_norm(m,n) = ((feature(m,n) - min(feature(:,n)))/(max(feature(:,n)) - min(feature(:,n))));
    end
end


%Layer Size
x_size  = 5; %Input layer 
z_size = 5;  %Hidden layer
y_size = 3; %Output layer

alpha = 0.1;
epoch = 300;


%Random Init
epsilon_init = sqrt(6)/sqrt(x_size+y_size);
weight_xz = rand(z_size, 1 + x_size) * 2 * epsilon_init - epsilon_init; 
weight_zy = rand(y_size, 1 + z_size) * 2 * epsilon_init - epsilon_init;

data_count = feature_row;
epoch_count = 1;

x = feature_norm';
m = feature_row;
t = class;


%Input layer
x_in = [ones(1, m); x];
 
%Hidden layer
z_in = weight_xz * x_in; 
z_m = sigmoid(z_in)

%Output layer
y_in = [ones(1, m); z_m];
y = sigmoid(weight_zy * y_in);
        
%Error Signal
do_k = (t'-y) .* (y.*(1-y));

delta_w_neuron = (alpha .* do_k * z_m');
delta_w_bias = (alpha * sum(do_k,2));

delta_w = [delta_w_neuron delta_w_bias]

sigma_j = do_k * weight_zy';
do_j = sigma_j .* (z_in.*(1-z_in))

delta_j_neuron = (alpha .* do_j * z_m');
delta_j_bias = (alpha * sum(do_j,2));

delta_j = [delta_j_neuron delta_j_bias]





