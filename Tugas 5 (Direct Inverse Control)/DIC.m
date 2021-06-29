load('BobotID.mat');
load('BobotINV.mat');
load('database.mat');

% Direct Inverse Control
reference = y(1:3010);
[n_in_row,n_in_col] = size(reference);
[n_out_row,n_out_col] = size(reference);
hidden_n = 7
for n = 1:n_in_row
    % Inverse System
    if n == 1
        in_inv = [0 reference(n,1) 0 0];
    elseif n == 2
        in_inv = [y_aktif1_inv(n-1,:) 0 reference(n,1) y_aktif_id(n-1,:)];
    else
        in_inv = [y_aktif1_inv(n-1,:) y_aktif1_inv(n-2,:) reference(n,1) y_aktif_id(n-1,:)];
    end
    
    target_test = reference(n,:);
    
    z_p_inv = bias_xz_inv + in_inv * weight_xz_inv;
    for j=1:hidden_n
        zp_aktif_inv(1,j) = tanh(z_p_inv(1,j));
    end

    %HIDDEN to OUTPUT
    y_in_inv = bias_zy_inv + zp_aktif_inv * weight_zy_inv;
    for k=1:1
        y_aktif1_inv(n,k) = y_in_inv;
    end
    
    %---------------------------------------------------------
    % Identification System
    if n == 1
        in_id = [0 0 0 y_aktif1_inv(n,:) 0];
    elseif n == 2
        in_id = [y_aktif_id(n-1,:) 0 y_aktif1_inv(n,:) y_aktif1_inv(n-1,:) 0];
    else
        in_id = [y_aktif_id(n-1,:) y_aktif_id(n-2,:) y_aktif1_inv(n,:) y_aktif1_inv(n-1,:) y_aktif1_inv(n-2,:)];
    end

    %step 3%
    z_p_id = bias_xz_id + in_id * weight_xz_id;
    
    for j=1:hidden_n
        zp_aktif_id(1,j) = tanh(z_p_id(j));
    end
    
    %HIDDEN to OUTPUT
    y_in_id = bias_zy_id + zp_aktif_id * weight_zy_id;
    for k=1:n_out_col
        y_aktif_id(n,k) = y_in_id(1,k);
    end
    
    error_test(1,n) = 0.5 .* sum((target_test-y_aktif_id(n,:)).^2);
end
error_tot_test = sum(error_test)/n_in_row;
fprintf('MSEnya adalah %d \n',error_tot_test);
figure(7)
plot(reference,'o')
hold on
plot(y_aktif_id,'p')
hold off
legend('Reference','Output');