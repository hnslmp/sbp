    %Threshold
    for n=1:105
        for s=1:y_size
            if y(n,s) >= 0.7
                y(n,s) = 1;
            end
            if y(n,s) <= 0.3
                y(n,s) = 0;
            end 
        end
    end
    