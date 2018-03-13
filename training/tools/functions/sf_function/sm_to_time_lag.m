function L = sm_to_time_lag(R)
    N = size(R, 1);
    L = zeros(N);
    for i = 1:N
        for j = 1:N
            k = mod(i+j-2, N); 
            L(i,j) = R(i, k+1);
        end  
    end
end

