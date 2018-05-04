function R = recurrence_plot(k, Xd)
    N = size(Xd, 2);
    
    K = round(k * N);
    tR = zeros(N);
    R = zeros(N);
    for i = 1:N
        tn = knnsearch(Xd(:,1:N)',Xd(:,i)','k',K);
        tR(i, tn) = 1;
    end
    for i = 1:N
        for j = 1:N
            if(tR(i, j) == 1 && tR(j, i) ==1)
                R(i, j) = 1;
            end
        end
    end
    
end

