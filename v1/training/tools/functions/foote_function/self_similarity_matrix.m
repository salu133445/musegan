function D = self_similarity_matrix(feature)
    [~, n] = size(feature);
    D = zeros(n);
    for i =1:n
       for j = i:n
           vi = feature(:,i);
           vj = feature(:,j);
           D(i, j) = sum(vi .* vj) / ( sqrt(sum(vi.^2)) * sqrt(sum(vj.^2))  );
           D(j, i) = D(i, j);
       end
    end
end

