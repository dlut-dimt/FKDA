function D = dis( X,Z )
[d1,n1] = size(X);
[d2,n2] = size(Z);
if d1~= d2
    error('the dimensions of two matrixs are inconsistent ');
end
D = zeros(n1,n2);
for i = 1:n1
    for j = 1:n2
        % European distance
        d = X(:,i)-Z(:,j);
        D(i,j) = norm(d,2);
    end
end
end

