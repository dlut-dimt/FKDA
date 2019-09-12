function K= Gaussian( X,z )
% for building kernel data matrix, reduced or full, with Gaussian kernel. 
[gamma,r] = size(X);
[m,mm] = size(z); % z is the set of test samples
K = zeros(r,mm);
if gamma ~= m
    error('the dimension of input data is inconsistent!');
else
    for i = 1: r   
        for j = 1:mm
            dis = X(:,i)-z(:,j); 
            K(i,j) = exp( -( norm( dis)^2/ 1) );   
        end
    end
end
end

