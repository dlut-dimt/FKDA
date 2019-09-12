function K = KGaussian(A)
% for building kernel data matrix, reduced or full, with Gaussian kernel. 
[gamma,r] = size(A);K1=zeros(r,r);
for i = 1: r 
    for j = i: r
      dis = A(:,i)-A(:,j); 
      K1(i,j) = exp( -(   norm( dis)^2/ (1) )); 
    end 
end
K = K1'+K1-eye(r);
end