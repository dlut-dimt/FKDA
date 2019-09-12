function K = ch_Gaussian(new_C,z,K_xz,alabel,Ytr )
[gamma,r] = size(new_C);
[m,mm]=size(z);% z is the set of test samples
K=zeros(r,mm);
[a,b]=size(K_xz);
if gamma ~=m
    error('the dimension of input data is inconsistent!');
else
    K(1:a,1:b)=K_xz;
    for ii=1:length(alabel)
        i=find(Ytr==alabel(ii));
        for j=1:b
            dis=new_C(:,i)-z(:,j); 
            K(i,j) = exp(-(   norm( dis )^2 / (1)) );   
        end
    end
    for i = 1: r   
        for j=(b+1):mm
            dis=new_C(:,i)-z(:,j); 
            K(i,j) = exp( -(   norm( dis )^2 / (1)) );   
        end
    end
end
end

