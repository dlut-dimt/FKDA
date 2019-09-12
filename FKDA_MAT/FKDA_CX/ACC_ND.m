%%%%%%%计算novel detection的错误率
clear
clc;
k=10;%%%随机次数
n1=1;%%%%每个chunk中已知类的个数
n2=1;%%%%每个chunk中新类的个数
pern=50;%%%%每个类中样本的个数
    Fp=0;
    Fe=0;
    Fn=0;
    M1=0;
for i=1:k
%     load(sprintf('AwA/vgg19_nd_%d.mat',i));
     load(sprintf('MNIST_fashion/ND/0.7_6_nd_dis_%d.mat',i));
%     load(sprintf('Caltech/ND/0.6_5_nd_dis_%d.mat',i));
%     load(sprintf('Cifar100/ND/0.7nd_dis_%d.mat',i));
%     load(sprintf('AR/ND/0.715_6_nd_dis_%d.mat',i));
%     load(sprintf('ORL/ND/0.715_6_nd_dis_%d.mat',i));
    Max=max(Ytr);

    for j=1:5
        n=size(La,1)/pern;
        L1=La(find(La(1:n1*pern,j)~=L(1:n1*pern,j)));
      
        Fp=Fp+length(find(L1>Max))/pern;
        Fe=Fe+length(find(L1<Max))/pern;
        L2=length(find(La(n1*pern+1:end,j)<Max));
        Fn=Fn+L2/pern;
        M1=M1+length(find(La(1:n1*pern,j)==L(1:n1*pern,j)))/pern;
    end
end
M=Fn/(j*n2*k);
F=Fp/(j*n1*k);
E=Fe/(j*n1*k);
Err=(Fn+Fp+Fe)/(j*n*k);
% save('AWA/Err.mat','M','F','E','Err','Fp','Fe','Fn');
 save('MNIST_fashion/ND/0.7_6_Err.mat','M','F','E','Err','Fp','Fe','Fn');
% save('Caltech/ND/0.6_5_Err.mat','M','F','E','Err','Fp','Fe','Fn');
% save('Cifar100/ND/0.7Err.mat','M','F','E','Err','Fp','Fe','Fn');
% save('AR/ND/0.715_6_Err.mat','M','F','E','Err','Fp','Fe','Fn');
% save('ORL/ND/0.715_6_Err.mat','M','F','E','Err','Fp','Fe','Fn');