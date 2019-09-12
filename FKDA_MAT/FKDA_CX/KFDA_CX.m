function [P,M,N] = KFDA_CX(X,XLabel,CL)
[d,n] = size(X); % the dimension and number of all samples
CLabel = unique(XLabel,'stable');
c = length(CLabel); % the number of all sample classes
M = zeros(1,c);
P = [];
N = [];
MM = [];
k = 1;
for i = 1:c
    loc = [];
    loc = find(XLabel==CLabel(i));
    v = [];
    v = X(:,loc);
    for j = 1:CL
        MM(1,j) = length(loc);
    end
    N = cat(2,N,MM);
    [Cdix,C] = kmeans(v',CL); 
    M(k) = CL;
    P = cat(1,P,C);
    k = k + 1;
end
P = P';
end

