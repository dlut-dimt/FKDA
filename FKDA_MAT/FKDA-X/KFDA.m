function [C,N,CLabel,c] = KFDA( X,XLabel )
[d,n] = size(X);
CLabel = unique(XLabel,'stable');
c = length(CLabel);  % the number of all sample classes  
N = zeros(1,c);
C = zeros(d,c);
k = 1;
for i = 1:c
    loc = [];
    loc = find(XLabel==CLabel(i));
    N(k) = length(loc);
    C(:,k) = mean(X(:,loc),2);
    k = k+1;
end
end

