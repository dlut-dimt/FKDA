function [new_X,K,N,new_Ytr,c] = Inc_KFDA(new_X,K,N,a,Ytr)
% this function calculates the updated data matrix new_X, and the updated kernel matrix K etc
y = unique(a.train.y,'stable');
new_Ytr = Ytr;

%% ==================== update new_X and K =======================
for i = 1:size(y,2)
    % update for known classes %
    loc1 = find(a.train.y==y(i));
    if ismember(y(i),Ytr)
        loc = find(Ytr == y(i));
        new_X = [new_X(:,1:sum(N(1:loc))),a.train.X(:,loc1),new_X(:,sum(N(1:loc))+1:sum(N))]; 
        K = [K(1:sum(N(1:loc)),1:sum(N(1:loc))),zeros(sum(N(1:loc)),length(loc1)),K(1:sum(N(1:loc)),sum(N(1:loc))+1:sum(N));
        zeros(length(loc1),sum(N)+length(loc1));
        
        K(sum(N(1:loc))+1:sum(N),1:sum(N(1:loc))),zeros(sum(N)-sum(N(1:loc)),length(loc1)),K(sum(N(1:loc))+1:sum(N),sum(N(1:loc))+1:sum(N))];
        K(:,(sum(N(1:loc))+1):(sum(N(1:loc))+length(loc1))) = Gaussian(new_X,new_X(:,(sum(N(1:loc))+1):(sum(N(1:loc))+length(loc1))));
        K((sum(N(1:loc))+1):(sum(N(1:loc))+length(loc1)),:) = K(:,(sum(N(1:loc))+1):(sum(N(1:loc))+length(loc1)))' ;
        N(loc)=N(loc)+length(loc1);

    else
        % update for novel classes %
        new_X = [new_X,a.train.X(:,loc1)];
        K = [K,zeros(size(K,1),length(loc1));
        zeros(length(loc1),size(K,2)+length(loc1))];
        K(:,(sum(N)+1):(sum(N)+length(loc1))) = Gaussian(new_X,new_X(:,(sum(N)+1):(sum(N)+length(loc1))));
        K((sum(N)+1):(sum(N)+length(loc1)),:) = K(:,(sum(N)+1):(sum(N)+length(loc1)))';
        N = [N,length(loc1)];
        new_Ytr = [new_Ytr,y(i)];
    end
end
c=length(N);
end

