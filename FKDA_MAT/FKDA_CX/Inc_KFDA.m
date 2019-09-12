function [new_C,new_K_c,M,N,Ytr,ytr,c] = Inc_KFDA(new_C,K_c,M,N,a,Ytr,ytr,cl,cll)
% this function calculates the updated kernel matrix new_K_c etc
new_M = M;
old_N = N;
old_n = length(old_N);
old_ytr = ytr;
y = unique(a.train.y,'stable');
alable = [];

%% ====================== update new_C ===========================
for i = 1:size(y,2)
    % update for known classes %
    loc1 = find(a.train.y==y(i));
    new_XX = [];
    new_yy = [];
    if ismember(y(i),Ytr)
        loc = find(Ytr == y(i));
        new_MM = M;
        for t = 1:length(loc1)
            new_C = [new_C(:,1:sum(new_MM(1:loc))),a.train.X(:,loc1(t)),new_C(:,sum( new_MM(1:loc))+1:sum( new_MM))]; 
            ytr = [ytr(:,1:sum( new_MM(1:loc))),a.train.y(:,loc1(t)),ytr(:,sum( new_MM(1:loc))+1:sum( new_MM))]; 
            new_XX = [new_XX,a.train.X(:,loc1(t))];
            new_yy = [new_yy,a.train.y(:,loc1(t))];
            new_MM(loc) = new_MM(loc)+1;
        end
        new_MM = M;
        new_XX = [new_C(:,(sum(new_MM(1:loc-1))+1):(sum(new_MM(1:loc)))),new_XX];
        new_yy = [ytr(:,(sum(new_MM(1:loc-1))+1):(sum(new_MM(1:loc)))),new_yy];
        [new_XX,p,o] = KFDA_CX(new_XX,new_yy,cl);
        new_yy = kron(new_yy(1),ones(1,cl));
        new_C = [new_C(:,1:sum(M(1:loc-1))),new_XX,new_C(:,sum(M(1:loc-1))+cll+cl+1:sum(M)+cll)]; 
        ytr = [ytr(:,1:sum(M(1:loc-1))),new_yy,ytr(:,sum(M(1:loc-1))+cl+cll+1:sum(M)+cll)];
        alable = [alable,y(i)];
    else
        % update for novel classes %
        for t = 1:length(loc1)
            new_XX = [new_XX,a.train.X(:,loc1(t))];
            new_yy = [new_yy,a.train.y(:,loc1(t))];
        end
        [new_XX,p,o] = KFDA_CX(new_XX,new_yy,cl);
        N = [N,o];
        new_yy = kron(new_yy(1),ones(1,cl));
        new_C = [new_C,new_XX];
        ytr = [ytr,new_yy];
        new_M = [new_M,cl];
        Ytr = [Ytr,y(i)];
    end
end
c = length(ytr);

%% ======================= update K_c ============================
new_K_c = zeros(c,c);
new_K_c(1:old_n,1:old_n) = K_c;
% update for known classes %
for i = 1:size(y,2)
    if ismember(y(i),old_ytr)
        loc1 = find(old_ytr == y(i));
        for j = 1:old_n
            for t = 1:length(loc1) 
            new_K_c(j,loc1(t)) = Gaussian(new_C(:,j),new_C(:,loc1(t)));
            new_K_c(loc1(t),j) = new_K_c(j,loc1(t));
            end
        end
    else
        % update for novel classes %
        loc1 = find(ytr == y(i));
        for j = 1:c
            for t = 1:length(loc1) 
            new_K_c(j,loc1(t)) = Gaussian(new_C(:,j),new_C(:,loc1(t)));
            new_K_c(loc1(t),j) = new_K_c(j,loc1(t));
            end
        end
    end
end
M = new_M;
end

