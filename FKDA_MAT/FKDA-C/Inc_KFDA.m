function [new_K_C,new_K_C_inv,N,new_C,new_Ytr,c] = Inc_KFDA(K_C,N,a,C,Ytr)
% this function calculates the updated kernel matrix new_K_C etc
old_N = N;
old_c = length(old_N);
new_C = C;
new_Ytr = Ytr;
y = unique(a.train.y,'stable');

%% ====================== update new_C ===========================
for i = 1:size(y,2)
    % update for known classes %
    loc1 = find(a.train.y==y(i));
    if ismember(y(i),Ytr)
        loc = find(Ytr==y(i));
        N(loc) = N(loc)+length(loc1);
        new_C(:,loc) = (old_N(loc)*new_C(:,loc)+sum(a.train.X(:,loc1),2))/N(loc);
    else
        % update for novel classes %
        N = [N,length(loc1)];
        new_C = [new_C,mean(a.train.X(:,loc1),2)];
        new_Ytr = [new_Ytr,y(i)];
    end
end
c = length(N);

%% ======================= update new_K_c ============================
new_K_C = zeros(c,c);
% update for known classes %
new_K_C(1:old_c,1:old_c) = K_C;
for i = 1:size(y,2)
    if ismember(y(i),Ytr)
        loc = find(Ytr == y(i));
        for j = 1:old_c
            new_K_C(j,loc) = Gaussian(new_C(:,j),new_C(:,loc));
            new_K_C(loc,j) = new_K_C(j,loc);
        end
    else
        % update for novel classes %
        loc = find(new_Ytr == y(i));
        for j = 1:c
            new_K_C(j,loc) = Gaussian(new_C(:,j),new_C(:,loc));
            new_K_C(loc,j) = new_K_C(j,loc);
        end
    end
end
new_K_C_inv = inv(new_K_C); % the inverse matrix for new_K_C
end