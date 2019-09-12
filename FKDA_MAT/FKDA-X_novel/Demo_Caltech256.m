% ==================== FKDA_X_novel(Demo_Caltech256) ==================== %
% "Novelty Detection and Online Learning for Chunk Data Streams"          %
% Y. Wang, Y. Ding, X. He, X. Fan, C. Lin, F. Li, T. Wang, Z. Luo, J. Luo %
% TPAMI-2019-02-0102                                                      %
% Y. Wang [dlutwangyi@dlut.edu.cn]                                        %
% Please email me if you find bugs, or have suggestions or questions!     %
% ======================================================================= %

clear
clc;
    
load(sprintf('./Data/Caltech256/Caltech256_chunk_non.mat'));

% read batch samples %
Xtr = batch.train.X;
ytr = batch.train.y;
Ytr = unique(ytr,'stable');
Xte = batch.test.X;
yte = batch.test.y;
[d,n] = size(Xtr); % the dimension and number of all samples
l = length(Ytr); % the number of all sample classes

%% ======================= offline ===========================
% construct the kernel matrix %
% construct the kernel vector %   
tic
K_x = KGaussian(Xtr); % kernel matrix
K_x_inv = pinv(K_x); % the inverse matrix of the kernel matrix
K_xz = Gaussian(Xtr,Xte); % kernel vector

N = zeros(1,l);
for i = 1:l
    loc = [];
    loc = find(ytr==Ytr(i));
    N(i) = length(loc);
end
t_K = toc;

tic
E = [ones(1,N(1))];
for m = 2:length(N)
    E = [E,zeros(size(E,1),N(m));zeros(1,size(E,2)),ones(1,N(m))];
end
P = E*K_x_inv*K_xz; % compute the projection of the samples
P_c = eye(l);
t_batch = toc;

% test %
[predictLabel,precision,t_p,probability] = predictWrap(P_c',Ytr,P',yte);

T_p = [];
T_p = [T_p;t_p];
T_sum = [];
T_sum = [T_sum;t_K];
T_sum = [T_sum;t_batch];
Pre = [];
Pre = [Pre;precision];

%% ======================= online ============================
new_X = Xtr;
old_Ytr = Ytr;
new_K_x = K_x;

for i = 1:size(Inc,2)
    % read chunk samples %
    z = Inc{i};  
    alable = unique(z.train.y,'stable');

    % extract the samples belong to novel classes %
    y = unique(z.test.y,'stable');
    loc_novelte = ismember(z.test.y,old_Ytr);
    loc_novelte =~loc_novelte;
    locnte = find(loc_novelte==1);
    yte_n = z.test.y(locnte);
    z_novelte = z.test.X(:,locnte);

    loc_noveltr = ismember(z.train.y,old_Ytr);
    loc_noveltr =~loc_noveltr;
    locntr = find(loc_noveltr==1);
    ytr_n = z.train.y(locntr);
    z_noveltr = z.train.X(:,locntr);

    z_novel = [z_noveltr,z_novelte];
    y_n = [ytr_n,yte_n];

    % update the kernel matrix %
    % update the kernel vector %
    tic
    [new_X,new_K_x,N,Ytr,l] = Inc_KFDA(new_X,new_K_x,N,z,Ytr);
    new_K_inv = pinv(new_K_x); % compute the inverse matrix of the updated kernel matrix K.
    new_K_xz = Gaussian(new_X,z_novel); % kernel vector
    E = [ones(1,N(1))];
    for m = 2:length(N)
        E = [E,zeros(size(E,1),N(m));zeros(1,size(E,2)),ones(1,N(m))];
    end
    new_P = E*new_K_inv*new_K_xz; % the projection of samples in chunk
    new_P_c = eye(l);
    sum_t = toc;

    % test %
    [predictLabel,precision,t_p,probability] = predictWrap(new_P_c',Ytr,new_P',y_n);

    T_sum = [T_sum;sum_t];
    T_p = [T_p;t_p];
    Pre = [Pre;precision];
end

%% ================== results display ========================
save(sprintf('Result/Batch_Inc/Caltech256/Caltech256_novel.mat'),'T_sum','T_p','Pre');
clear
clc   