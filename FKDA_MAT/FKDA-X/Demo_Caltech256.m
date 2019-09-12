% ==================== FKDA_X_chunk(Demo_Caltech256) ==================== %
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
% compute the centroid matrix %
% construct the kernel matrix %
% construct the kernel vector %
tic
K = KGaussian(Xtr); % kernel matrix
K_inv = pinv(K); % the inverse matrix of the kernel matrix
K_xz = Gaussian(Xtr,Xte); % kernel vector
t_K = toc;

tic
N = zeros(1,l);
for i = 1:l
    loc = [];
    loc = find(ytr==Ytr(i));
    N(i) = length(loc);
end
E = [ones(1,N(1))];
for m = 2:length(N)
    E = [E,zeros(size(E,1),N(m));zeros(1,size(E,2)),ones(1,N(m))];
end
P = E*K_inv*K_xz; % compute the projection of the samples
P_c = eye(l);
t_batch = toc;

% test %
[predictLabel, precision,t_p,probability] = predictWrap(P_c',Ytr,P',yte);

T_p = [];
T_p = [T_p;t_p];
T_sum = [];
T_sum = [T_sum;t_K];
T_sum = [T_sum;t_batch];
Pre = [];
Pre = [Pre;precision];

%% ======================= online ============================
new_X = Xtr;

for i = 1:size(Inc,2)
    % read chunk samples %
    z = Inc{i};  
    Xte = [Xte,z.test.X];
    yte = [yte,z.test.y];

    % update the centroid matrix %
    % update the kernel matrix %
    % update the kernel vector %
    tic
    [K_xz,new_X,K,N,Ytr,l] = Inc_KFDA(Xte,K_xz,new_X,K,N,z,Ytr);
    new_K_inv = pinv(K); % compute the inverse matrix of the updated kernel matrix K.
    E = [ones(1,N(1))];
    for m = 2:length(N)
        E = [E,zeros(size(E,1),N(m));zeros(1,size(E,2)),ones(1,N(m))];
    end
    P = E*new_K_inv*K_xz; % the projection of samples in chunk
    P_c = eye(l);
    sum_t = toc;

    % test %
    [predictLabel, precision,t_p,probability] = predictWrap(P_c',Ytr,P',yte);

    T_sum = [T_sum;sum_t];
    T_p = [T_p;t_p];
    Pre = [Pre;precision];
end

%% ================== results display ========================
save(sprintf('Result/Batch_Inc/Caltech256/Caltech256_chunk.mat'),'T_sum','T_p','Pre');
clear
clc    