% =================== FKDA_CX_chunk(Demo_Caltech256) ==================== %
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
cl = 5; % the sub-class number in a class  

%% ======================= offline ===========================
% adaptive cluster %
% compute the centroid matrix %
% construct the kernel matrix %
% construct the kernel vector %
tic
e = ones(1,cl);
[C_m,M,N] = KFDA_CX(Xtr,ytr,cl); % construct the centroid matrix of micro-clusters
K_cm = KGaussian(C_m); % kernel matrix
K_cm_inv = inv(K_cm); % the inverse matrix of the kernel matrix
K_cmz = Gaussian(C_m,Xte); % kernel vector
t_K = toc;

tic
E = [];
for i = 1:l
    E = [E,zeros(size(E,1),cl);zeros(1,size(E,2)),e];
end
P = E*K_cm_inv*K_cmz; % compute the projection of the samples
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
ytr = kron(Ytr,ones(1,cl));
new_C_m = C_m;
new_K_cm = K_cm;

for i = 1:size(Inc,2)     
    % read chunk samples %
    z = Inc{i};  
    alable = unique(z.train.y,'stable');
    XXte = length(z.train.y);
    cll = XXte/length(alable);

    Xte=[Xte,z.test.X];
    yte=[yte,z.test.y];

    % update the centroid matrix %
    % update the kernel matrix %
    % update the kernel vector %
    tic
    [new_C_m,new_K_cm,M,N,Ytr,ytr,l] = Inc_KFDA(new_C_m,new_K_cm,M,N,z,Ytr,ytr,cl,cll);
    K_cmz = ch_Gaussian(new_C_m,Xte,K_cmz,alable,ytr); % kernel vector
    new_K_cm_inv = inv(new_K_cm); % compute the inverse matrix of the updated kernel matrix new_K_cm.
    E = [];
    l = l/cl;
    for ii = 1:l
        E = [E,zeros(size(E,1),cl);zeros(1,size(E,2)),e];
    end
    new_P = E*new_K_cm_inv*K_cmz; % the projection of samples in chunk
    new_P_c = eye(l);
    sum_t = toc;

    % test %
    [predictLabel, precision,t_p,probability] = predictWrap(new_P_c',Ytr,new_P',yte);

    T_sum = [T_sum;sum_t];
    T_p = [T_p;t_p];
    Pre = [Pre;precision];
end

%% ================== results display ========================
save(sprintf('Result/Batch_Inc/Caltech256/Caltech256_chunk.mat'),'T_sum','T_p','Pre');
clear
clc    
