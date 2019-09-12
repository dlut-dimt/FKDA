% ==================== FKDA_C_novel(Demo_Caltech256) ==================== %
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
Xte = batch.test.X;
Ytr = unique(ytr,'stable');
yte = batch.test.y;
[d,n] = size(Xtr); % the dimension and number of all samples
l = length(Ytr); % the number of all sample classes

%% ======================= offline ===========================
% compute the centroid matrix %
% construct the kernel matrix %
% construct the kernel vector %
tic
[C_o,N] = KFDA(Xtr,ytr); % compute the centroid matrix
K_o = KGaussian(C_o); % kernel matrix
K_o_inv = inv(K_o); % the inverse matrix of the kernel matrix
K_cz = Gaussian(C_o,Xte);% kernel vector
t_K = toc;
tic
P = K_o_inv*K_cz; % compute the projection of the samples
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
new_C_o = C_o;
new_K_o = K_o;
old_Ytr = Ytr;

for i = 1:size(Inc,2)
    % read chunk samples %
    z = Inc{i};  
    alable = unique(z.train.y,'stable'); %the labels required to be updated

    % extract the samples belong to novel classes %
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

    % update the centroid matrix %
    % update the kernel matrix %
    % update the kernel vector %
    tic
    [new_K_o,new_K_o_inv,N,new_C_o,Ytr,l] = Inc_KFDA(new_K_o,N,z,new_C_o,Ytr);
    new_K_cz = Gaussian(new_C_o,z_novel); % kernel vector

    new_P = new_K_o_inv*new_K_cz;%the projection of samples in chunk
    new_P_c = eye(l);
    sum_t=toc;

    % test %
    [predictLabel,precision,t_p,probability]=predictWrap(new_P_c',Ytr,new_P',y_n);

    T_sum = [T_sum;sum_t];
    T_p = [T_p;t_p];
    Pre = [Pre;precision];
end

%% ================== results display ========================
save(sprintf('Result/Batch_Inc/Caltech256/Caltech256_novel.mat'),'T_sum','T_p','Pre');
clear
clc    