% ============== FKDA_C_Novelty_Detection(Demo_Caltech256) ============== %
% "Novelty Detection and Online Learning for Chunk Data Streams"          %
% Y. Wang, Y. Ding, X. He, X. Fan, C. Lin, F. Li, T. Wang, Z. Luo, J. Luo %
% TPAMI-2019-02-0102                                                      %
% Y. Wang [dlutwangyi@dlut.edu.cn]                                        %
% Please email me if you find bugs, or have suggestions or questions!     %
% ======================================================================= %

clear
clc;
threshold = 0.7;
DD = [];

load(sprintf('./Data/Caltech256/Caltech256_chunk_non.mat'));

% read batch samples %
Xtr = full(batch.train.X);
ytr = batch.train.y;
Ytr = unique(ytr,'stable');
Xte = full(batch.test.X);
yte = batch.test.y;
[d,n] = size(Xtr); % the dimension and number of all samples
c = length(Ytr); % the number of all sample classes

%% ======================= offline ===========================
% adaptive cluster %
% compute the centroid matrix %
% construct the kernel matrix %
% construct the kernel vector %
tic
[C,N] = KFDA(Xtr,ytr); % construct the centroid matrix of micro-clusters
K_C = KGaussian(C); % kernel matrix 
K_C_inv = pinv(K_C); % the inverse matrix of the kernel matrix

tic
P_c = eye(c);
t_batch = toc;

%% ================== novelty detection ======================
new_X = Xtr;
new_XLabel = ytr;
La = []; % predicted labels
L = []; % labels of samples in chunk
T = []; % time of novelty detection
inc_C = [];
D_c = [];
Ind = [];
PP = [];
for i = 1:size(Inc,2)
    % read chunk samples %
    z = Inc{i}; 
    x = full(z.train.X);
    y = z.train.y;
    L = [L;y];
    new_C = C;
    new_X = [Xtr,x];
    alable = unique(y,'stable');
    % compute the number of samples in each chunk %
    N_new = [];
    for ii = 1:length(alable)
        N_new = [N_new,length(find(y == alable(ii)))];
    end
    new_XLabel = ytr;

    tic
    % map new samples to low-dimensional space %
    K_cx = Gaussian(C,x);
    P_x = K_C_inv*K_cx;
    % detect weather the class in the chunk is a new class or a known class %
    % compare samples in chunk and class centers %
    Max_label = max(new_XLabel);
    new_N = N;
    m = 0;
    k = 1; % add novel class one by one
    label = [];
    for j = 1:length(alable)
        CC = mean(P_x(:,m+1:m+N_new(j)),2);
        inc_C = [inc_C,CC];
        Dis = dis(CC,P_c);
        [d_c,index] = min(Dis); % find the nearest known class
        D_c = [D_c,d_c];
        Ind = [Ind,index];
        if d_c < threshold
            l1 = Ytr(index);
        else
            l1 = Max_label+k;
        end
        if ismember(l1,Ytr)
            new_N(index) = N(index)+N_new(j);
            new_C(:,index) = (N(index)*C(:,index)+sum(x(:,m+1:m+N_new(j)),2))/new_N(index);
        else
            new_N = [new_N,N_new(j)];
            new_C = [new_C,mean(x(:,m+1:m+N_new(j)),2)];
            k = k+1;
        end
        label = [label,repmat(l1,1,N_new(j))];  
        m = m+N_new(j);
    end
    t = toc;

    T = [T;t];
    PP = [PP,P_x];
    La = [La,label'];
end
L = L';

%% ================ calculate accuracy =======================
n1 = 5; % the number of konwn classes in each chunk
n2 = 3; % the number of new classes in each chunk
pern = 12; % the number of samples in each class
Fp = 0;
Fe = 0;
Fn = 0;
M1 = 0;
Max = max(Ytr);
for jj = 1:size(La,2)
    n = size(La,1)/pern;
    L1 = La(find(La(1:n1*pern,jj)~= L(1:n1*pern,jj)));
    Fp = Fp+length(find(L1>Max))/pern;
    Fe = Fe+length(find(L1<Max))/pern;
    L2 = length(find(La(n1*pern+1:end,jj)<Max));
    Fn = Fn+L2/pern;
    M1 = M1+length(find(La(1:n1*pern,jj) == L(1:n1*pern,jj)))/pern;
end

M = Fn/(jj*n2); % novelty-class instances falsely identified to known classes
F = Fp/(jj*n1); % known-class instances falsely identified to novel classes
E = Fe/(jj*n1); % known-class instances falsely identified to otherknown classes
Err = (Fn+Fp+Fe)/(jj*n);

%% ================== results display ========================
save(sprintf('Result/ND/Caltech256/0.7_1.mat'),'M','F','E','Err','Fp','Fe','Fn');
clear
clc    