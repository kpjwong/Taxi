cd C:\Users\jerem\Documents\MATLAB\ehail_model
load ehail_SMM.mat
load run_SMM_ehail.mat
beta = 1;
e = 1;
i = 0;
N_S = size(Q,1); % number of states

K_set = [2,5,10,20];
iter = 10;
Perm2 = zeros(200,22000);
for i = 1:200
    Perm2(i,:) = randperm(22000);
end
ep = zeros(size(K_set,2),iter);
ep_e = zeros(size(K_set,2),iter);
convg_LL = zeros(2880, iter+1, size(K_set,2));

for n = 1:size(K_set,2)
    K = K_set(n);
    eQ = repmat(eta,1,K);
    ll = importdata('initial_ll.txt');
    mu = importdata('initial_mu.txt');
    LL = [ll];
    MU = [mu];
    PP = zeros(N_S,73,7,4); % array that stores search probability for regular drivers
    PP_e = zeros(N_S,73,7,4);
for i = 1:iter
    fprintf('K = %i, i = %i\n', K, i);
    old_ll = ll; % update the belief
    old_mu = mu; % update the belief for ehail-drivers
    perm = Perm2((i-1)*K+1:i*K,:);
    [V,P,W,Ve,Pe,We,p_take1,p_take2] = DP_C_omp_ehail(ll,x,Q,PI2,tz,beta,ehail_param(41),int32(73),int32(ta2),fuel_cost,mu,ehail_param(42),pp,ehail_param(43));    
    [VC VC2 fare st m shift_ride e_match eC_N eC_N2 ll rej mu] = sim_C_omp_ehail(int32(C),cumsum(P,3),cumsum(Pe,3),p_take1,p_take2,cumsum(Q,2),PI2,x,ehail_param(1:41),int32(K),int32(tz),int32(ta2),int32(TD),U,int32(perm),int32(map),eQ,beta,3.6*fuel_cost,Q,pp,ehail_param(42),gC_N,ehail_param(44));
    ll = mean(ll(:,2:end),2);
    mu = mean(mu(:,2:end),2);
    LL = [LL,ll];
    MU = [MU,mu];
    ep(n,i) = mean(abs(ll-old_ll));
    ep_e(n,i) = mean(abs(mu-old_mu));
end
end

dlmwrite('convg.txt',ep);
dlmwrite('convg_e.txt',ep_e);

legendInfo = {};
hold on
for n = 1:size(K_set,2)
    plot(ep(n,1:end));
    legendInfo{n} = ['K = ',num2str(K_set(n))];
end
legend(legendInfo);

legendInfo = {};
hold on
for n = 1:size(K_set,2)
    plot(ep_e(n,1:end));
    legendInfo{n} = ['K = ',num2str(K_set(n))];
end
legend(legendInfo);