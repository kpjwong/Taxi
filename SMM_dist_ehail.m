function [D,g,SMM_struct] = SMM_dist_ehail(param, C, true_mom, Q, PI, U, eta, Perm, flag, print_flag)
 
cd C:\Users\jerem\Documents\MATLAB\ehail_model

load ehail_SMM.mat

beta = 1;
e = 1;
i = 0;
K = size(Perm,1)/4;
LL = [ll];
MU = [mu];
N_S = size(Q,1); % number of states
PP = zeros(N_S,73,7,4); % array that stores search probability for regular drivers
PP_e = zeros(N_S,73,7,4);
eQ = repmat(eta,1,K);

for i = 1:4
    i
    old_ll = ll; % update the belief
    old_mu = mu; % update the belief for ehail-drivers
    perm = Perm((i-1)*K+1:i*K,:);
    [V,P,W,Ve,Pe,We,p_take1,p_take2] = DP_C_omp_ehail(ll,x,Q,PI,tz,beta,param(41),int32(73),int32(ta2),fuel_cost,mu,param(42),pp,param(43));    
    PP(:,:,:,i) = P;
    PP_e(:,:,:,i) = Pe;
    [VC VC2 fare st m shift_ride e_match eC_N eC_N2 ll rej mu] = sim_C_omp_ehail(int32(C),cumsum(P,3),cumsum(Pe,3),p_take1,p_take2,cumsum(Q,2),PI,x,param(1:41),int32(5),int32(tz),int32(ta2),int32(TD),U,int32(perm),int32(map),eQ,beta,3.6*fuel_cost,Q,pp,param(42),gC_N,param(44));
    eC_N_store(:,i) = mean(eC_N(:,2:end),2);
    eC_N2_store(:,i) = mean(eC_N2(:,2:end),2);
    ll = mean(ll(:,2:end),2);
    mu = mean(mu(:,2:end),2);
    LL = [LL,ll];
    MU = [MU,mu];
    e = mean(abs(ll-old_ll));
    e_e = mean(abs(mu-old_mu));
    f = norm(ll-old_ll);
end

mom = double([m(:,2:end);e_match(:,2:end)]);
if strcmp(flag,'norm')
    W = eye(N_S);
    D = norm(mean(mom,2)-true_mom);
    g = mean(mom,2)-true_mom;
else
    g = zeros(N_S,K-1);
    for k = 1:K-1
        g(:,k) = mom(:,k+1)-true_mom;
    end
    gg = g-repmat(mean(g,2),1,K-1);
    S = zeros(N_S,N_S);
    for k = 1:K-1
        S = S+1/(K-1)*gg(:,k)*gg(:,k)';
    end
    if strcmp(flag,'W')
        W = pinv(S);
        D = mean(g,2)'*W*mean(g,2);
    end
    g = mean(g,2);
end

if print_flag==1
    figure;
    hold off
    zoneInfo = {'Bloomingdale','Upper West Side', 'Lincoln Sq', 'Clinton', 'Chelsea', 'Central Park', 'Times Square', 'Penn Station', 'Flatiron', 'E Harlem', 'Yorkville', 'Lenox Hill', 'Sutton Pl', 'Murray Hill', 'Gramercy', 'Greenwich Village', 'West Village', 'Lower Manhattan', 'non-Manhattan', 'LGA & JFK'};
    true_matches = reshape(true_mom(1:2880),20,144);
    true_ematches = reshape(true_mom(2881:5760),20,144);
    sim_matches = reshape(mean(m(:,2:end),2),20,144);
    sim_ematches = reshape(mean(e_match(:,2:end),2),20,144);
    for z = 1:20
        subplot(5,4,z);
        plot(sim_matches(z,:))
        hold on
        plot(true_matches(z,:),'r--')
        xticks([1 37 73 109 144])
        xticklabels({'12am','6am','12pm','6pm','12am'});
        title([zoneInfo{z}]);
        hold off
    end
    figure;
    hold off
    for z = 1:20
        subplot(5,4,z);
        plot(sim_ematches(z,:))
        hold on
        plot(true_ematches(z,:),'r--')
        xticks([1 37 73 109 144])
        xticklabels({'12am','6am','12pm','6pm','12am'});
        title([zoneInfo{z}]);
        hold off
    end
end

SMM_struct = struct('convg',[e,e_e],'matches',m,'lambda',ll,'norm_momdist',f,'VC',VC,'VC2',VC2,'eC_N',eC_N,'eC_N2',eC_N2,'fare',fare,'weight',W,'policy',P,'e_policy',Pe,'LL',LL,'MU',MU,'shift_ride',shift_ride,'PP',PP,'st',st,'e_match',e_match,'V',V,'Ve',Ve);
    