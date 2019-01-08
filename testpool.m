C = 60;
D = 80;
Q = rand(1,50);
Q = Q/sum(Q);
pm = .3;
w = 1;
K = 5;

psi = dot(Q,Q)*(1-pm^K)*w;
mu = min(D/((1+psi)*C),1);

N = 1000;
m = zeros(N,1);
for n = 1:N
    for c = 1:C
        ride = ((rand() < mu*(1-pm^K)));
        if ride
            m(n) = m(n) + 1;
            dest = sum(rand() > cumsum(Q)) + 1;
            pool = ((rand() < Q(dest)*(1-pm^K)*w));
            if pool
                m(n) = m(n) + 1;
            end
        end
    end
end
       
mean(m)
(1+psi)*(1-pm^K)*min(D/(1+psi),C)
