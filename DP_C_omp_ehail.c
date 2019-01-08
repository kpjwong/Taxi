/* DP_mex.C solves the DP given a vector of belief, ouputs the choice probability
Syntax: P = DP_mex(lambda)*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "mex.h"
#include "float.h"
#include <omp.h>

int td(int, int, int, int);
int next_s(int, int, int, int);

// [V, P, W, Ve, Pe, We, p_take1, p_take2] = DP_C_omp_gen(ll,x,Q,pi,tz,beta,sigma,TT,ta,fuel_cost*oil_price);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 

{	
    if ( nrhs != 14 ) mexErrMsgTxt("Wrong number of Input Arguments.");
    #define V_OUT           plhs[0]
	#define P_OUT           plhs[1]
	#define W_OUT           plhs[2]
    #define Ve_OUT          plhs[3]
    #define Pe_OUT          plhs[4]
    #define We_OUT          plhs[5]
    #define p_take1_OUT     plhs[6]
    #define p_take2_OUT     plhs[7]
	#define ll_IN           prhs[0]
	#define x_IN            prhs[1]
	#define Q_IN            prhs[2]
	#define pi_IN           prhs[3]
	#define tz_IN           prhs[4]
	#define beta_IN         prhs[5]
	#define sigma_IN        prhs[6]
    #define TT_IN           prhs[7]
    #define ta_IN           prhs[8]
    #define fuel_cost_IN    prhs[9]
    #define mu_IN           prhs[10]
    #define kappa_IN        prhs[11]
    #define pp_IN           prhs[12] // pp(s,1) = P(same|s), pp(s,2) = P(cancel|s)
    #define e_sigma_IN      prhs[13]

	double *ll, *x, *Q, *pi, *tz, *P, *V, *W, *Pe, *Ve, *We, beta, sigma, *mu, kappa, *pp, e_sigma, *p_take1, *p_take2;
	ll = mxGetPr(ll_IN);
    mu = mxGetPr(mu_IN);
	x = mxGetPr(x_IN);    
	Q = mxGetPr(Q_IN);
	pi = mxGetPr(pi_IN);
	tz = mxGetPr(tz_IN);
	beta = mxGetScalar(beta_IN);
	sigma = mxGetScalar(sigma_IN);
    kappa = mxGetScalar(kappa_IN);
    pp = mxGetPr(pp_IN);
    e_sigma = mxGetScalar(e_sigma_IN);
    const int TT = mxGetScalar(TT_IN);
    const int Z = mxGetM(tz_IN);
    const int D = mxGetN(tz_IN); // D = number of non-resting moves
    int S = mxGetM(ll_IN);
    int T = S/Z;
    int S_pi = mxGetM(pi_IN);
    int S_Q = mxGetM(Q_IN);
    int S_x = mxGetM(x_IN);
    double *fuel_cost = mxGetPr(fuel_cost_IN);
        
    if( S_pi != S) mexErrMsgTxt("Inconsistent state spaces-PI");
    if( S_Q != S) mexErrMsgTxt("Inconsistent state spaces-Q");
    if( S_x != S) mexErrMsgTxt("Inconsistent state spaces-x");
	mwSize dims_PW[3] = { S, TT, D+1 };
	mwSize dims_V[2] = { S, TT };
	P_OUT = mxCreateNumericArray(3, dims_PW, mxDOUBLE_CLASS, mxREAL);
	V_OUT = mxCreateNumericArray(2, dims_V, mxDOUBLE_CLASS, mxREAL);
	W_OUT = mxCreateNumericArray(3, dims_PW, mxDOUBLE_CLASS, mxREAL);
    Pe_OUT = mxCreateNumericArray(3, dims_PW, mxDOUBLE_CLASS, mxREAL);
	Ve_OUT = mxCreateNumericArray(2, dims_V, mxDOUBLE_CLASS, mxREAL);
	We_OUT = mxCreateNumericArray(3, dims_PW, mxDOUBLE_CLASS, mxREAL);
	p_take1_OUT = mxCreateNumericArray(3, dims_PW, mxDOUBLE_CLASS, mxREAL); 
    p_take2_OUT = mxCreateNumericArray(3, dims_PW, mxDOUBLE_CLASS, mxREAL); 
    
    P = mxGetPr(P_OUT);
	V = mxGetPr(V_OUT);
	W = mxGetPr(W_OUT);
    Pe = mxGetPr(Pe_OUT);
    Ve = mxGetPr(Ve_OUT);
    We = mxGetPr(We_OUT); 
    p_take1 = mxGetPr(p_take1_OUT);
    p_take2 = mxGetPr(p_take2_OUT);
    int *ta = mxGetPr(ta_IN);
    double *p1 = malloc (sizeof (double) * S);
    double *pe = malloc (sizeof (double) * S);
    for (int i = 0; i < S; i++) {
		p1[i] = 1.0 - exp(-ll[i] * x[i]);
        pe[i] = mu[i] * (1 - pow((1 - pp[i] * (1-pp[i+S])), kappa));
	}
    
    omp_set_num_threads(80);
    int count_bad = 0;        
    
    // main routine
	for (int tau = 1; tau < TT; tau++)
	{
        int t;
        #pragma omp parallel for
		for (t = 0; t < T; t++)
		{
            // routine for non-airport starting points
            int z, s1, V_idx2; double pe1;
            #pragma omp parallel for private(s1, V_idx2, pe1)
			for (z = 0; z < Z; z++)
			{
                s1 = Z * t + z + 1;
                pe1 = pe[s1-1] * x[s1-1]/10;
                P[ (s1-1) + D * (S * TT) ] = 1;
                Pe[ (s1-1) + D * (S * TT) ] = 1;
                V[ (s1-1) + S * 0 ] = 0;
                Ve[ (s1-1) + S * 0 ] = 0;
                V_idx2 = (next_s(s1, 1, T, Z)-1) + S * (tau - 1);
                int d, z2, PW_idx, s2, V_idx2_s2, d_len, disc; double p2, pe2;
                #pragma omp parallel for private(z2, PW_idx, s2, V_idx2_s2, p2, pe2, d_len, disc) 
				for (d = 0; d < D; d++)
				{
                    PW_idx = (s1-1) + tau * S + d * (S * TT);
                    d_len = ta[z + d * Z + t/6 * D * Z];
                    z2 = tz[z + Z * d];
                    if ((z2 > 0) && (d_len < tau))
                    {
                        s2 = next_s(s1-(z+1)+z2, d_len, T, Z); // z is in index form
                        if (d_len==0) {
                            p2 = (1 - p1[s1-1])*(1 - exp(-ll[s2-1] * (10 - x[s1-1]))); 
                            pe2 = (10 - x[s1- 1]) / 10 * pe[s2-1];
                            V_idx2_s2 = next_s(s2, 1, T, Z) + S * (tau - 1); 
                            disc = 1;
                        } else {
                            p2 = 0; 
                            pe2 = 0;
                            V_idx2_s2 = s2 + S * (tau - d_len); 
                            disc = d_len;
                        }
                        int ss, td1, V_idx, tr_idx, tr_idx_s2; 
                        double We_take_2 = 0;
                        double We_not_take_2 = 0;
                        #pragma omp parallel for private (td1, V_idx, tr_idx, tr_idx_s2), shared (We_take_2, We_not_take_2)
                        for (ss = 0; ss < S; ss++)
                        {
                            td1 = td(s1, ss+1, T, Z); 
                            V_idx = ss + S * (tau - td1);
                            tr_idx = (s1-1) + S * ss;
                            tr_idx_s2 = (s2-1) + S * ss;

                            if (td1 <= tau && td1 > 0)
                            {
                                W[PW_idx] += Q[tr_idx] * (p1[s1-1] * (pi[tr_idx] + pow(beta, td1) * V[V_idx] - fuel_cost[tr_idx]) - x[s1-1] / 10 * fuel_cost[s1-1+S*(s1-1)]);
                                W[PW_idx] += Q[tr_idx_s2] * (p2 * (pi[tr_idx_s2] + pow(beta, td1) * V[V_idx] - fuel_cost[tr_idx_s2]) - (10-x[s1-1]) / 10 * fuel_cost[z2+Z*t-1+S*(z2+Z*t-1)]); 
                                W[PW_idx] += Q[tr_idx_s2] * ((1 - p1[s1-1] - p2) * pow(beta, disc) * V[V_idx2_s2] - fuel_cost[z2+Z*t-1+S*(s2-1)]); 
                                We_not_take_2 += Q[tr_idx_s2] * ((1 - exp(-ll[s2-1] * (10 - x[s1-1]))) * (pi[tr_idx_s2] + pow(beta, td1) * (Ve[V_idx] - fuel_cost[tr_idx_s2])) - (10-x[s1-1]) / 10 * fuel_cost[z2+Z*t-1+S*(z2+Z*t-1)]);
                                We_not_take_2 += Q[tr_idx_s2] * exp(-ll[s2-1] * (10 - x[s1-1])) * pow(beta, disc) * (Ve[V_idx2_s2] - fuel_cost[z2+Z*t-1+S*(s2-1)]);
                                if (z!=19) {We_take_2 += Q[tr_idx_s2] * (pi[tr_idx_s2] + pow(beta, td1) * Ve[V_idx] - fuel_cost[tr_idx_s2] - (10-x[s1-1]) / 10 * fuel_cost[z2+Z*t-1+S*(z2+Z*t-1)]);}
                            } 
//                             if (We_not_take_2!=We_not_take_2 && count_bad==0) {
//                                 count_bad+=1; 
//                                 mexPrintf("%f*(1-exp(%f*(10-%f)))*(%f+%f*(%f-%f))-%f/10*%f\n", Q[tr_idx_s2], -ll[s2-1], x[s1-1], pi[tr_idx_s2], pow(beta, td1), Ve[V_idx], fuel_cost[tr_idx_s2], (10-x[s1-1]), fuel_cost[z2+Z*t-1+S*(z2+Z*t-1)]);
//                             }
                            else if ((td1 > tau) && (td1 > 0)) {
                                if (z!=19) {
                                    We_take_2 += Q[tr_idx_s2] * (pi[tr_idx_s2] + pow(beta, td1) * (-30 - fuel_cost[tr_idx_s2]) - (10-x[s1-1]) / 10 * fuel_cost[z2+Z*t-1+S*(z2+Z*t-1)]);
                                }
                            }
                        }
                        P[PW_idx] = exp(W[PW_idx] / sigma);
                        if (z==19) {p_take2[PW_idx] = 0;} else {p_take2[PW_idx] = exp(We_take_2 / e_sigma) / (exp(We_take_2 / e_sigma) + exp(We_not_take_2 / e_sigma));}                      
                        
                        double We_take_1 = 0.0;
                        double We_not_take_1 = 0.0;
                        double log_sum;
                        
                        if (We_take_2 > We_not_take_2) {
                            log_sum = We_take_2 + e_sigma * log( exp((We_not_take_2-We_take_2)/e_sigma) + exp(0)/e_sigma );
                        } else {
                            log_sum = We_not_take_2 + e_sigma * log( exp(We_take_2-We_not_take_2)/e_sigma + exp(0)/e_sigma );
                        }
                        
                        int err_ind;
                        #pragma omp parallel for private (td1, V_idx, tr_idx, tr_idx_s2, err_ind), shared( We_take_1, We_not_take_1, log_sum)
                        for (ss = 0; ss < S; ss++)
                        {
                            td1 = td(s1, ss+1, T, Z);
                            V_idx = ss + S * (tau - td1);
                            tr_idx = (s1-1) + S * ss;
                            tr_idx_s2 = (s2-1) + S * ss;
                            
                            if (td1 <= tau && td1 > 0)
                            {
                                We_not_take_1 += Q[tr_idx] * (p1[s1-1] * (pi[tr_idx] + pow(beta, td1) * Ve[V_idx] - fuel_cost[tr_idx]) - x[s1-1] / 10 * fuel_cost[s1-1+S*(s1-1)]);
                                //We_not_take_1 += Q[tr_idx] * (1-p1[s1-1]) * (pe2 * p_take2[PW_idx] * We_take_2);
                                //We_not_take_1 += Q[tr_idx] * (1-p1[s1-1]) * (1 - pe2 * p_take2[PW_idx]) * We_not_take_2;
                                We_not_take_1 += Q[tr_idx] * (1-p1[s1-1]) * pe2 * log_sum;
                                We_not_take_1 += Q[tr_idx] * (1-p1[s1-1]) * (1 - pe2) * We_not_take_2;
                                if (z!=19) {We_take_1 += Q[tr_idx] * (pi[tr_idx] + pow(beta, td1) * Ve[V_idx] - fuel_cost[tr_idx] - x[s1-1] / 10 * fuel_cost[s1-1+S*(s1-1)]);}
                            }
                            else if (td1 > tau && td1 > 0) {
                                if (z!=19) {We_take_1 += Q[tr_idx] * (pi[tr_idx] - pow(beta, td1) * 30 - fuel_cost[tr_idx] - x[s1-1] / 10 * fuel_cost[s1-1+S*(s1-1)]);}
                            }
                        }
                        //if (s1==1627 && tau==60) {mexPrintf("We_take_2 = %f, We_not_take_2 = %f, We_take_1 = %f, We_not_take_1 = %f\n", We_take_2, We_not_take_2, We_take_1, We_not_take_1);}
                        p_take1[PW_idx] = exp(We_take_1/e_sigma) / (exp(We_take_1/e_sigma) + exp(We_not_take_1/e_sigma));
                        //if (tau >= 60) {mexPrintf("s = %d, tau = %d, d = %d, We_take_1 = %f, We_not_take_1 = %f, p_take1 = %f\n", s1, tau, d, We_take_1, We_not_take_1, p_take1[PW_idx]);}
                        if (z==19) {p_take1[PW_idx] = 0;}
                        //We[PW_idx] = pe1 * p_take1[PW_idx] * We_take_1 + (1-pe1*p_take1[PW_idx]) * We_not_take_1;
                        We[PW_idx] = pe1 * (We_take_1 + e_sigma * log(exp((We_take_1-We_not_take_1)/e_sigma) + exp(0/e_sigma))) + (1-pe1) * We_not_take_1;
                        Pe[PW_idx] = exp(We[PW_idx] / sigma);
                    }
                    else
                    {
                        P[PW_idx] = 0;
                        Pe[PW_idx] = 0;
                        W[PW_idx] = -10;
                        We[PW_idx] = -10;
                    }
                }
                // routine for rest
                W[(s1-1) + tau * S + D * S * TT] = beta * V[(next_s(s1, 1, T, Z)-1) + S * (tau - 1)];
                We[(s1-1) + tau * S + D * S * TT] = beta * Ve[(next_s(s1, 1, T, Z)-1) + S * (tau - 1)];
				P[(s1-1) + tau * S + D * S * TT] = exp(W[(s1-1) + tau * S + D * S * TT] / sigma);
                Pe[(s1-1) + tau * S + D * S * TT] = exp(We[(s1-1) + tau * S + D * S * TT] / sigma);
				
                // aggregation
                double total_p = 0.0; double total_pe = 0.0;
				for (int d = 0; d < D+1; d++)
				{
					total_p += P[(s1-1) + tau * S + d * S * TT];
                    total_pe += Pe[(s1-1) + tau * S + d * S * TT];
				}
                V[(s1-1) + tau * S] = sigma * log(total_p);
                Ve[(s1-1) + tau * S] = sigma * log(total_pe);
                //if ((Ve[(s1-1) + tau * S]!=Ve[(s1-1) + tau * S]) && (count_bad==0)) {count_bad += 1; mexPrintf("s = %d, tau = %d\n", s1, tau);}
				for (int d = 0; d < D+1; d++)
				{
					P[(s1-1) + tau * S + d * S * TT] /= total_p;
                    Pe[(s1-1) + tau * S + d * S * TT] /= total_pe;
					//V[(s1-1) + tau * S] += P[(s1-1) + tau * S + d * S * TT] * W[(s1-1) + tau * S + d * S * TT];
                    //Ve[(s1-1) + tau * S] += Pe[(s1-1) + tau * S + d * S * TT] * We[(s1-1) + tau * S + d * S * TT];
				}
			} 
		}
	}
	return;
}



int td(int s1, int s2, int T, int Z)
{
	int t1 = (s1 + Z - 1) / Z;
	int t2 = (s2 + Z - 1) / Z;
	int d;
	if (t1 > t2)
	{
		d = t2 - t1 + T;
	}
	else
	{
		d = t2 - t1;
	}
	return d;
}

int next_s(int s, int step, int T, int Z)
{
    // s, ss in real form
	int ss, t2;
    int t = (s + Z - 1) / Z; // t in real form
    int z = s - (t - 1) * Z;
    t2 = t + step;
    if (t2 > T)
	{
		t2 -= T;
	}
	ss = Z * (t2 - 1) + z;
    return ss;
}