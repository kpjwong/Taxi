#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mex.h"
#include <omp.h>

// prototype declaration
int weighted_draw(double*, int, int, int, int, int, double);
int next_s(int, int, int, int);
void F(double *ll, double *VC, double *VC2, int i, int S, double *alpha, double *beta, double *gamma, double *x, int t, int Z, int *map, double *gC_N);
void G(double* mu, double* eC_N, double* eC_N2, double *x, int i, int t, int Z, int S, double* eQ, double kappa, double *kappa_hat, double *gC_N, double gC_epct);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{    
    if (nrhs != 24)  mexErrMsgTxt("Wrong number of Input Arguments.");
    
    #define VC_OUT              plhs[0]
    #define VC2_OUT             plhs[1]
    #define fare_OUT            plhs[2]
    #define st_OUT              plhs[3]
    #define m_OUT               plhs[4]
    #define shift_ride_OUT      plhs[5]
    #define e_match_OUT         plhs[6]
    #define eC_N_OUT            plhs[7] // eC at the beginning of each period
    #define eC_N2_OUT           plhs[8] // eC after search movements
    #define ll_OUT              plhs[9]
    #define rej_OUT             plhs[10]
    #define mu_OUT              plhs[11]
    #define temp_C_OUT          plhs[12]
    #define C_IN                prhs[0]
    #define P_IN                prhs[1] // cumulative sum
    #define Pe_IN               prhs[2] // cumulative sum
    #define p_take1_IN          prhs[3] 
    #define p_take2_IN          prhs[4] 
    #define Q_IN                prhs[5] // cumulative sum
    #define pi_IN               prhs[6]
    #define x_IN                prhs[7]
    #define param_IN            prhs[8] // 41x1
    #define iter_IN             prhs[9]
    #define tz_IN               prhs[10]
    #define ta_IN               prhs[11]
    #define TD_IN               prhs[12]
    #define U_IN                prhs[13] // N x (T x 11) rand matrix
    #define perm_IN             prhs[14] // iter X N permutation matrix
    #define map_IN              prhs[15]
    #define eQ_IN               prhs[16] // e-hail request generator
    #define delta_IN            prhs[17] // discount factor
    #define fuel_cost_IN        prhs[18]
    #define tr_Q_IN             prhs[19] // Q
    #define pp_IN               prhs[20] // [p(same) p(cancel)]
    #define kappa_IN            prhs[21] // double
    #define gC_N_IN             prhs[22]
    #define gC_epct_IN          prhs[23]
        
    const int S = mxGetM(Q_IN);
    const int iter = mxGetScalar(iter_IN);
    mwSize dims1[2] = { S, iter };
    const mwSize *dims4 = mxGetDimensions(P_IN);
    const int TT = dims4[1]; // TT = max shift + 1
    mwSize dims5[2] = { iter, 1 };
    const int Z = mxGetM(tz_IN);
    const int T = S/Z;
    const int D = mxGetN(tz_IN);  // D = 6;
    const int N = mxGetM(C_IN);
    mwSize dims2[2] = { N, iter };
    mwSize dims_temp_C[2] = { N, 6 };
        
    VC_OUT = mxCreateNumericArray(2, dims1, mxDOUBLE_CLASS, mxREAL);
    VC2_OUT = mxCreateNumericArray(2, dims1, mxDOUBLE_CLASS, mxREAL);
    fare_OUT = mxCreateNumericArray(2, dims2, mxDOUBLE_CLASS, mxREAL);
    st_OUT = mxCreateNumericArray(2, dims2, mxINT32_CLASS, mxREAL);
    m_OUT = mxCreateNumericArray(2, dims1, mxINT32_CLASS, mxREAL);
    shift_ride_OUT = mxCreateNumericArray(2, dims2, mxINT32_CLASS, mxREAL);
    e_match_OUT = mxCreateNumericArray(2, dims1, mxDOUBLE_CLASS, mxREAL);
    eC_N_OUT = mxCreateNumericArray(2, dims1, mxDOUBLE_CLASS, mxREAL);
    eC_N2_OUT = mxCreateNumericArray(2, dims1, mxDOUBLE_CLASS, mxREAL);
    ll_OUT = mxCreateNumericArray(2, dims1, mxDOUBLE_CLASS, mxREAL);
    rej_OUT = mxCreateNumericArray(2, dims5, mxINT32_CLASS, mxREAL);
    mu_OUT = mxCreateNumericArray(2, dims1, mxDOUBLE_CLASS, mxREAL);
    temp_C_OUT = mxCreateNumericArray(2, dims_temp_C, mxINT32_CLASS, mxREAL);
            
    double *VC = mxGetPr(VC_OUT);
    double *VC2 = mxGetPr(VC2_OUT);
    double *fare = mxGetPr(fare_OUT);
    int *st = mxGetPr(st_OUT);
    int *m = mxGetPr(m_OUT);
    int *shift_ride = mxGetPr(shift_ride_OUT); for (int idx = 0; idx < N * iter; idx++) {shift_ride[idx] = 0;}
    double *e_match = mxGetPr(e_match_OUT);
    double *eC_N = mxGetPr(eC_N_OUT); 
    double *eC_N2 = mxGetPr(eC_N2_OUT); 
    double* ll = mxGetPr(ll_OUT);
    int* rej = mxGetPr(rej_OUT); 
    double* mu = mxGetPr(mu_OUT);  
    
    int *C = mxGetPr(C_IN);
    double *P = mxGetPr(P_IN);
    double *Pe = mxGetPr(Pe_IN);
    double *p_take1 = mxGetPr(p_take1_IN);
    double *p_take2 = mxGetPr(p_take2_IN);
    double *Q = mxGetPr(Q_IN);
    double *pi = mxGetPr(pi_IN);
    double *x = mxGetPr(x_IN);
    double *param = mxGetPr(param_IN);
    // iter defined in header
    int *tz = mxGetPr(tz_IN);
    int *ta = mxGetPr(ta_IN);
    int *TD = mxGetPr(TD_IN);
    double *U = mxGetPr(U_IN);
    int *perm = mxGetPr(perm_IN);
    int *map = mxGetPr(map_IN);
    double *eQ = mxGetPr(eQ_IN);
    double delta = mxGetScalar(delta_IN);
    double *fuel_cost = mxGetPr(fuel_cost_IN);
    double *tr_Q = mxGetPr(tr_Q_IN);
    double *pp = mxGetPr(pp_IN);
    double kappa = mxGetScalar(kappa_IN);
    double *gC_N = mxGetPr(gC_N_IN);
    double gC_epct = mxGetScalar(gC_epct_IN);
    
    //int *temp_C = malloc(sizeof (int) * N * 6); for (int idx = 0; idx < 6 * N; idx++) {temp_C[idx] = C[idx];}
    int *temp_C = mxGetPr(temp_C_OUT); for (int idx = 0; idx < 6 * N; idx++) {temp_C[idx] = C[idx];}
    double *alpha = malloc (sizeof (double) * 5);
    for (int zz = 0; zz < 5; zz++) {
        alpha[zz] = param[zz];
    }
    double *beta = malloc (sizeof (double) * Z);
    for (int z = 0; z < Z; z++) {
        beta[z] = param[z+5];
    }
    double *gamma = malloc (sizeof (double) * 15);
    for (int s = 0; s < 15; s++) {
        gamma[s] = param[ s+Z+5 ];
    }
    double sigma = param[40];
    
    // base supply
    for (int s = 0; s < iter*S; s++) {
        VC[s] = 5;
        VC2[s] = 5;
        eC_N[s] = 0.0;
        eC_N2[s] = 0.0;
    }    
        
    omp_set_num_threads(80);
    
    // main routines
    int t = 1;
    int i = 0;
    while ( i < iter )
    {
        // count eC, VC - VC includes eC
        int *make_dec = malloc (sizeof (int) * N);
        int n, s, t_check;
        double total_eC = 0.0;
        for ( n = 0; n < N; n++ )
        {
            t_check = TD[(temp_C[n+2*N]-1)+T*(t-1)];
            if ((i==0) && (t==1)) {
                if (t_check < temp_C[n+3*N]) {
                    make_dec[n] = 1;
                    s = Z * (t - 1) + temp_C[n];
                    VC[s - 1 + i*S] += 1.0;
                    if (temp_C[n+5*N] == 1) { 
                        eC_N[s - 1 + i*S] += 1.0; //eC_N(z_n) += 1;
                        total_eC += 1.0;
                    } 
                } else {make_dec[n] = 0;}
            } else {
                if ((t_check < temp_C[n+3*N]) && (temp_C[n+N] == t)) {
                    make_dec[n] = 1;
                    s = Z * (t - 1) + temp_C[n];
                    VC[s - 1 + i*S] += 1.0;
                    if (temp_C[n+5*N] == 1) { 
                        eC_N[s - 1 + i*S] += 1.0; 
                        total_eC += 1.0;  
                    }
                } else {make_dec[n] = 0;}
            }
        } // endfor n
        
        // mexPrintf("i = %d, t = %d, total eC = %f, matched_eC = %f, active_eC = %f\n", i+1, t, total_eC, matched_eC, total_eC + matched_eC);
        
        double *kappa_hat = malloc (sizeof (double) * Z); for (int z = 0; z < Z; z++) {kappa_hat[z] = kappa;}
        G(mu, eC_N, eC_N2, x, i, t, Z, S, eQ, kappa, kappa_hat, gC_N, gC_epct);
        F(ll, VC, VC2, i, S, alpha, beta, gamma, x, t, Z, map, gC_N);
            
        
        int tl, d, z2, ride, ss, ss_idx, td1, s2, s2_idx, ss2, ss2_idx, td2, ride2, u1_idx, u2_idx, ud_idx, d_len, e_ride, e_dec, er1_idx, ed1_idx, ess1_idx, e_ride2, e_dec2, er2_idx, ed2_idx, ess2_idx, e_ss1, e_ss2, e_td1, e_td2;
        double p1, p2, pe1, pe2;
        //int enter_vol = 0;
        #pragma omp parallel for private(s, td1, tl, d, z2, ride, ss, ss_idx, s2, s2_idx, ss2, ss2_idx, ride2, td2, u1_idx, u2_idx, ud_idx, p1, p2, pe1, pe2, d_len, e_ride, e_ride2, e_dec, e_dec2, er1_idx, ed1_idx, ess1_idx, er2_idx, ed2_idx, ess2_idx, e_ss1, e_ss2, e_td1, e_td2) 
        for ( n = 0; n < N; n++ )
        {
            if (make_dec[n]==1) 
            {
                s = Z * (t - 1) + temp_C[n]; // s in real form
                tl = TD[(t-1)+T*(temp_C[n+4*N]-1)];
                ud_idx = perm[(i)+(n)*iter]-1 + (4*T+t-1)*N;
                
                if (temp_C[n+5*N]==1) {
                    d = weighted_draw(Pe, S, TT, D+1, s-1, tl, U[ud_idx]);
                } else {
                    d = weighted_draw(P, S, TT, D+1, s-1, tl, U[ud_idx]); // in index form
                }
                
                if (d < D)
                {
                    // draw e-hail
                    if (temp_C[n+5*N]==1) {
                        pe1 = x[s-1]/10*mu[s-1+i*S]*(1-pow(1-pp[s-1]*(1-pp[s-1+S]),kappa_hat[temp_C[n]-1]));
                        //mexPrintf("%f/10*%f*(1-%f^%f) = %f\n",x[s-1],mu[s-1+i*S],1-pp[s-1]*(1-pp[s-1+S]),kappa_hat[temp_C[n]-1],pe1);
                        er1_idx = perm[(i)+(n)*iter]-1 + (5*T+t-1)*N;
                        ed1_idx = perm[(i)+(n)*iter]-1 + (6*T+t-1)*N;
                        e_ride = ((U[er1_idx] < pe1) && (e_match[s-1+i*S]<eQ[s-1+i*S]));
                        e_dec = (U[ed1_idx] < p_take1[s-1 + tl*S + d*TT*S]);
                        if ((e_ride==1) && (e_dec==1)) {
                            e_match[s-1+i*S] += 1.0;
                            ess1_idx = perm[(i)+(n)*iter]-1 + (7*T+t-1)*N;
                            e_ss1 = weighted_draw(Q, S, 1, S, s-1, 0, U[ess1_idx]);
                            e_td1 = TD[(e_ss1+Z-1)/Z-1+((s-1+Z)/Z-1)*T];
                            fare[n+i*N] += pi[(s-1)+S*e_ss1];
                            temp_C[n+N] = (e_ss1+Z)/Z; // next active t in real form
                            temp_C[n] = e_ss1 + 1 - Z * (temp_C[n+N]-1); // next zone in real form
                            shift_ride[n+i*N] += 1;
                            m[s-1+i*S] += 1;
                            if (TD[(t-1)+T*((e_ss1+Z)/Z-1)] > tl) {
                                temp_C[n+N] = temp_C[n+2*N];
                                fare[n+i*N] -= 30;
                            }
                            //if (s==19) {mexPrintf("n = %d, s = %d, eQ = %f, eC_N = %f, pe = %f, pre-update mu: %f\n",n,s,eQ[s-1+i*S],eC_N[s-1+i*S],pe1,mu[s-1+i*S]);}
                            //eQ[s-1+i*S] = max(eQ[s+i*S]-1.0,0.0);
                            //eC_N[s-1+i*S] = max(eC_N[s+i*S]-1.0,0.0);
                            //mu[s-1+i*S] = min(eQ[s-1+i*S]/eC_N[s-1+i*S],1.0);
                            //if (s==19) {mexPrintf("n = %d, s = %d, eQ = %f, eC_N = %f, pe = %f, updated mu: %f \n",n,s,eQ[s-1+i*S],eC_N[s-1+i*S],x[s-1]/10*mu[s-1+i*S]*(1-pow(1-pp[s-1]*(1-pp[s-1+S]),kappa_hat[temp_C[n]-1])),mu[s-1+i*S]);}
                            continue;
                        } else if ((e_ride==1) && (e_dec==0)) {
                            rej[i-1] += 1;
                        }
                    }
                    z2 = tz[temp_C[n]-1 + Z * d]; // z2 in real form
                    u1_idx = perm[(i)+(n)*iter]-1 + (t-1)*N;
                    p1 = 1-exp(-ll[s-1+i*S]*x[s-1]);
                    ride = ( U[u1_idx] < p1 ); 
                    ss_idx = perm[(i)+(n)*iter]-1 + (2*T+t-1)*N;
                    ss = weighted_draw(Q, S, 1, S, s-1, 0, U[ss_idx]); // in index form
                    td1 = TD[(s+Z-1)/Z-1+((ss+Z)/Z-1)*T];
                    if ((ride == 1) && (td1 <= tl))
                    {
                        temp_C[n+N] = (ss+Z)/Z; // next active t in real form
                        temp_C[n] = ss + 1 - Z * (temp_C[n+N]-1); // next zone in real form
                        fare[n+i*N] += pi[(s-1)+S*ss]; 
                        shift_ride[n+i*N] += 1;
                        m[s-1+i*S] += 1;
                    } 
                    else { //end first draw, begin second draw
                       	d_len = ta[temp_C[n]-1+d*Z+(t-1)/6*Z*D];
                        s2 = next_s(Z*(t-1)+z2, d_len, T, Z); 
                        if (d_len==0)
                        {
                            if (temp_C[n+5*N]==1) {
                                eC_N2[s2 - 1 + i*S] += 1.0;
                                pe2 = (10-x[s-1])/10*mu[s2-1+i*S]*(1-pow(1-pp[s2-1]*(1-pp[s2-1+S]),kappa_hat[z2-1]));
                                er2_idx = perm[(i)+(n)*iter]-1 + (8*T+t-1)*N;
                                ed2_idx = perm[(i)+(n)*iter]-1 + (9*T+t-1)*N;
                                e_ride2 = ((U[er2_idx] < pe2) && (e_match[s2-1+i*S]<eQ[s2-1+i*S]));
                                e_dec2 = (U[ed2_idx] < p_take2[s2-1 + tl*S + d*TT*S]);
                                if ((e_ride2==1) && (e_dec2==1)) {
                                    e_match[s2-1+i*S] += 1;
                                    ess2_idx = perm[(i)+(n)*iter]-1 + (10*T+t-1)*N;
                                    e_ss2 = weighted_draw(Q, S, 1, S, s2-1, 0, U[ess2_idx]);
                                    e_td2 = TD[(e_ss2+Z)/Z-1+((s2-1+Z)/Z-1)*T];
                                    fare[n+i*N] += pi[(s2-1)+S*e_ss2];
                                    temp_C[n+N] = (e_ss2+Z)/Z; // next active t in real form
                                    temp_C[n] = e_ss2 + 1 - Z * (temp_C[n+N]-1); // next zone in real form
                                    if (TD[(t-1)+T*((e_ss2+Z)/Z-1)] > tl) {
                                        temp_C[n+N] = temp_C[n+2*N];
                                        fare[n+i*N] -= 30;
                                    }
                                    
                                    //if (s2==19) {mexPrintf("n = %d, s = %d, eQ = %f, eC_N = %f, pe = %f, pre-update mu: %f\n",n,s2,eQ[s2-1+i*S],eC_N[s2-1+i*S],pe2,mu[s2-1+i*S]);}
                                    //eQ[s2-1+i*S] = max(eQ[s2-1+i*S]-1.0,0.0);
                                    //eC_N[s2-1+i*S] = max(eC_N[s2-1+i*S]-1.0,0.0);
                                    //mu[s2-1+i*S] = min(eQ[s2-1+i*S]/eC_N[s2-1+i*S],1.0);
                                    //if (s2==19) {mexPrintf("n = %d, s = %d, eQ = %f, eC_N = %f, pe = %f, updated mu: %f\n",n,s2,eQ[s2-1+i*S],eC_N[s2-1+i*S],(10-x[s-1])/10*mu[s2-1+i*S]*(1-pow(1-pp[s2-1]*(1-pp[s2-1+S]),kappa_hat[z2-1])),mu[s2-1+i*S]);}
                                    continue;
                                } else if ((e_ride2==1) && (e_dec2==0)) {rej[i-1] += 1;}
                            }
                            VC2[s2 - 1 + i*S] += 1.0;
                            u2_idx = perm[(i)+(n)*iter]-1 + (T+t-1)*N;
                            p2 = 1-exp(-ll[s2-1+i*S]*(10-x[s-1]));
                            ride2 = ( U[u2_idx] < p2 );
                            ss2_idx = perm[(i)+(n)*iter]-1 + (3*T+t-1)*N;
                            ss2 = weighted_draw(Q, S, 1, S, s2-1, 0, U[ss2_idx]);
                            td2 = TD[(s2+Z-1)/Z-1+((ss2+Z)/Z-1)*T];
                            if ((ride2 == 1) && td2 <= tl)
                            {
                                temp_C[n+N] = (ss2+Z)/Z; // next time
                                temp_C[n] = ss2 + 1 - Z * (temp_C[n+N]-1); // next zone in real form
                                fare[n+i*N] += pi[(s2-1)+S*ss2]; 
                                shift_ride[n+i*N] += 1;
                                m[s2-1+i*S] += 1;
                            } else {
                                temp_C[n] = z2;
                                temp_C[n+N] = t+1; 
                                if (temp_C[n+N] > T) {temp_C[n+N] = 1;}
                                st[n + i*N] += 1;
                            }
                        }
                        else {
                            temp_C[n] = z2;
                            temp_C[n+N] = t+d_len;
                            if (t+d_len > T) {temp_C[n+N] -= T;}
                            st[n+i*N] += d_len;
                        }
                    } // end second draw
                } else { //end case d < 7, begin case d = 7
                    if (t==T) {temp_C[n+N] = 1;} else {temp_C[n+N] = t+1;}
                } // if d = 8, take rest for one period
            } else { // end if for eligible cabs before end of shift, begin case for end of shift
                if (temp_C[n+4*N]==t) {temp_C[n+N] = temp_C[n+2*N];}
            }   
        } // end for loop n
             
        for (int z = 1; z <= Z; z++) {
            int s = z + Z * (t-1);
            e_match[s-1+i*S] += gC_N[s-1] * gC_epct*mu[s-1+i*S] * (1-pow(1-pp[s-1]*(1-pp[s-1+S]),kappa_hat[z-1]));
        }

        free(make_dec);
               
        t += 1;
        if (t == T + 1) 
        {
            i += 1;
            t = 1;
        }
    } // end while
    return;
} // end mexFunction


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


// Draw a choice out of Z possible choices with proabability matrix P, an X*Y*. array
// i, j defines the state of the agent - in idx form
// output d is in idx form
int weighted_draw(double *P, int X, int Y, int Z, int i, int j, double u) 
{
    int d = 0;
    int idx = i + j * X + d * X * Y;
    while ((u > P[idx]) && (d < Z-1)) {   // need to take care of u = 1!!
        d += 1;
        idx += X * Y;
    }
    return d;
}


void F(double *ll, double *VC, double *VC2, int i, int S, double *alpha, double *beta, double *gamma, double *x, int t, int Z, int *map, double *gC_N) 
{
    for (int z = 0; z < Z; z++) {
        int s = Z * (t-1) + z; // s in index form
        int zz = map[z];
        double r;
        if ((t >=1 ) && (t <=38)) {r = gamma[zz];}
        else if ((t >= 39) && (t <= 66)) {r = gamma[5+zz];}
        else if ((t >= 97) && (t <=120)) {r = gamma[10+zz];}
        else {r = 1.0;}
        double c;
        if (i > 0) {
            c = VC[s + i*S]*x[s]/10+VC2[s + (i-1)*S]*(10-x[s]) / 10 + gC_N[s];
        } else {
            c = VC[s + i*S] + gC_N[s];
        }
        if (c <= 0) {c = 5;}
        if ((beta[z]*r)*pow(c,alpha[zz]-1) < 1) {
            ll[s + i*S] = -log(1-(beta[z]*r)*pow(c,alpha[zz]-1)) / x[s];
        } else {
            ll[s + i*S] = 0.005;
        }
        //mexPrintf("i = %d, z = %d, t = %d, s = %d, c = %f, alpha = %f, beta = %f, gamma = %f, x = %f, ll = %f\n", i+1, z+1, t, s+1, c, alpha[zz], beta[z], r, x[s], ll[s+i*S]);
    }
}


void G(double* mu, double* eC_N, double* eC_N2, double *x, int i, int t, int Z, int S, double* eQ, double kappa, double *kappa_hat, double *gC_N, double gC_epct) 
{
	for (int z = 0; z < Z; z++) {
        int s = z + Z*(t-1); double c; 
        if (i==0) {
            c = eC_N[s] + gC_N[s] * gC_epct;
        } else {
            c = eC_N[s + i*S] * x[s]/10 + eC_N2[s + (i-1)*S] * (10-x[s])/10 + gC_N[s] * gC_epct;
        }
        if ((c == 0) && (eQ[s+i*S] == 0)) {
            mu[s + i*S] = 0;
        } else if ((c == 0) && (eQ[s+i*S] > 0)) {
            mu[s + i*S] = 1;
        } else {
            if (c > eQ[s+i*S]) {mu[s + i*S] = eQ[s + i*S] / c;} else {mu[s + i*S] = 1;}
        }
        if (kappa > c) {kappa_hat[z] = c;}
    }
}
