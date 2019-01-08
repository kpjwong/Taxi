function [param fval history] = run_SMM_ehail(p0,C_IN,true_mom,S,Q,PI,U,eta,Perm,flag,print_flag,iter)
    history.param = [];
    history.fval = [];
    options = optimset('OutputFcn', @outfun, 'Display', 'iter', 'PlotFcns', @optimplotx, 'MaxIter', iter);
 
    N = size(C_IN,1);
    T = 144;
    
    full_S = [1:length(p0)];
    if length(S) < length(full_S)
        SC = setdiff(full_S, S);
        S = sort(S);
        dS = [S(1),diff(S)];
        dSC = [SC(1),diff(SC)];
        do_flag = (SC(1)==1);
        param_str = '[';
        initial_str = '[';
        s_idx = 1;
        sc_idx = 1;
        while (s_idx <= length(S)) || (sc_idx <= length(SC))
            if do_flag == 0
                param_str = strcat(param_str, 'theta(', num2str(s_idx), ')');
                initial_str = strcat(initial_str, 'p0(', num2str(S(s_idx)), ')');
                s_idx = s_idx+1;
                if (s_idx <= length(S)), initial_str = strcat(initial_str, ';'); else, initial_str = strcat(initial_str, ']'); end
                if (s_idx <= length(S)), do_flag = (dS(s_idx)~=1); else, do_flag = 1; end
            else
                param_str = strcat(param_str, 'p0(', num2str(SC(sc_idx)), ')');
                sc_idx = sc_idx+1;
                if (sc_idx <= length(SC)), do_flag = (dSC(sc_idx)==1); else, do_flag = 0; end
            end
            if (s_idx <= length(S)) || (sc_idx <= length(SC))
                param_str = strcat(param_str, ';');
            else
                param_str = strcat(param_str, ']');
            end
        end
        cmd_str = strcat('[param fval] = fminsearch(@(theta)SMM_dist_ehail(',param_str,', C_IN, true_mom, Q, PI, U, eta, Perm, flag, print_flag),',initial_str,', options);');
        eval(cmd_str);
    elseif length(S) == length(full_S)
        [param fval] = fminsearch(@(theta)SMM_dist_ehail(theta, C_IN, true_mom, Q, PI, U, eta, Perm, flag, print_flag), p0, options);
    end
    
    function stop = outfun(param,optimValues,state)
        stop = false;
        if isequal(state,'iter')
          history.param = [history.param; param];
          history.fval = [history.fval; optimValues.fval];
        end
    end 
end