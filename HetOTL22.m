function [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = HetOTL22(Y, Kernel, P_it, text_gnd, options, id_list)
% HomOTL2: HomOTL-II
%--------------------------------------------------------------------------
% Input:
%        Y:    the vector of lables
%        K:    precomputed kernel for all the example, i.e., K_{ij}=K(x_i,x_j)
%  id_list:    a randomized ID list
%  options:    a struct containing rho, sigma, C, n_label and n_tick;
% Output:
%   err_count:  total number of training errors
%    run_time:  time consumed by this algorithm once
%    mistakes:  a vector of mistake rate
% mistake_idx:  a vector of number, in which every number corresponds to a
%               mistake rate in the vector above
%         SVs:  a vector records the number of support vectors
%     size_SV:  the size of final support set
%--------------------------------------------------------------------------

%% initialize parameters
% beta = sqrt(length(id_list))/(sqrt(length(id_list))+sqrt(log(2)));
beta = options.beta;
C = options.C; % 1 by default
T_TICK = options.t_tick;
alpha2 = [];
SV2 = [];
SV1 = [];
ID = id_list;
err_count = 0;
mistakes = [];
mistakes_idx = [];
SVs = [];
TMs=[];
%=============================
w_2t=1/2;
w_3t=1/2;
%=============================

t_tick = T_TICK; %10;
%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    
    if (isempty(alpha2)),
        f2_t=0;
    else
        k2_t=Kernel(id,SV2(:))';
        f2_t=alpha2*k2_t;
    end
     
		%======================================
		f3_t = 0;
		gnd = zeros(1,options.K);
		sim = zeros(1,options.K);
		similarity = P_it(id,:);
		%sim = similarity / sum(similarity);
		%f3_t = sim * text_gnd;
    for iter = 1 : options.K
			  max_value = 0;
				max_index = 1;
		    for jter = 1 : length(similarity)
					  if (similarity(jter) > max_value)
							  max_value = similarity(jter);
								max_index = jter;
						end
				end
				%f3_t = f3_t + text_gnd(max_index);
				gnd(iter) = text_gnd(max_index);
				sim(iter) = similarity(max_index);
				similarity(max_index) = 0;
		end
		%f3_t = f3_t / K;
		sim = sim / sum(sim);
		f3_t = gnd * sim';

		f_t = w_2t*sign(f2_t) + w_3t*sign(f3_t);
		%======================================
    
    hat_y_t = sign(f_t);        % prediction
    if (hat_y_t==0)
        hat_y_t=1;
    end
    % count accumulative mistakes
    if (hat_y_t~=Y(id)),
        err_count = err_count + 1;
    end
    
		%======================================
    z_2 = (Y(id)*f2_t<=0);
    z_3 = (Y(id)*f3_t<=0);
    w_2t=w_2t*beta^z_2;
    w_3t=w_3t*beta^z_3;
    sum_w=w_2t+w_3t;
    w_2t=w_2t/sum_w;
    w_3t=w_3t/sum_w;
		%======================================
    
    l2_t = max(0,1-Y(id)*f2_t);   % hinge loss
    if (l2_t>0)
        % update
        s2_t=Kernel(id,id);
        gamma_t = l2_t/(s2_t+1/(2*C));
        alpha2 = [alpha2 Y(id)*gamma_t;];
        SV2 = [SV2 id];
    end
    run_time=toc;
    if t<T_TICK
        if (t==t_tick)
            mistakes = [mistakes err_count/t];
            mistakes_idx = [mistakes_idx t];
            SVs = [SVs length(SV1)+length(SV2)];
            TMs=[TMs run_time];
            
            t_tick=2*t_tick;
            if t_tick>=T_TICK,
                t_tick = T_TICK;
            end
            
        end
    else
        if (mod(t,t_tick)==0)
            mistakes = [mistakes err_count/t];
            mistakes_idx = [mistakes_idx t];
            SVs = [SVs length(SV1)+length(SV2)];
            TMs=[TMs run_time];
        end
    end
    
    
end
classifier.SV1 = SV1;
classifier.SV2 = SV2;
classifier.alpha2 = alpha2;
%fprintf(1,'The number of mistakes = %d\n', err_count);
run_time = toc;
