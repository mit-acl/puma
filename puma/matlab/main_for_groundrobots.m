% /* ----------------------------------------------------------------------------
%  * Copyright 2024, Kota Kondo, Aerospace Controls Laboratory
%  * Massachusetts Institute of Technology
%  * All Rights Reserved
%  * Authors: Kota Kondo, et al.
%  * See LICENSE file for the license information
%  * -------------------------------------------------------------------------- */

%% Initialization

close all; clc; clear;
doSetup();
import casadi.*
const_p = {};    
opti = casadi.Opti();

%% Parameters for the optimization 

optimize_n_planes = false;     %Optimize the normal vector "n" of the planes (the "tilt") (see Panther paper diagram)
optimize_d_planes = false;     %Optimize the scalar "d" of the planes (the distance) (see Panther paper diagram)
optimize_time_alloc = true;

%% Whether or not the dynamic limits and obstacle avoidance constraints are formulated as hard constraints (as equalities/inequalities) or soft constraints (in the objective function)
soft_dynamic_limits_constraints = false;
soft_obstacle_avoid_constraint = false;

%% Plots?
make_plots = true;

%%
%% Problem formulation parameters
%%

deg_pos = 3;                %The degree of the position polynomial
num_seg = 7;                %The number of segments in the trajectory (the more segments the less conservative the trajectory is [also makes optimization problem harder])
num_max_of_obst = 5;        % This is the maximum num of the obstacles that will be considered in the constraints
dim_pos = 2;                %The dimension of the position trajectory (R3)
offset_vel = 0.1;
basis = "MINVO";            %MINVO OR B_SPLINE or BEZIER. This is the basis used for collision checking (in position, velocity, accel and jerk space), both in Matlab and in C++
linear_solver_name = 'ma27';%mumps [default, comes when installing casadi], ma27, ma57, ma77, ma86, ma97 
print_level = 0;            %From 0 (no verbose) to 12 (very verbose), default is 5
jit = false;

%%
%% Constants for spline fitted to the obstacle trajectory
%%

fitter.deg_pos = 3;                                         % The degree of the fit past obstacle trajectory
fitter.num_seg = 7;                                         % The number of segments in the fit in the past obstacle trajectory
fitter.dim_pos = dim_pos;                                         % The dimension of the fit past obstacle (3 for R3)
fitter.num_samples = 20;                                    % The number of samples used to fit the past obstacle trajectory
fitter_num_of_cps = fitter.num_seg + fitter.deg_pos;        % The number of control points of fit past obstacle trajectory (B-spline)
for i = 1:num_max_of_obst
    fitter.ctrl_pts{i} = opti.parameter(fitter.dim_pos,fitter_num_of_cps);          % This comes from C++
    fitter.bbox_inflated{i} = opti.parameter(fitter.dim_pos,1);                     % This comes from C++
    fitter.bs_casadi{i} = MyCasadiClampedUniformSpline(0,1,fitter.deg_pos,fitter.dim_pos,fitter.num_seg,fitter.ctrl_pts{i}, false);
end
fitter.bs=MyClampedUniformSpline(0,1, fitter.deg_pos, fitter.dim_pos, fitter.num_seg, opti); 
fitter.total_time = 6.0; %Time from (time at point d) to end of the fitted spline %The total time of the fit past obstacle trajectory

%%%%
%%%% NOTE: Everything uses B-Spline control points except for obstacle constraints which use the set basis (usually MINVO)
%%%%

%% The number of segments used to discritize the past obstacle trajectory in the collision avoidance constraints 
sampler.num_samples_obstacle_per_segment = 4;                           %This is used for both the feature sampling (simpson), and the obstacle avoidance sampling
sampler.num_samples=sampler.num_samples_obstacle_per_segment*num_seg;   %This will also be the num_of_layers in the graph yaw search of C++

%% The trajectory is optimized from t=0 to t=1 then the total time is scaled by the decision variable alpha
t0_n=0.0; tf_n=1.0;
assert(tf_n>t0_n);
assert(t0_n==0.0); %This must be 0! (assumed in the C++ and MATLAB code)
assert(tf_n==1.0); %This must be 1! (assumed in the C++ and MATLAB code)

%NOTE: All of the opti.parameter values are set by the C++ code by puma.yaml
%% factors for the cost
c_pos_smooth = opti.parameter(1,1); %Position smoothing cost factor
c_final_pos  = opti.parameter(1,1); %Distance to goal position cost factor
c_total_time = opti.parameter(1,1); %Total time cost factor

%% The radius of the planning horizon sphere
Ra = opti.parameter(1,1);

%% If we are optimizing the total time (time allocation) then setup alpha as a decision variable, else it is a parameter read from puma.yaml
alpha=opti.variable(1,1); 
total_time=alpha*(tf_n-t0_n); %Total time is (tf_n-t0_n)*alpha. (should be 1 * alpha)

%% Initial and final conditions, and max values
p0=opti.parameter(dim_pos,1); v0=opti.parameter(dim_pos,1); a0=opti.parameter(dim_pos,1);
pf=opti.parameter(dim_pos,1); vf=opti.parameter(dim_pos,1); af=opti.parameter(dim_pos,1);
v_max=opti.parameter(dim_pos,1); a_max=opti.parameter(dim_pos,1); j_max=opti.parameter(dim_pos,1);

%https://github.com/mit-acl/deep_panther/blob/master/panther/matlab/other/explanation_normalization.svg
%Normalized v0, a0, v_max,... (Normalized values for time 0 to 1 * alpha, non-normalized are for time 0 to 1)
v0_n=v0*alpha; a0_n=a0*(alpha^2);
vf_n=vf*alpha; af_n=af*(alpha^2);
v_max_n=v_max*alpha; a_max_n=a_max*(alpha^2); j_max_n=j_max*(alpha^3);

%% Planes
n={}; d={};
for i=1:(num_max_of_obst*num_seg)

    %If we are optimizing the plane normals, add them as decision variables, else they are parameters
    if(optimize_n_planes)
        n{i}=opti.variable(dim_pos,1); 
    else
        n{i}=opti.parameter(dim_pos,1); 
    end
    
    %If we are optimizing the plane distances, add them as decision variables, else they are parameters
    if(optimize_d_planes)
        d{i}=opti.variable(1,1);
    else
        d{i}=opti.parameter(1,1); 
    end    
end

%% Min/max x, y (in flight space)
x_lim=opti.parameter(2,1); %[min max]
y_lim=opti.parameter(2,1); %[min max]

%% CREATION OF THE SPLINES! 
sp=MyClampedUniformSpline(t0_n,tf_n,deg_pos, dim_pos, num_seg, opti); %spline position.

%This part below uses the Casadi implementation of a BSpline. However, that
%implementation does not allow SX variables (which means that it cannot be
%expanded). See more at https://github.com/jtorde/useful_things/blob/master/casadi/bspline_example/bspline_example.m
% t_eval_sym=MX.sym('t_eval_sym',1,1); 
% %See https://github.com/jtorde/useful_things/blob/master/casadi/bspline_example/bspline_example.m
% my_bs_parametric_tmp=casadi.bspline(t_eval_sym, fitter.ctrl_pts(:), {knots_obstacle}, {[deg_obstacle]}, dim_obstacle); %Note that here we use casadi.bspline, NOT casadi.Function.bspline
% my_bs_parametric = Function('my_bs_parametric',{t_eval_sym,fitter.ctrl_pts},{my_bs_parametric_tmp});
deltaT=total_time/num_seg; %Time allocated for each segment
obst={}; %Obs{i}{j} Contains the vertexes (as columns) of the obstacle i in the interval j

%%
%% CONSTRAINTS! 
%%

%% Creates points (centers of the obstacles and verticies) used for obstacle constraints
%% reuse the time t from the previous iteration
for i=1:num_max_of_obst
    all_centers=[];
    replan_time_index = 1;
    for j=1:num_seg
        all_vertexes_segment_j=[];
        for k=1:sampler.num_samples_obstacle_per_segment
            t_obs = deltaT*(j-1) + (k/sampler.num_samples_obstacle_per_segment)*deltaT;
            t_nobs= min( t_obs/fitter.total_time,  1.0 );  %Note that fitter.bs_casadi{i}.knots=[0...1]
            
            % get the center of the obstacle
            pos_center_obs=fitter.bs_casadi{i}.getPosT(t_nobs);
            all_centers=[all_centers pos_center_obs];
            all_vertexes_segment_j=[all_vertexes_segment_j vertexesOfBox(pos_center_obs, fitter.bbox_inflated{i}, 2)];
            replan_time_index = replan_time_index + 1;
        end
        obst{i}.vertexes{j}=all_vertexes_segment_j;
    end  
    obst{i}.centers=all_centers;
end

%% Set the total time and calculate alpha
total_time_n=(tf_n-t0_n);
alpha=total_time/total_time_n;  %Please read explanation_normalization.svg

%% Initial conditions
const_p{end+1}= sp.getPosT(t0_n)== p0 ;
const_p{end+1}= sp.getVelT(t0_n)== v0_n ;
const_p{end+1}= sp.getAccelT(t0_n)== a0_n ;

%% Final conditions
% opti.subject_to( sp.getPosT(tf)== pf );
const_p{end+1}= sp.getVelT(tf_n)== vf_n ;
const_p{end+1}= sp.getAccelT(tf_n)== af_n ;

%% Need to ensure total time of trajectory being optimized is less than the predicted time of th obstacles
% if optimize_time_alloc
%     const_p{end+1}= total_time<=fitter.total_time; %Samples for visibility/obs_avoidance are only taken for t<fitter.total_time
% end

%%
%% One plane per segment per obstacle
%%

const_p_obs_avoid={};
for j=1:(sp.num_seg)

    %Get the control points of the interval
    Q=sp.getCPs_XX_Pos_ofInterval(basis, j);

    %Plane constraints
    for obst_index=1:num_max_of_obst
      ip = (obst_index-1) * sp.num_seg + j;  % index plane
       
      % The obstacle should be on one side
      % I need this constraint if alpha is a dec. variable OR if n is a dec
      % variable OR if d is a dec variable
      
      % if(optimize_n_planes || optimize_d_planes || optimize_time_alloc)
      if(optimize_n_planes || optimize_d_planes)
      
          for i=1:num_max_of_obst
            vertexes_ij=obst{i}.vertexes{j};
            for kk=1:size(vertexes_ij,2)
                const_p_obs_avoid{end+1} = n{ip}'*vertexes_ij(:,kk) + d{ip} >= 1; 
            end
          end
      
      end
      
      % the control points on the other side
      for kk=1:size(Q,2)
        const_p_obs_avoid{end+1}= n{ip}'*Q{kk} + d{ip} <= -1;
      end
    end  
    
    % Sphere constraints
    for kk=1:size(Q,2) 
        tmp=(Q{kk}-p0);
        const_p{end+1}= (tmp'*tmp)<=(Ra*Ra) ;
    end

    % Min max xyz constraints
    for kk=1:size(Q,2) 
        t_obs=Q{kk};
        const_p{end+1} = x_lim(1)<=t_obs(1);
        const_p{end+1} = x_lim(2)>=t_obs(1);
        const_p{end+1} = y_lim(1)<=t_obs(2);
        const_p{end+1} = y_lim(2)>=t_obs(2);
    end
end

%%
%% OBJECTIVE!
%%

clear i

pos_smooth_cost=sp.getControlCost()/(alpha^(sp.p-1));
final_pos_cost=(sp.getPosT(tf_n)- pf)'*(sp.getPosT(tf_n)- pf);
total_time_cost=alpha*(tf_n-t0_n);

total_cost=c_pos_smooth*pos_smooth_cost+...
           c_final_pos*final_pos_cost+...
           c_total_time*total_time_cost;

%%
%% First option: Hard constraints
%%

const_p_dyn_limits={};
const_p_dyn_limits = addDynLimConstraints(const_p_dyn_limits, sp, basis, v_max_n, a_max_n, j_max_n);

%%
%% get translational dyn. limits violation
%%

opti_tmp=opti.copy;
opti_tmp.subject_to(); %Clear constraints
opti_tmp.subject_to(const_p_dyn_limits);
translatoinal_violation_dyn_limits=getViolationConstraints(opti_tmp);

%%
%% get translational dyn. limits violation
%%

opti_tmp=opti.copy;
opti_tmp.subject_to(); %Clear constraints
opti_tmp.subject_to(const_p_dyn_limits);
violation_dyn_limits=getViolationConstraints(opti_tmp);

%%
%% get obstacle avoidance violation
%%

opti_tmp=opti.copy;
opti_tmp.subject_to(); %Clear constraints
opti_tmp.subject_to(const_p_obs_avoid);
violation_obs_avoid=getViolationConstraints(opti_tmp);

%%
%% Add all the constraints
%%

if soft_dynamic_limits_constraints
    total_cost = total_cost + (1/numel(violation_dyn_limits))*sum(violation_dyn_limits.^2);
else
    const_p=[const_p, const_p_dyn_limits];
end

if soft_obstacle_avoid_constraint
    total_cost = total_cost + 100*(1/numel(violation_obs_avoid))*sum(violation_obs_avoid.^2);
else
    const_p=[const_p, const_p_obs_avoid];
end

% total_cost=total_cost+c_dyn_lim*getCostDynLimSoftConstraints(sp, sy, basis, v_max_n, a_max_n, j_max_n, ydot_max_n);

opti.minimize(simplify(total_cost));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% SOLVE! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

all_nd=[];
for i=1:(num_max_of_obst*num_seg)
    all_nd=[all_nd [n{i};d{i}]];
end

pCPs=sp.getCPsAsMatrix();

%% Setting all of the parameters for testing
v_max_value=1.6*ones(dim_pos,1);
a_max_value=5*ones(dim_pos,1);
j_max_value=50*ones(dim_pos,1);
Ra_value=12.0;
p0_value=[-4; 0.0];
v0_value=[0;0];
a0_value=[0;0];
pf_value=[4.0;0.0];
vf_value=[0;0];
af_value=[0;0];
dist_betw_planes=0.01; %See last figure of https://github.com/mit-acl/separator
norm_n=2/dist_betw_planes;
all_nd_value=[];
for j=1:floor(num_seg/2)
    all_nd_value=[all_nd_value norm_n*(1/sqrt(2))*[1; 1; 2] ];
end
for j=(floor(num_seg/2)+1):num_seg
    all_nd_value=[all_nd_value norm_n*(1/sqrt(2))*[-1; 1; 2] ];
end
x_lim_value=[-100;100]; y_lim_value=[-100;100];
alpha_value = 3.53467;
tmp1=[0, 0, 0, 1.64678, 2.85231, 4.05784, 5.70462, 5.70462, 5.70462, 5.70462; 
      0, 0, 0, -0.378827, -1.05089, -1.71629, -2.08373, -2.08373, -2.08373, -2.08373];
% all_obstacle_bbox_inflated_value= ones(size(all_obstacle_bbox_inflated));

par_and_init_guess= [ {createStruct('Ra', Ra, Ra_value)},...
              {createStruct('p0', p0, p0_value)},...
              {createStruct('v0', v0, v0_value)},...
              {createStruct('a0', a0, a0_value)},...
              {createStruct('pf', pf, pf_value)},...
              {createStruct('vf', vf, vf_value)},...
              {createStruct('af', af, af_value)},...
              {createStruct('v_max', v_max, v_max_value)},...
              {createStruct('a_max', a_max, a_max_value)},...
              {createStruct('j_max', j_max, j_max_value)},...
              {createStruct('x_lim', x_lim, x_lim_value)},...
              {createStruct('y_lim', y_lim, y_lim_value)},...
              {createStruct('alpha', alpha, alpha_value)},...
              {createStruct('c_pos_smooth', c_pos_smooth, 0.0)},...
              {createStruct('c_final_pos', c_final_pos, 2000)},...
              {createStruct('c_total_time', c_total_time, 1000.0)},...
              {createStruct('all_nd', all_nd, all_nd_value)},...
              {createStruct('pCPs', pCPs, tmp1)},...
              createCellArrayofStructsForObstacles(fitter)];
              
[par_and_init_guess_exprs, par_and_init_guess_names, names_value]=toExprsNamesAndNamesValue(par_and_init_guess);

opts = struct;
opts.expand=true;                   %When this option is true, it goes WAY faster!
opts.print_time=0;
opts.ipopt.print_level=print_level; 
opts.ipopt.max_iter=500;
opts.ipopt.linear_solver=linear_solver_name;
opts.jit=jit;                       %If true, when I call solve(), Matlab will automatically generate a .c file, convert it to a .mex and then solve the problem using that compiled code
opts.compiler='shell';
opts.jit_options.flags='-Ofast';    %Takes ~15 seconds to generate if O0 (much more if O1,...,O3)
opts.jit_options.verbose=false;     %See example in shallow_water.cpp
opts.ipopt.acceptable_constr_viol_tol=1e-20;
opti.solver('ipopt',opts);          %{"ipopt.hessian_approximation":"limited-memory"} 
opti.subject_to(const_p);
results_expresion={pCPs, all_nd, total_cost, pos_smooth_cost, alpha, final_pos_cost}; %Note that this containts both parameters, variables, and combination of both. If they are parameters, the corresponding value will be returned
results_names={'pCPs','all_nd','total_cost', 'pos_smooth_cost', 'alpha', 'final_pos_cost'};

%% compute cost
compute_cost = Function('compute_cost', par_and_init_guess_exprs ,{total_cost}, par_and_init_guess_names, {'total_cost'});
compute_cost(names_value{:})
compute_cost=compute_cost.expand();
compute_cost.save('./casadi_generated_files/acslam_compute_cost.casadi') %The file generated is quite big

%% compute dyn limits constraints
compute_dyn_limits_constraints_violation = casadi.Function('compute_dyn_limits_constraints_violation', par_and_init_guess_exprs ,{violation_dyn_limits}, par_and_init_guess_names ,{'violation'});
compute_dyn_limits_constraints_violation=compute_dyn_limits_constraints_violation.expand();
compute_dyn_limits_constraints_violation.save('./casadi_generated_files/acslam_compute_dyn_limits_constraints_violation.casadi'); 

%% get translational dynamic limit constraints violation
compute_trans_dyn_limits_constraints_violation = casadi.Function('compute_trans_dyn_limits_constraints_violation', par_and_init_guess_exprs ,{translatoinal_violation_dyn_limits}, par_and_init_guess_names ,{'trans_violation'});
compute_trans_dyn_limits_constraints_violation=compute_trans_dyn_limits_constraints_violation.expand();
compute_trans_dyn_limits_constraints_violation.save('./casadi_generated_files/acslam_compute_trans_dyn_limits_constraints_violation.casadi'); 

%% get optimization problem
my_func = opti.to_function('my_func', par_and_init_guess_exprs, results_expresion, par_and_init_guess_names, results_names);
my_func.save('./casadi_generated_files/acslam_op.casadi'); %Optimization Problam. The file generated is quite big
tic();
sol=my_func(names_value{:});
toc();
statistics=get_stats(my_func); %See functions defined below
results_solved=[];
for i=1:numel(results_expresion)
    results_solved=[results_solved, {createStruct(results_names{i}, results_expresion{i}, full(sol.(results_names{i})))}];
end

full(sol.pCPs) 
cprintf('Green','Total time trajec=%.2f s (alpha=%.2f) \n', full(sol.alpha*(tf_n-t0_n)), full(sol.alpha))

%%
%% FUNCTION TO FIT A SPLINE TO POSITION SAMPLES     
%%

%samples should be sampled uniformly, including first and last point
%The total number of samples is num_samples.
%If you find the error "evaluation failed" --> increase num_samples or reduce deg_pos or num_seg

samples=MX.sym('samples',fitter.dim_pos, fitter.num_samples);
cost_function=0;
i=1;
for ti=linspace(fitter.bs.knots(1), fitter.bs.knots(end), fitter.num_samples)
    dist=(fitter.bs.getPosT(ti)-samples(:,i)); % TODO: not sure if this is correct since getPosT returns 3d vector?? 
    cost_function = cost_function + dist'*dist; 
    i=i+1;
end
lagrangian = cost_function;
fitter_bs_CPs=fitter.bs.getCPsAsMatrix();
variables=fitter_bs_CPs; 
kkt_eqs=jacobian(lagrangian, variables)'; %I want kkt=[0 0 ... 0]'
%Obtain A and b
b=-casadi.substitute(kkt_eqs, variables, zeros(size(variables))); %Note the - sign
A=jacobian(kkt_eqs, variables);
solution=A\b;  %Solve the system of equations
f= Function('f', {samples }, {reshape(solution(1:end), fitter.dim_pos,-1)}, ...
                 {'samples'}, {'result'} );
t=linspace(0, 2, fitter.num_samples);
samples_value=[sin(t)+2*sin(2*t);cos(t)-2*cos(2*t)];
solution=f(samples_value);
cost_function=substitute(cost_function, fitter.bs.getCPsAsMatrix, full(solution));
cost_function=substitute(cost_function, samples, samples_value);
convertMX2Matlab(cost_function)
fitter.bs.updateCPsWithSolution(full(solution));
f.save('./casadi_generated_files/acslam_fit2d.casadi') 


%%
%% Write param file with the characteristics of the casadi function generated
%%

my_file=fopen('./casadi_generated_files/acslam_params_casadi.yaml','w'); %Overwrite content. This will clear its content
fprintf(my_file,'#DO NOT EDIT. Automatically generated by MATLAB\n');
fprintf(my_file,'#If you want to change a parameter, change it in main.m and run the main.m again\n');
fprintf(my_file,'deg_pos: %d\n',deg_pos);
fprintf(my_file,'num_seg: %d\n',num_seg);
fprintf(my_file,'num_max_of_obst: %d\n',num_max_of_obst);
fprintf(my_file,'sampler_num_samples: %d\n',sampler.num_samples);
fprintf(my_file,'fitter_num_samples: %d\n',fitter.num_samples);
fprintf(my_file,'fitter_total_time: %d\n',fitter.total_time);
fprintf(my_file,'fitter_num_seg: %d\n',fitter.num_seg);
fprintf(my_file,'fitter_deg_pos: %d\n',fitter.deg_pos);
fprintf(my_file,'deg_yaw: %d\n',2);
fprintf(my_file,'num_obst_in_FOV: %d\n',1);
fprintf(my_file,'num_of_yaw_per_layer: %d\n',40);
fprintf(my_file,'basis: "%s"\n',basis);

%%
%% Store solution! 
%%

sp_cpoints_var=sp.getCPsAsMatrix();
sp.updateCPsWithSolution(full(sol.pCPs))

%%
%% PLOTTING!
%%

if make_plots
    import casadi.*
    alpha_sol=full(sol.alpha);
    v_max_n_value= v_max_value*alpha_sol;
    a_max_n_value= a_max_value*(alpha_sol^2);
    j_max_n_value= j_max_value*(alpha_sol^3);
    sp.plotPosVelAccelJerk(v_max_n_value, a_max_n_value, j_max_n_value)

    figure;
    % plot the trajectory in 2D
    sp.plotPos2D();
    for i=1:num_max_of_obst
        tmp = substituteWithSolution(obst{i}.centers, results_solved, par_and_init_guess);
        % plot the obstacles in 2D
        plot(tmp(1,:),tmp(2,:),'r*');
        hold on;
    end
end

%%
%% Functions
%%

function result=createCellArrayofStructsForObstacles(fitter)
         
    num_obs=size(fitter.bbox_inflated,2);
    result=[];
     
    for i=1:num_obs
        name_crtl_pts=['obs_', num2str(i-1), '_ctrl_pts'];  %Note that we use i-1 here because it will be called from C++
        name_bbox_inflated=['obs_', num2str(i-1), '_bbox_inflated'];          %Note that we use i-1 here because it will be called from C++
        crtl_pts_value=zeros(size(fitter.bs_casadi{i}.CPoints));
        result=[result,...
                {createStruct(name_crtl_pts,   fitter.ctrl_pts{i} , crtl_pts_value)},...
                {createStruct(name_bbox_inflated, fitter.bbox_inflated{i} , [1;1]  )}]
    end

end

%%
%% ---------------------------------------------------------------------------------------------------------------
%%

function [par_and_init_guess_exprs, par_and_init_guess_names, names_value]=toExprsNamesAndNamesValue(par_and_init_guess)
    par_and_init_guess_exprs=[]; %expressions
    par_and_init_guess_names=[]; %guesses
    names_value={};
    for i=1:numel(par_and_init_guess)
        par_and_init_guess_exprs=[par_and_init_guess_exprs {par_and_init_guess{i}.expression}];
        par_and_init_guess_names=[par_and_init_guess_names {par_and_init_guess{i}.name}];

        names_value{end+1}=par_and_init_guess{i}.name;
        names_value{end+1}=double2DM(par_and_init_guess{i}.value); 
    end
end

%%
%% ---------------------------------------------------------------------------------------------------------------
%%

function const_p_dyn_limits = addDynLimConstraints(const_p_dyn_limits, sp, basis, v_max_n, a_max_n, j_max_n)

    const_p_dyn_limits=[const_p_dyn_limits sp.getMaxVelConstraints(basis, v_max_n)];      %Max vel constraints (position)
    const_p_dyn_limits=[const_p_dyn_limits sp.getMaxAccelConstraints(basis, a_max_n)];    %Max accel constraints (position)
    const_p_dyn_limits=[const_p_dyn_limits sp.getMaxJerkConstraints(basis, j_max_n)];     %Max jerk constraints (position)

end

%%
%% ---------------------------------------------------------------------------------------------------------------
%%

%Taken from https://gist.github.com/jgillis/9d12df1994b6fea08eddd0a3f0b0737f
%See discussion at https://groups.google.com/g/casadi-users/c/1061E0eVAXM/m/dFHpw1CQBgAJ
function [stats] = get_stats(f)
    dep = 0;
    % Loop over the algorithm
    for k=0:f.n_instructions()-1
  %      fprintf("Trying with k= %d\n", k)
      if f.instruction_id(k)==casadi.OP_CALL
        fprintf("Found k= %d\n", k)
        d = f.instruction_MX(k).which_function();
        if d.name()=='solver'
            my_file=fopen('./casadi_generated_files/acslam_index_instruction.txt','w'); %Overwrite content
          fprintf(my_file,'%d\n',k);
          dep = d;
          break
        end
      end
    end
    if dep==0
      stats = struct;
    else
      stats = dep.stats(1);
    end
  end

%%
%% ---------------------------------------------------------------------------------------------------------------
%%

function a=createStruct(name,expression,value)
    a.name=name;
    a.expression=expression;
    a.value=value;
end


%%
%% ---------------------------------------------------------------------------------------------------------------
%%

function result=substituteWithSolution(expression, all_var_solved, all_params_and_init_guesses)

    import casadi.*
      result=zeros(size(expression));
      for i=1:size(expression,1)
          for j=1:size(expression,2)
              
                  tmp=expression(i,j);
                  
                  %Substitute FIRST the solution [note that this one needs to be first because in all_params_and_init_guesses we have also the initial guesses, which we don't want to use]
                  for ii=1:numel(all_var_solved)
                      if(isPureParamOrVariable(all_var_solved{ii}.expression)==false) 
                          continue;
                      end
                      tmp=substitute(tmp,all_var_solved{ii}.expression, all_var_solved{ii}.value);
                  end
                  
                   %And THEN substitute the parameters
                  for ii=1:numel(all_params_and_init_guesses)
                      tmp=substitute(tmp,all_params_and_init_guesses{ii}.expression, all_params_and_init_guesses{ii}.value);
                  end
              
              result(i,j)=convertMX2Matlab(tmp);
          end
      end
  
  
  end

%%
%% ---------------------------------------------------------------------------------------------------------------
%%

%This checks whether it is a pure variable/parameter in the optimization (returns true) or not (returns false). Note that with an expression that is a combination of several double/variables/parameters will return false
function result=isPureParamOrVariable(expression)
    result=expression.is_valid_input();
end