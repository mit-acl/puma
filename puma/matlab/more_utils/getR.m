function R = getR(sp, sy, t_n, alpha, b_T_c, pos_center_obs, thetax_half_FOV_deg, fov_depth, max_variance, infeasibility_adjust)

    %%
    %% Get R depending on FOV 
    %%

    %% get pos, accel, and yaw from spline
    yaw = sy.getPosT(t_n);
    w_t_b = sp.getPosT(t_n); %Translation between the body and the world frame
    a = sp.getAccelT(t_n)/(alpha^(2));
        
    %Definition 3 (hopf fibration) in the Panther paper table 3
    qpsi=[cos(yaw/2), 0, 0, sin(yaw/2)]; %Note that qpsi has norm=1 (qyaw)
    qabc=qabcFromAccel(a,9.81);
    q=multquat(qabc,qpsi); %Note that q is guaranteed to have norm=1
    w_R_b=toRotMat(q); %Rotation between the body and the world frame
    
    w_T_b=[w_R_b w_t_b; zeros(1,3) 1];
    c_T_b=invPose(b_T_c);
    b_T_w=invPose(w_T_b);
    
    % Take the center of the obstacle and get the position of the obstacle in the world frame
    w_fevar=pos_center_obs;
    
    c_P=c_T_b*b_T_w*[w_fevar;1]; %Position of the feature (the center of the obstacle) in the camera frame
    
    %FOV is a cone:  (See more possible versions of this constraint at the end of this file) (inFOV in Panther paper table 2)
    is_in_FOV_tmp=-cos(thetax_half_FOV_deg*pi/180.0) + (c_P(1:3)'/norm(c_P((1:3))))*[0;0;1]; % Constraint is is_in_FOV1>=0
    R = diag(max_variance) * 1 / (1+is_in_FOV_tmp + infeasibility_adjust) * (1-is_in_FOV_tmp);

    % this approach is not working (maybe because of the if_else)
    % R = if_else(is_in_FOV_tmp(1) > 0.0, R_small, R_large); % Constraint is is_in_FOV1>=0
    % is_in_FOV = if_else(c_P(3) < fov_depth, true, false); %If the obstacle is farther than fov_depth, then it is not in the FOV (https://www.mathworks.com/matlabcentral/answers/714068-cannot-convert-logical-to-casadi-sx)
    % is_in_FOV_tmp = is_in_FOV_tmp * (1.5*fov_depth - c_P(3))/(1.5*fov_depth);

    % R = if_else(is_in_FOV, R_small, R_large);
end