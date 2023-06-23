close all; clc;clear;
doSetup();
import casadi.*
const_p={};    
const_y={};

opti = casadi.Opti();

fitter.deg_pos = 3; %The degree of the fit past obstacle trajectory
fitter.num_seg = 7; %The number of segments in the fit in the past obstacle trajectory
fitter.dim_pos = 3; %The dimension of the fit past obstacle (3 for R3)
fitter.num_samples = 20; %The number of samples used to fit the past obstacle trajectory
fitter_num_of_cps = fitter.num_seg + fitter.deg_pos; %The number of control points of fit past obstacle trajectory (B-spline)

fitter.ctrl_pts = [
 2.48 -0.26  0.47
  2.59 -0.441  0.334
  3.05 -0.847  0.238
 4.47  -1.6 0.626
 6.62 -2.48  1.55
  9.5 -3.51     3
 13.1 -4.67  4.98
 17.5 -5.98  7.49
 20.8 -6.94  9.52
 22.6 -7.44  10.6
]';

fitter.uncertainty_ctrl_pts = [
0.00865 0.00732 0.00907
0.0168 0.0155 0.0172
0.0552 0.0539 0.0556
0.181  0.18 0.182
0.381  0.38 0.381
0.662  0.66 0.662
1.03 1.03 1.03
1.51 1.51 1.51
1.91 1.91 1.91
2.13 2.13 2.13
]';

fitter.bbox = [0.814 0.597 0.687];

fitter.bs_casadi = MyCasadiClampedUniformSpline(0,1,fitter.deg_pos,fitter.dim_pos,fitter.num_seg,fitter.ctrl_pts, false);
fitter.uncertainty_bs_casadi = MyCasadiClampedUniformSpline(0,1,fitter.deg_pos,fitter.dim_pos,fitter.num_seg,fitter.uncertainty_ctrl_pts, false);

%The total time of the fit past obstacle trajectory (horizon length[NOTE: This is also the max horizon length of the drone's trajectory])
fitter.total_time = 6.0; %Time from (time at point d) to end of the fitted spline

%Interpolate spline points from both trajectories
num_points = 100;
spline_points = zeros(fitter.dim_pos,num_points);
uncertainty_spline_points = zeros(fitter.dim_pos,num_points);
for i = 1:num_points
    t = (i-1)/(num_points-1);
    spline_points(:,i) = full(fitter.bs_casadi.getPosT(t));
    uncertainty_spline_points(:,i) = full(fitter.uncertainty_bs_casadi.getPosT(t));
end


%3D plot the fit past obstacle trajectory control points
figure(1)
plot3(fitter.ctrl_pts(1,:),fitter.ctrl_pts(2,:),fitter.ctrl_pts(3,:),'o','MarkerSize',10,'LineWidth',2)

%3D Plot the fit past obstacle trajectory
hold on
plot3(spline_points(1,:),spline_points(2,:),spline_points(3,:),'LineWidth',2)
hold off

axis padded
pbaspect([1 1 1])

%3D plot the fit past obstacle trajectory control points
figure(2)
plot3(fitter.uncertainty_ctrl_pts(1,:),fitter.uncertainty_ctrl_pts(2,:),fitter.uncertainty_ctrl_pts(3,:),'o','MarkerSize',10,'LineWidth',2)

%3D Plot the fit past obstacle trajectory
hold on
plot3(uncertainty_spline_points(1,:),uncertainty_spline_points(2,:),uncertainty_spline_points(3,:),'LineWidth',2)
hold off

axis padded
pbaspect([1 1 1])


%Plot the bounding box at each point of the fit past obstacle trajectory
figure(3)
plot3(spline_points(1,:),spline_points(2,:),spline_points(3,:),'LineWidth',2)

%Plot the bounding box at each point of the fit past obstacle trajectory
hold on
for i = 1:num_points
    center = spline_points(:,i)';
    dim = fitter.bbox;

    %Inflate the bbox
    dim = dim + uncertainty_spline_points(:,i)';

    origin = center - dim / 2;
    plotcube(dim, origin, 0.1 ,[1 0 0]);
end
hold off

axis padded
pbaspect([1 1 1])


