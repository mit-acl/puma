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
 3.08 0.886  1.26
 3.28 0.886  1.46
 3.71 0.883  1.91
 4.47 0.875  2.72
 5.36 0.864  3.66
 6.37 0.849  4.72
 7.5 0.83 5.92
 8.76 0.808  7.25
 9.68 0.791  8.22
 10.2 0.782  8.73
]';

fitter.uncertainty_ctrl_pts = [
0.000293 0.000283 0.000401
 0.00054  0.00052 0.000738
0.000845 0.000815  0.00115
0.0261 0.0251 0.0356
0.128 0.124 0.175
0.388 0.374  0.53
0.914 0.881  1.25
1.84 1.78 2.52
2.84 2.74 3.88
3.46 3.34 4.74
]';

fitter.bbox = [0.903 0.661 0.566];

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


