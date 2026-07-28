function R = rotx(a)
% Local shim for Phased Array System Toolbox rotx (toolbox not installed).
% Right-handed rotation about X by 'a' DEGREES. Matches MATLAB's rotx.
c = cosd(a); s = sind(a);
R = [1 0 0; 0 c -s; 0 s c];
end
