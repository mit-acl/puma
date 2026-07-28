function R = roty(a)
% Local shim for Phased Array System Toolbox roty (toolbox not installed).
% Right-handed rotation about Y by 'a' DEGREES. Matches MATLAB's roty.
c = cosd(a); s = sind(a);
R = [c 0 s; 0 1 0; -s 0 c];
end
