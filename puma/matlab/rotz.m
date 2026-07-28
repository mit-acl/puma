function R = rotz(a)
% Local shim for Phased Array System Toolbox rotz (toolbox not installed).
% Right-handed rotation about Z by 'a' DEGREES. Matches MATLAB's rotz.
c = cosd(a); s = sind(a);
R = [c -s 0; s c 0; 0 0 1];
end
