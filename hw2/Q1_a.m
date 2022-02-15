clear;
close all;
x0 = -0.5:0.1:1.2;
y0 = -0.5:0.1:1.2;
[xx, yy] = meshgrid(x0, y0);
zz = (1 - xx).^2 + 100*(yy - xx.^2).^2;

figure();
hold on;
plot3(1,1,0,'*r');
surf(xx, yy, zz, 'FaceAlpha', 0.5);
% shading interp
contour(xx, yy, zz, 20);
xlabel('x'), ylabel('y'); zlabel('z')
hold off;
