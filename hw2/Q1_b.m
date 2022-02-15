clear;
close all;
x0 = -0.5:0.1:1.2;
y0 = -0.5:0.1:1.2;
[xx, yy] = meshgrid(x0, y0);
zz = (1 - xx).^2 + 100*(yy - xx.^2).^2;

x = rand;% 0.11;
y = rand;% 0.92;
max_iter = 1e5;
max_err = 1e-4;
lr = 2.5e-3;

for iter = 1:max_iter
    dzdx = 2*x - 400*x*(- x^2 + y) - 2;
    dzdy = - 200*x^2 + 200*y;
    x = x - lr * dzdx;
    y = y - lr * dzdy;
    z = (1 - x)^2 + 100*(y - x^2)^2;
    rollout(iter, :) = [x y z];
    if z-0 < max_err
        fprintf('Converge to minima (%.4f, %.4f) at iteration %d!\n', x, y, iter);
        break
    end
end

figure();
hold on;
plot3(rollout(:, 1),rollout(:, 2),rollout(:, 3),'.r');
plot3(rollout(:, 1),rollout(:, 2),rollout(:, 3),'-g');
plot3(rollout(1, 1),rollout(1, 2),rollout(1, 3),'ob');
plot3(rollout(iter, 1),rollout(iter, 2),rollout(iter, 3),'oy');
plot(rollout(:, 1), rollout(:, 2), 'linewidth', 2);
scatter(rollout(1, 1), rollout(1, 2), 'filled');
scatter(rollout(iter, 1), rollout(iter, 2), 'filled');
surf(xx, yy, zz, 'FaceAlpha', 0.5);
% shading interp
contour(xx, yy, zz, 20);
xlabel('x'), ylabel('y'); zlabel('z')
hold off;

figure();
semilogx(rollout(:, 3), 'linewidth', 2);
xlabel('Iteration'), ylabel('z');
