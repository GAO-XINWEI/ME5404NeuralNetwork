clear all
close all
% y1 = 1 / k;
% y2 = 100 / (100 + k);
% y3 = (1 + log(k)) / k;
% y4 = (1 + 5 * log(k)) / k;
% y5 = exp(-0.001*k)
clear;
clc;
k=[1:200];
% y1 = 1 ./ k;
% y2 = 100 ./ (100 + k);
% y3 = (1 + log(k)) ./ k;
% y4 = (1 + 5 * log(k)) ./ k;
y5 = exp(-0.001*k);
y6 = exp(-0.01*k);
y7 = exp(-0.025*k);
y8 = exp(-0.1*k);
% y7 = init_rate_6./(1+0.9.*k);
% y7 = init_rate_7.*decay_rate_7.^(golabl_step_7./k);
% y8=init_rate_8.*exp(-decay_rate_8./k);

figure()
hold on
% plot(k,y1)
% plot(k,y2)
% plot(k,y3)
% plot(k,y4)
plot(k,y5)
plot(k,y6)
plot(k,y7)
plot(k,y8)
ylim([0,2.5]);
hold off
% legend('1/k','100 / (100 + k)','(1 + log(k)) / k','(1 + 5 * log(k)) / k')
legend('exp(-0.001*k)','exp(-0.01*k)','exp(-0.05*k)','exp(-0.1*k)')

% legend('1/k','100 / (100 + k)','(1 + log(k)) / k','(1 + 5 * log(k)) / k','exp(-0.001*k)','exp(-0.01*k)','exp(-0.1*k)')