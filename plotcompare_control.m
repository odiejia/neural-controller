clear
clc
%%
%data control
load('control_1.mat')
load('control_2.mat')
%%
% 定义参数
k1 = 0.5;
k2 = 0.3;
k3 = 0.7;
k4 = 1.0;
A = 1.0;
B = 1.5;
a = 0.6;
b = 1.0;
D11 = 0.01;
D22 = 0.01;
S11 = 0.001;
S22 = 0.001;
sigma_X = 0.08;
sigma_Y = 0.08;


X = X1_mat;
Y = X2_mat;

% 计算u1和u2
u1 = (D11 * a)./ sigma_X^2. - k1 * A + (2 * S22 - k3 * X.^2).* Y + (k2 * B + k4 - D11./ sigma_X^2).* X - S22 * Y.^2 .* (Y - b)./ sigma_Y^2;
u2 = (D22 * b)./ sigma_Y^2. + (k3 * X.^2 - D22./ sigma_Y^2).* Y + (2 * S11 - k2 * B).* X - S11 * X.^2 .* (X - a)./ sigma_X^2;
%%
u1_abs_err = abs(control_1_mat-u1);
u2_abs_err = abs(control_2_mat-u2);
%% joint pdf
figure(1)
set(gcf,'Position',[100, 100, 1200, 400]) % Set the figure size and position

% First subplot
subplot(1,2,1)
set(gca,'FontSize',16,'FontName','Times New Roman') % Sets the font size and type for the current axis
surf(X1_mat(:,:), X2_mat(:,:), control_1_mat(:,:))
shading interp   
axis([0.4 0.8 0.8 1.2])
xlabel('$x_1$','Interpreter','latex','FontSize',16) % Sets the font size and type for the x-axis label
ylabel('$x_2$','Interpreter','latex','FontSize',16) % Sets the font size and type for the y-axis label
title('$\psi _1^I$','Interpreter','latex','FontSize',16) % Sets the title with LaTeX formatting
colormap jet
view(0,90)
colorbar
caxis([-0.3 0.4])
set(gca,'FontSize',16,'FontName','Times New Roman')

% Second subplot
subplot(1,2,2)
set(gca,'FontSize',16,'FontName','Times New Roman') % Sets the font size and type for the current axis
surf(X1_mat(:,:), X2_mat(:,:), control_2_mat(:,:))
shading interp   
axis([0.4 0.8 0.8 1.2])
xlabel('$x_1$','Interpreter','latex','FontSize',16) % Sets the font size and type for the x-axis label
ylabel('$x_2$','Interpreter','latex','FontSize',16) % Sets the font size and type for the y-axis label
title('$\psi _2^I$','Interpreter','latex','FontSize',16) 
colormap jet
view(0,90)
colorbar
caxis([-0.4 0.4])
set(gca,'FontSize',16,'FontName','Times New Roman')

% Save the combined figure
saveas(gcf, 'figure-v1\psi.fig')
saveas(gcf, 'figure-v1\psi.eps')
saveas(gcf, 'figure-v1\psi.png')



figure(2)
set(gcf,'Position',[100, 100, 1200, 400]) % Set the figure size and position

% First subplot
subplot(1,2,1)
set(gca,'FontSize',16,'FontName','Times New Roman') % Sets the font size and type for the current axis
surf(X1_mat(:,:), X2_mat(:,:), u1(:,:))
shading interp   
axis([0.4 0.8 0.8 1.2])
xlabel('$x_1$','Interpreter','latex','FontSize',16) % Sets the font size and type for the x-axis label
ylabel('$x_2$','Interpreter','latex','FontSize',16) % Sets the font size and type for the y-axis label
title('$u_1$','Interpreter','latex','FontSize',16) % Sets the title with LaTeX formatting
colormap jet
view(0,90)
colorbar
caxis([-0.3 0.4])
set(gca,'FontSize',16,'FontName','Times New Roman')

% Second subplot
subplot(1,2,2)
set(gca,'FontSize',16,'FontName','Times New Roman') % Sets the font size and type for the current axis
surf(X1_mat(:,:), X2_mat(:,:), u2(:,:))
shading interp   
axis([0.4 0.8 0.8 1.2])
xlabel('$x_1$','Interpreter','latex','FontSize',16) % Sets the font size and type for the x-axis label
ylabel('$x_2$','Interpreter','latex','FontSize',16) % Sets the font size and type for the y-axis label
title('$u_2$','Interpreter','latex','FontSize',16) 
colormap jet
view(0,90)
colorbar
caxis([-0.4 0.4])
set(gca,'FontSize',16,'FontName','Times New Roman')


figure(3)
set(gcf,'Position',[100, 100, 1200, 400]) % Set the figure size and position

% First subplot
subplot(1,2,1)
set(gca,'FontSize',16,'FontName','Times New Roman') % Sets the font size and type for the current axis
surf(X1_mat(:,:), X2_mat(:,:), u1_abs_err(:,:))
shading interp   
axis([0.4 0.8 0.8 1.2])
xlabel('$x_1$','Interpreter','latex','FontSize',16) % Sets the font size and type for the x-axis label
ylabel('$x_2$','Interpreter','latex','FontSize',16) % Sets the font size and type for the y-axis label
title('$u_1^{err}$','Interpreter','latex','FontSize',16) % Sets the title with LaTeX formatting
colormap jet
view(0,90)
colorbar
caxis([0.0 0.09])
set(gca,'FontSize',16,'FontName','Times New Roman')

% Second subplot
subplot(1,2,2)
set(gca,'FontSize',16,'FontName','Times New Roman') % Sets the font size and type for the current axis
surf(X1_mat(:,:), X2_mat(:,:), u2_abs_err(:,:))
shading interp   
axis([0.4 0.8 0.8 1.2])
xlabel('$x_1$','Interpreter','latex','FontSize',16) % Sets the font size and type for the x-axis label
ylabel('$x_2$','Interpreter','latex','FontSize',16) % Sets the font size and type for the y-axis label
title('$u_2^{err}$','Interpreter','latex','FontSize',16) 
colormap jet
view(0,90)
colorbar
caxis([0.0 0.09])
set(gca,'FontSize',16,'FontName','Times New Roman')

% Save the combined figure
saveas(gcf, 'figure-v1\u.fig')
saveas(gcf, 'figure-v1\u.eps')
saveas(gcf, 'figure-v1\u.png')


