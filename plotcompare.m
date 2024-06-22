clear
clc

%%
%data control
load('data_target')
load('pdf.mat')
load('control_1.mat')
load('control_2.mat')
%data target
load('data_target.mat')

%% data control
[Num_X2,Num_X1]=size(pdf_mat);
x1_arr=reshape(X1_mat(1,:),[],1);
x2_arr=reshape(X2_mat(:,1),[],1);

p = trapz(x2_arr,pdf_mat,1);
normal = trapz(x1_arr,reshape(p,[],1),1);
pdf_mat = pdf_mat/normal;

pdf_x1=trapz(x2_arr,pdf_mat,1);
pdf_x2=trapz(x1_arr,pdf_mat,2);

%% joint pdf
figure(1)

subplot(1,3,2)
set(gcf,'position',[200 300 1500 300])
set(gca,'fontsize',12,'fontname','timesnewroman')
%subplot(1,2,2)
surf(X1_mat(:,:),X2_mat(:,:),pdf_mat(:,:))
shading interp   
axis([0.4 0.8 0.8 1.2])
xlabel('x')
ylabel('y')
%zlabel('$$','interpreter','latex')
title('(b) p_{control}(x, y)')
colormap jet
view(0,90)
colorbar
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(1,3,1)
%set(gcf,'position',[300 300 600 400])
surf(q1_arr,p1_arr,pdf_target)
axis([0.4 0.8 0.8 1.2])
shading interp   
xlabel('x')
ylabel('y')
%zlabel('$$','interpreter','latex')
title('(a) p_{target}(x, y)')
view(0,90)
colorbar
%err=(pdf_mat(:,:)-pdf_control);
%imagesc(q1_arr,p1_arr,err)
%colorbar
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(1,3,3)
err=abs(pdf_mat(:,:)-pdf_target);
surf(q1_arr,p1_arr,err)
shading interp
xlabel('x')
ylabel('y')
%zlabel('$$','interpreter','latex')
axis([0.4 0.8 0.8 1.2])
title('(c) absolute error')
view(0,90)
colorbar

set(gca,'LooseInset',get(gca,'TightInset'))
%set(gca, 'LooseInset', [0,0,0,0]);
%% marginal pdf
figure(2)
set(gcf,'position',[200 200 1000 350])
set(gca,'looseInset',[0 0 0 0])
fig = figure(2);
left_color = [0 0 0];
right_color = [1 0.2 1];
set(fig,'defaultAxesColorOrder',[left_color; right_color]);


subplot(1,2,1)
yyaxis left
set(gca,'ycolor','r');
plot(q1_arr(1:1:end),pdf_q1(1:1:end),'ro')
hold on
plot(q1_arr(1:1:end),pdf_q1(1:1:end),'r-')
axis([0.4 0.8 0 6])
xlabel('x','fontname','timesnewroman')
ylabel('p(x)')


yyaxis right
set(gca,'yscale','log')
err1=abs(pdf_x1-pdf_q1);
semilogy(q1_arr(1:1:end),err1(1:1:end),'m*--')
axis([0.4 0.8 0 2e-1])
xlabel('x')
ylabel('absolute error')
legend('p_{target}(x)','p_{control}(x)','error','box','off','location','northwest')
%zlabel('$$','interpreter','latex')
title('(a)')
set(gca,'linewidth',1.2)
%%
subplot(1,2,2)
yyaxis left
plot(p1_arr(1:1:end),pdf_p1(1:1:end),'bs')
hold on
plot(x2_arr,pdf_x2,'b-')
axis([0.8 1.2 0 6])
set(gca,'ycolor','b');
xlabel('y')
ylabel('p(y)')

yyaxis right
set(gca,'yscale','log')
err2=abs(pdf_x2-pdf_p1);
semilogy(p1_arr(1:1:end),err2(1:1:end),'m*--')
axis([0.8 1.2 0 2e-1])
xlabel('y')
ylabel('absolute error')
%zlabel('$$','interpreter','latex')
title('(b)')
legend('p_{target}(y)','p_{control}(y)','error','box','off','location','northwest')
set(gca,'linewidth',1.2)
%% control force
figure(3)
set(gcf,'position',[200 300 1500 300])
subplot(1,3,2)
surf(X1_mat(:,:),X2_mat(:,:),control_1_mat(:,:))
shading interp
axis([0.4 0.8 0.8 1.2])
xlabel('x')
ylabel('y')
%zlabel('$$','interpreter','latex')
title('(b) control1_{DNN}')
view(0,90)
colorbar
colormap jet
caxis([-0.3 0.4])

subplot(1,3,1)
surf(Q1_mat(:,:),P1_mat(:,:),u1(:,:))
shading interp
axis([0.4 0.8 0.8 1.2])
xlabel('x')
ylabel('y')
%zlabel('$$','interpreter','latex')
title('(a) analytical control')
view(0,90)
colorbar
caxis([-0.3 0.4])

subplot(1,3,3)
surf(Q1_mat(:,:),P1_mat(:,:),abs(u1(:,:)-control_1_mat(:,:)))
shading interp
axis([0.4 0.8 0.8 1.2])
xlabel('x')
ylabel('y')
%zlabel('$$','interpreter','latex')
title('(c) absolute error')
view(0,90)
colorbar

%%
figure(4)
set(gcf,'position',[200 300 1500 300])
subplot(1,3,2)
surf(X1_mat(:,:),X2_mat(:,:),control_2_mat(:,:))
shading interp
axis([0.4 0.8 0.8 1.2])
xlabel('x')
ylabel('y')
%zlabel('$$','interpreter','latex')
title('(b) control2_{DNN}')
view(0,90)
colorbar
colormap jet
caxis([-0.4 0.4])
subplot(1,3,1)
surf(Q1_mat(:,:),P1_mat(:,:),u2(:,:))
shading interp
axis([0.4 0.8 0.8 1.2])
xlabel('x')
ylabel('y')
%zlabel('$$','interpreter','latex')
title('(a) analytical control')
view(0,90)
colorbar
caxis([-0.4 0.4])
subplot(1,3,3)
surf(Q1_mat(:,:),P1_mat(:,:),abs(u2(:,:)-control_2_mat(:,:)))
shading interp
axis([0.4 0.8 0.8 1.2])
xlabel('x')
ylabel('y')
%zlabel('$$','interpreter','latex')
title('(c) absolute error')
view(0,90)
colorbar

saveas(figure(1),'joint_spdf.png')
saveas(figure(2),'marginal_spdf.png')
saveas(figure(3),'control1_force.png')
saveas(figure(4),'control2_force.png')