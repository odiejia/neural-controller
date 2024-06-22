clear
clc

load('history.mat')
%mesh(X1_mat(:,:),X2_mat(:,:),pdf_mat(:,:))
figure(1)
set(gcf,'position',[300 300 400 300])
semilogy(loss,'r')
hold on 
semilogy(lossf,'b')
hold on 
% semilogy(lossu)
% hold on
semilogy(loss_sample,'m')


legend('total loss','residual','sample','FontName','Times New Roman','FontWeight','bold')
xlabel('×100 epochs','FontName','Times New Roman','FontWeight','bold')
ylabel('loss','FontName','Times New Roman','FontWeight','bold')
saveas(figure(1),'loss.png')


figure(2)
set(gcf,'position',[300 300 400 300])
plot(x1_f_adaptive(1:100,:),x2_f_adaptive(1:100,:),'b.')
hold on
plot(x1_f_adaptive(101:170,:),x2_f_adaptive(101:170,:),'k.')
hold on
plot(x1_f_adaptive(171:200,:),x2_f_adaptive(171:200,:),'rx')