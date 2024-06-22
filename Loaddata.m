close all
clear
clc

%% parameter
miu1=0.6
miu2=1.0
% sigma1=0.01
% sigma2=0.01
sigma1=0.1
sigma2=0.1

step =0.01;
q1_arr =0.2:step:1.0;
p1_arr = 0.6:step:1.4;

[Q1_mat,P1_mat]=meshgrid(q1_arr,p1_arr);
%% exact pdf
phi_mat =  -((Q1_mat-miu1).^2/(2*sigma1^2)+(P1_mat-miu2).^2/(2*sigma2^2));

exp_phi_mat = exp(phi_mat);

C_const = sum(sum(exp_phi_mat))*step*step;

pdf=exp_phi_mat/C_const;

mesh(Q1_mat,P1_mat,pdf)

phi_target = -((Q1_mat-miu1).^2/(2*sigma1^2)+(P1_mat-miu2).^2/(2*sigma2^2));
exp_phi_target = exp(phi_target);

C_const_target = sum(sum(exp_phi_target))*step*step;
pdf_target = exp_phi_target/C_const_target;

pdf_q1=trapz(p1_arr, pdf_target,1);
pdf_p1=trapz(q1_arr, pdf_target,2);

save data_transient_PINN I1_arr I2_arr pdfI1I2 pdfI1 pdfI2

load('F:\博士研究内容2024\2024-1-10\2-7\Chemical reaction systems\points400+ada+lhs\xy.mat');  % 加载数据
save('F:\博士研究内容2024\2024-1-10\2-7\Chemical reaction systems\points400+ada+lhs\xy.txt', 'xy', '-ascii');  % 保存为文本文件

