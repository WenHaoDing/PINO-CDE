clear;
clc;
Path='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\checkpoints\BSARunner\R1';
Pick=1;
dt=5e-3;
load('F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\data\Project_BSA\150.mat','MeanV','StdV');
% LossHistory=importdata([Path,'\','LossHistory.txt']);
for index=1:2:499
    Data=importdata([Path,'\','Performance',num2str(index),'.txt']);
    Signal=Data(:,[Pick Pick+200]);
    Signal(:,1)=Signal(:,1)*StdV(7+Pick)+MeanV(7+Pick);
    Signal(:,2)=Signal(:,2)*StdV(7+Pick)+MeanV(7+Pick);
    First_Derivative=(Signal(3:end,:)-Signal(1:end-2,:))/(2*dt);
    Second_Derivative=(Signal(3:end,:)-2.*Signal(2:end-1,:)+Signal(1:end-2,:))/(dt^2);
    Time=[dt:dt:length(Signal)*dt];
    figure(1);
    set(gcf,'unit','normalized','position',[0.1,0.1,0.6,0.6]);
    subplot(3,1,1)
    plot(Time,Signal(:,1),'b','LineWidth',1.2);hold on;
    plot(Time,Signal(:,2),'r','LineWidth',1.2,'LineStyle','--');hold off;
    title(['Solution',num2str(Pick),' at epoch ',num2str(index)]);
    subplot(3,1,2)
    plot(Time(2:end-1),First_Derivative(:,1),'b','LineWidth',1.2);hold on;
    plot(Time(2:end-1),First_Derivative(:,2),'r','LineWidth',1.2,'LineStyle','--');hold off;
    title('First Derivative');
    subplot(3,1,3)
    plot(Time(2:end-1),Second_Derivative(:,1),'b','LineWidth',1.2);hold on;
    plot(Time(2:end-1),Second_Derivative(:,2),'r','LineWidth',1.2,'LineStyle','--');hold off;
    title('Second Derivative');
    pause(0.05);
    DataPack=[Signal(2:end-1,:) First_Derivative Second_Derivative];
end
Time=linspace(0.5,5,size(First_Derivative,1))';
Pic=[1e3.*Signal(2:end-1,:) First_Derivative Second_Derivative];

% Compound_Data=[Data1(:,1:6) Data2(:,1:6)];
