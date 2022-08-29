clear;
clc;
for Pick=1:5
%% Part0.制作UD，选择空间函数点集
n=10000;
s=14;                                                                       %仅考虑车辆系统自身参数
Fn=20000;
% [UD_n]=Function_PglpUD_NoMeasure(n,s);                                   %生成当前参数控制下的PGLP设计集合
% UD_n=UD_n-0.5;
% save('19997_14.mat','UD_n');
load('19997_14.mat','UD_n');
% Shadow_UD_n=(UD_n+0.5);
%% Part0-2.顶控台
Batchsize=2000;
N_Start=1:Batchsize:n;
N_End=N_Start+Batchsize;N_End(end)=n;
% Pick=1;
%% Part1.制作参数包
% Mc_Interval = [38500 52000];
% Jc_Interval = [2.31e6 2.966e6];
% Mt_Interval = [2200 3200];
% Jt_Interval = [2320 3605];
% Mw_Interval = [1350 1900];
% Kp_Interval = [1.87e6, 2.14e6];
% Cp_Interval = [1.15e5, 1.25e5];
% Ks_Interval = [8e5, 2.535e6];
% Cs_Interval = [25e4, 2.174e5];
% Lc_Interval = [8.4 9];
% Lt_Interval = [1.15 1.2];
% Vc_Interval = [250 350];
% Kkj_Interval = [3.5e7 4.5e7];
% Ckj_Interval = [4e4 5e4];

Mc_Interval = [0.8 1.2].*38880;
Jc_Interval = [0.8 1.2].*1.91e6;
Mt_Interval = [0.8 1.2].*3060;
Jt_Interval = [0.8 1.2].*3200;
Mw_Interval = [0.8 1.2].*1517;
Kp_Interval = [0.8 1.2].*1.772e6;
Cp_Interval = [0.8 1.2].*2e4;
Ks_Interval = [0.8 1.2].*4.5e5;
Cs_Interval = [0.8 1.2].*2e4;
Lc_Interval = [0.8 1.2].*9;
Lt_Interval = [0.8 1.2].*1.2;
Vc_Interval = [300 350];
Kkj_Interval = [3.5e7 4.5e7];
Ckj_Interval = [4e4 5e4];
ParaPack=zeros(n,s);
for i=1:n
    ParaPack(i,1)=Mc_Interval(1)+(Mc_Interval(2)-Mc_Interval(1))*(UD_n(i,1)/n);
    ParaPack(i,2)=Jc_Interval(1)+(Jc_Interval(2)-Jc_Interval(1))*(UD_n(i,2)/n);
    ParaPack(i,3)=Mt_Interval(1)+(Mt_Interval(2)-Mt_Interval(1))*(UD_n(i,3)/n);
    ParaPack(i,4)=Jt_Interval(1)+(Jt_Interval(2)-Jt_Interval(1))*(UD_n(i,4)/n);
    ParaPack(i,5)=Mw_Interval(1)+(Mw_Interval(2)-Mw_Interval(1))*(UD_n(i,5)/n);
    ParaPack(i,6)=Kp_Interval(1)+(Kp_Interval(2)-Kp_Interval(1))*(UD_n(i,6)/n);
    ParaPack(i,7)=Cp_Interval(1)+(Cp_Interval(2)-Cp_Interval(1))*(UD_n(i,7)/n);
    ParaPack(i,8)=Ks_Interval(1)+(Ks_Interval(2)-Ks_Interval(1))*(UD_n(i,8)/n);
    ParaPack(i,9)=Cs_Interval(1)+(Cs_Interval(2)-Cs_Interval(1))*(UD_n(i,9)/n);
    ParaPack(i,10)=Lc_Interval(1)+(Lc_Interval(2)-Lc_Interval(1))*(UD_n(i,10)/n);
    ParaPack(i,11)=Lt_Interval(1)+(Lt_Interval(2)-Lt_Interval(1))*(UD_n(i,11)/n);
    ParaPack(i,12)=Vc_Interval(1)+(Vc_Interval(2)-Vc_Interval(1))*(UD_n(i,12)/n);
    ParaPack(i,13)=Kkj_Interval(1)+(Kkj_Interval(2)-Kkj_Interval(1))*(UD_n(i,13)/n);
    ParaPack(i,14)=Ckj_Interval(1)+(Ckj_Interval(2)-Ckj_Interval(1))*(UD_n(i,14)/n);
end
%% 生成均匀分布不相干的数组
MCParaPack=zeros(n,s);
MCParaPack(:,1)=Mc_Interval(1)+(Mc_Interval(2)-Mc_Interval(1))*rand(n,1);
MCParaPack(:,2)=Jc_Interval(1)+(Jc_Interval(2)-Jc_Interval(1))*rand(n,1);
MCParaPack(:,3)=Mt_Interval(1)+(Mt_Interval(2)-Mt_Interval(1))*rand(n,1);
MCParaPack(:,4)=Jt_Interval(1)+(Jt_Interval(2)-Jt_Interval(1))*rand(n,1);
MCParaPack(:,5)=Mw_Interval(1)+(Mw_Interval(2)-Mw_Interval(1))*rand(n,1);
MCParaPack(:,6)=Kp_Interval(1)+(Kp_Interval(2)-Kp_Interval(1))*rand(n,1);
MCParaPack(:,7)=Cp_Interval(1)+(Cp_Interval(2)-Cp_Interval(1))*rand(n,1);
MCParaPack(:,8)=Ks_Interval(1)+(Ks_Interval(2)-Ks_Interval(1))*rand(n,1);
MCParaPack(:,9)=Cs_Interval(1)+(Cs_Interval(2)-Cs_Interval(1))*rand(n,1);
MCParaPack(:,10)=Lc_Interval(1)+(Lc_Interval(2)-Lc_Interval(1))*rand(n,1);
MCParaPack(:,11)=Lt_Interval(1)+(Lt_Interval(2)-Lt_Interval(1))*rand(n,1);
MCParaPack(:,12)=Vc_Interval(1)+(Vc_Interval(2)-Vc_Interval(1))*rand(n,1);
MCParaPack(:,13)=Kkj_Interval(1)+(Kkj_Interval(2)-Kkj_Interval(1))*rand(n,1);
MCParaPack(:,14)=Ckj_Interval(1)+(Ckj_Interval(2)-Ckj_Interval(1))*rand(n,1);

DataStore=zeros(n,1+4+14+30,5000);
Number=N_End(Pick)-N_Start(Pick);
Batch_Input=zeros(Number,5000,1+4+14);
Batch_Output=zeros(Number,5000,30+4);
FlagStore=zeros(Number,1);
for ib=1:Number
    Index=N_Start(Pick)+ib-1;
% TemParaPack=ParaPack(Index,:);
TemParaPack=MCParaPack(Index,:);
[A,B,Flag]=VTCD_DataGeneration(TemParaPack);
% [A,B,Flag]=VTCD_DataGeneration_RK4(TemParaPack);
Batch_Input(ib,:,:)=A;
Batch_Output(ib,:,:)=B;
FlagStore(ib)=Flag;
if Flag==1
    disp(['出现无法收敛情况:Epoch',num2str(ib)]);
else
    disp(['完成VTCD计算:Epoch',num2str(ib)]);
end
end
EmitSe=find(FlagStore==1);
Batch_Input(EmitSe,:,:)=[];
Batch_Output(EmitSe,:,:)=[];
Path='G:';
FileName=['MCM',num2str(Pick),'.mat'];FileName=[Path,'\',FileName];
save(FileName,'Batch_Input','Batch_Output','FlagStore');
end



