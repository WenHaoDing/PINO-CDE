clear;
clc;
Path='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\data\Project_VTCD';
load([Path,'\','Big_MCM_10000.mat'],'StdV','MeanV');
load([Path,'\','Weights_Big_MCM_10000_Medium.mat']);
%% 制作虚拟数据
n=10000;
s=14;                                                                      
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
for i=1:n
    i
    TemParaPack=MCParaPack(i,:);
    [input_tem]=VTCD_VirtualDataGeneration(TemParaPack);
    for Column=1:size(input_tem,2)
        input_tem(:,Column)=(input_tem(:,Column)-MeanV(Column))./StdV(Column);
    end
    if i==1
        input=zeros(n,size(input_tem,1),size(input_tem,2));
    end
    input(i,:,:)=input_tem;
end
save([Path,'\','VirtualData_10000V2.mat'],'input','-v7.3');

%% 制作平均权重
Path='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\data\Project_VTCD';
load([Path,'\','Weights_10000V2.mat']);
Weights=mean(Weights);
save([Path,'\','Weights_VirtualV2.mat'],'Weights','-v7.3');

    
    