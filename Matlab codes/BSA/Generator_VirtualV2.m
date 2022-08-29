clear;
clc;
%% /虚拟数据制作/
n=5e4;
nSlice=0;
input = zeros(n, 1000, 7);
Path='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\data\Project_BSA';
load([Path,'\','150.mat'],'MeanV','StdV');
for DataIndex=1:n
    tempo=strcat(num2str(DataIndex/n*100),'%');
    disp(tempo);
    sample=3;
    Signal=Simulation_SeismicWave(sample);
%     Signal=Simulation_SeismicWave_Uniform(sample, DataIndex);
    Signal(:,1)=0.85.*Signal(:,1);
    Signal(:,2)=1.*Signal(:,2);
    Signal(:,3)=0.65.*Signal(:,3);
    Trans=15000;TransMap=[linspace(0,1,Trans).^1.1]';
    Signal(1:Trans,:)=kron(ones(1,size(Signal,2)),TransMap).*Signal(1:Trans,:);
    Tstep=1e-4;
    T=5+2*Tstep;
    Nt=round(T/Tstep);
    Time=[Tstep:Tstep:T]';
    Signal=Signal(1:length(Time),:);
    for i=1:size(Signal,2)
    [disint,~] = IntFcn(Signal(:,i)', Time', Tstep, 2);
    Signal_Dis(:,i)=disint'-disint(1);
    end
    Signal_Vel=(Signal_Dis(3:end,:)-Signal_Dis(1:end-2,:))./(2*Tstep);
    Signal_Acl=(Signal_Dis(3:end,:)-2.*Signal_Dis(2:end-1,:)+Signal_Dis(1:end-2,:))./(Tstep^2);
    Signal_Dis=Signal_Dis(2:end-1,:);
    Time=Time(2:end-1);
    T=T-2*Tstep;
    Nt=round(T/Tstep);
    DataPack=[Time(1:Nt,:) Signal_Dis(1:Nt,:) Signal_Vel(1:Nt,:)];
    DataPack=DataPack(1:10:size(DataPack,1),:);
    DataPack=DataPack(nSlice+1:end,:);
    for i=1:size(DataPack,2)
        DataPack(:,i)=(DataPack(:,i)-MeanV(i))./StdV(i);                   % 网络输入数据归一化
    end
    input(DataIndex,:,:)=DataPack(1:5:size(DataPack,1),:);
    A=1;
    clear Signal_Dis
    
end
save([Path,'\PDEM\VirtualData_',num2str(n),'.mat'],'input','-v7.3');
