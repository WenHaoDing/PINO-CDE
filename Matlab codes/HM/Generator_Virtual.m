clear;
clc;
n=8000;
version=3;
for index=1:n
    index
    for i=1:3
        v=250;
    [g] =Generator_Irre(v);
    Signal(:,i)=g;
    end
    Signal=1e-3.*Signal;
    Trans=1000;TransMap=[linspace(0,1,Trans).^1.1]';%每一个线型需要单独调整此处
    Signal(1:Trans,:)=kron(ones(1,3),TransMap).*Signal(1:Trans,:);
    Tstep=1e-4;
    T=1;
    Nt=round(T/Tstep);
    K1=1e4+(0.5e4)*rand(1);
    C1=1e2+0.5e2*rand(1);
    K2=1e4+(0.5e4)*rand(1);
    C2=1e2+0.5e2*rand(1);
    K=[K1 K2];
    C=[C1 C2];
    MPack = 0.8+0.4*rand(1,2);
    Time=[1e-4:1e-4:Nt*1e-4]';
    DataPack=[Time Signal(1:Nt,:) kron(ones(Nt,1),K) kron(ones(Nt,1),C) kron(ones(Nt,1),MPack)];
    DataPack=DataPack(1:end-1,:);
    DPI=10;
    DataPack=DataPack(1:DPI:size(DataPack,1),:);
    input(index,:,:)=DataPack;
end

load('F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\data\Project_HM\5000_V3.mat','MaxV','MinV','MeanV','StdV');
InDim=10;
MeanV_input=MeanV(1:InDim);
StdV_input=StdV(1:InDim);
for i=1:size(input, 3)
    input(:,:,i)=(input(:,:,i)-MeanV_input(i))/StdV_input(i);
end
Path='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\data\Project_HM';
FileName=['VirtualData_',num2str(n),'_V',num2str(version),'.mat'];FileName=[Path,'\',FileName];
save(FileName,'input','-v7.3');


