clear;
clc;
Path='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\data\Project_BSA';
FileName='150.mat';FileName=[Path,'\',FileName];
load(FileName,'input','output','MeanV','StdV','MaxV','MinV');
n=size(input,1);
dt=0.005;
DOF_flex=200;
Weights=zeros(n,DOF_flex);
DOF_excit=6;                                                               %底座激励，包括位移和速度
nSlice=40;
M=1;
K = 5e6;
C = 1e6;
K_Contact=1e4;
ConnectNum=3;
alpha_k=0.01;
alpha_c=0.01;

[~,~,~,~,~,~ ,~,Nodes,~,...
 ~,W2,Eigens2,~]=Subroutine_FlexBodyImport(1);
[TrackDOFs,Nloc]=Subroutine_InteractionSearch(0.001,Nodes);
W2_CX = W2(TrackDOFs(:,1),:);
W2_CY = W2(TrackDOFs(:,2),:);
W2_CZ = W2(TrackDOFs(:,3),:);
W2_G = [W2_CX;W2_CY;W2_CZ];
DataType='diff';
r = 0.1;                                                                  %误差容忍率

for DataIndex=1:n
    DataIndex
%% 抽取信号
    Data=[squeeze(input(DataIndex,:,:)) squeeze(output(DataIndex,:,:))];
    Data=Data(nSlice+1:end,:);
    for i=1:size(MeanV, 2)
        Data(:,i)=Data(:,i)*StdV(i)+MeanV(i);
    end
    E=Data(2:end-1,2:7);
    U_flex=Data(:,8:8+DOF_flex-1);
%     dU_flex=Data(2:end-1,20:20+DOF_flex-1);
%     ddU_flex=Data(2:end-1,35:35+DOF_flex-1);
%     dU_rigid=Data(2:end-1,59:59+DOF_rigid-1);
%     ddU_rigid=Data(2:end-1,68:68+DOF_rigid-1);
%% 计算差分张量
switch DataType
    case 'diff'
        dU_flex = (U_flex(3:end,:)-U_flex(1:end-2,:))./(2*dt);
        ddU_flex = (U_flex(3:end,:)-2.*U_flex(2:end-1,:)+U_flex(1:end-2,:))./(dt^2);
        U_flex = U_flex(2:end-1,:);
    case 'real'
        dU_flex = Data(2:end-1,8+DOF_flex:8+2*DOF_flex-1);
        ddU_flex = Data(2:end-1,8+2*DOF_flex:end);
        U_flex = U_flex(2:end-1,:);
end
%% 为所有变量执行噪声施加
    List={'dU_flex','ddU_flex','U_flex'};
    dimension = size(U_flex,1);
%     plot(dU_flex(:,1));hold on;
    for  index=1:length(List)
        var=List{index};
    eval(['Column=size(',var,',2);']);
    for column=1:Column
    eval(['Std=std(',var,'(:,column));']);
    eval([var,'(:,column)=',var,'(:,column)+r.*Std.*2.*(rand(dimension,1)-0.5);']);
    end
    end
%     plot(dU_flex(:,1));hold off;
%% 计算PDE损失
    % 仅提取连接控制位置结点的三向位移和速度
    ConnectDis = [kron(ones(1,length(Nloc)),E(:,1))-(W2_CX * U_flex')' ...
                  kron(ones(1,length(Nloc)),E(:,2))-(W2_CY * U_flex')' ...
                  kron(ones(1,length(Nloc)),E(:,3))-(W2_CZ * U_flex')'];   % 所有连接结点的三向相对位移
    ConnectVel = [kron(ones(1,length(Nloc)),E(:,4))-(W2_CX * dU_flex')' ...
                  kron(ones(1,length(Nloc)),E(:,5))-(W2_CY * dU_flex')' ...
                  kron(ones(1,length(Nloc)),E(:,6))-(W2_CZ * dU_flex')']; % 所有连接结点的三向相对速度     
    Fvec = zeros(size(U_flex,1),size(W2,1));
    % 柔性体部分ODE损失
    ForceElementOutput = K.*ConnectDis + C.*ConnectVel;                    % 接触力
    Gvec = ForceElementOutput * W2_G;                                      % 用重组顺序的有效模态信息计算广义模态力
    Du = ddU_flex-Gvec+kron(ones(dimension,1),Eigens2').*U_flex+(kron(ones(dimension,1),Eigens2').*alpha_k+alpha_c).*dU_flex;
    % 激励部分ODE项
    for i=1:size(Du,2)
        Weights_Tem(i,1)=max(Du(50:end,i));                                % 注意切除长度的设置！
    end
    Weights(DataIndex,:)=Weights_Tem';
    A=1;
    clear ConnectDis ConnectVel
end
SaveFileName=['Weights_Medium_',num2str(n),'.mat'];SaveFileName=[Path,'\',SaveFileName];
save(SaveFileName,'Weights','-v7.3');