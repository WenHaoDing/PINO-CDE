clear;
clc;
version=4;
Path='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\data\Project_HM';
FileName='5000_V3.mat';FileName=[Path,'\',FileName];
load(FileName,'input','output','MeanV','StdV','MaxV','MinV');
n=size(input,1);
dt=0.001;
Weights=zeros(n,43);
DOF_flex1=15;
DOF_flex2=10;
DOF_flex=DOF_flex1+DOF_flex2;
DOF_rigid1=9;
DOF_rigid2=9;
DOF_rigid=DOF_rigid1+DOF_rigid2;
DOF_excit=3;

M=1;
K=1e4;
C=1e2;
K_Contact=1e4;
ConnectNum1=6;
ConnectNum2=3;
alpha_k=0.001;
alpha_c=0.001;

Type='diff';
r = 0.02;

% [~,~,~,~,~,~ ,~,Nodes,~,...
%  ~,W2,Eigens2,~]=Subroutine_FlexBodyImport(1);
[~,~,~,~,~,~,~,Nodes1,~,~,W21,Eigens21,~,...
 ~,~,~,~,~,~,~,Nodes2,~,W22,Eigens22,~]=Subroutine_FlexBodyImport(1);
[TrackDOFs1,Nloc1]=Subroutine_InteractionSearch(0.001,Nodes1);
[TrackDOFs2,Nloc2]=Subroutine_InteractionSearchV2(0.001,Nodes2);
for DataIndex=1:n
    DataIndex
%% 抽取信号
    Data=[squeeze(input(DataIndex,:,:)) squeeze(output(DataIndex,:,:))];
    for i=1:size(MeanV, 2)
        Data(:,i)=Data(:,i)*StdV(i)+MeanV(i);
    end
    INPUT=Data(1,1:10);
    K1=INPUT(5);K2=INPUT(6);
    C1=INPUT(7);C2=INPUT(8);
    M1=INPUT(9);M2=INPUT(10);
    OUTPUT=Data(:,11:end);
    E=Data(2:end-1,2:4);
    U_flex=OUTPUT(:,1:25);
    U_rigid=OUTPUT(:,26:43);
%% 计算差分张量
if strcmp(Type,'diff')==1
    dU_flex = (U_flex(3:end,:)-U_flex(1:end-2,:))./(2*dt);
    ddU_flex = (U_flex(3:end,:)-2.*U_flex(2:end-1,:)+U_flex(1:end-2,:))./(dt^2);
    U_flex = U_flex(2:end-1,:);
    dU_rigid = (U_rigid(3:end,:)-U_rigid(1:end-2,:))./(2*dt);
    ddU_rigid = (U_rigid(3:end,:)-2.*U_rigid(2:end-1,:)+U_rigid(1:end-2,:))./(dt^2);
    U_rigid = U_rigid(2:end-1,:);
else
    % dU_flex1=OUTPUT(2:end-1,44:58);
%     ddU_flex1=OUTPUT(2:end-1,87:101);
%     dU_flex2=OUTPUT(2:end-1,59:68);
%     ddU_flex2=OUTPUT(2:end-1,102:111);
%     dU_rigid=OUTPUT(2:end-1,69:86);
%     ddU_rigid=OUTPUT(2:end-1,112:129);

%     dU_rigid1=dU_rigid(:,1:9);
%     dU_rigid2=dU_rigid(:.10:end);
    U_flex = U_flex(2:end-1,:);
    dU_flex = OUTPUT(2:end-1,44:68);
    ddU_flex = OUTPUT(2:end-1,87:111);
    U_rigid = U_rigid(2:end-1,:);
    dU_rigid = OUTPUT(2:end-1,69:86);
    ddU_rigid = OUTPUT(2:end-1,112:129);
end
%% 为所有变量执行噪声施加
    List={'dU_flex','ddU_flex','U_flex','dU_rigid','ddU_rigid','U_rigid'};
    dimension = size(U_flex,1);
%     plot(dU_flex(:,1));hold on;
    for index=1:length(List)
        var=List{index};
    eval(['Column=size(',var,',2);']);
    for column=1:Column
    eval(['Std=std(',var,'(:,column));']);
    eval([var,'(:,column)=',var,'(:,column)+r.*Std.*2.*(rand(dimension,1)-0.5);']);
    end
    end
%% 分裂，刚体和柔体分别分成两个对象
U_flex1 = U_flex(:,1:15);
U_flex2 = U_flex(:,16:end);
dU_flex1 = dU_flex(:,1:15);
dU_flex2 = dU_flex(:,16:end);
ddU_flex1 = ddU_flex(:,1:15);
ddU_flex2 = ddU_flex(:,16:end);
U_rigid1 = U_rigid(:,1:9);
U_rigid2 = U_rigid(:,10:end);
dU_rigid1 = dU_rigid(:,1:9);
dU_rigid2 = dU_rigid(:,10:end);
ddU_rigid1 = ddU_rigid(:,1:9);
ddU_rigid2 = ddU_rigid(:,10:end);
% plot(dU_flex(:,1));hold off;
%% 计算PDE损失
    NodeDis1 = W21 * U_flex1';
    NodeVel1 = W21 * dU_flex1';
    NodeDis2 = W22 * U_flex2';
    NodeVel2 = W22 * dU_flex2';
    Fvec1 = zeros(size(U_flex1,1),size(W21,1));
    Fvec2 = zeros(size(U_flex2,1),size(W22,1));
%     ConnectDis = zeros(ConnectNum,3);
%     ConnectVel = zeros(ConnectNum,3);
    for NS=1:length(Nloc1)
        ConnectDis1{1,NS}=(NodeDis1((Nloc1(NS)-1)*3+(1:3),:))';
        ConnectVel1{1,NS}=(NodeVel1((Nloc1(NS)-1)*3+(1:3),:))';
    end
    for NS=1:length(Nloc2)
        ConnectDis2{1,NS}=(NodeDis2((Nloc2(NS)-1)*3+(1:3),:))';
        ConnectVel2{1,NS}=(NodeVel2((Nloc2(NS)-1)*3+(1:3),:))';
    end    
    for i=1:3
        for j=1:3
            if j==1
                ConnectDis_reshape1{1,i}=ConnectDis1{1,j}(:,i);
                ConnectVel_reshape1{1,i}=ConnectVel1{1,j}(:,i);
            else
                ConnectDis_reshape1{1,i}=[ConnectDis_reshape1{1,i} ConnectDis1{1,j}(:,i)];
                ConnectVel_reshape1{1,i}=[ConnectVel_reshape1{1,i} ConnectVel1{1,j}(:,i)];
            end
        end
        for j=4:6
            if j==4
                ConnectDis_reshape2{1,i}=ConnectDis1{1,j}(:,i);
                ConnectVel_reshape2{1,i}=ConnectVel1{1,j}(:,i);
            else
                ConnectDis_reshape2{1,i}=[ConnectDis_reshape2{1,i} ConnectDis1{1,j}(:,i)];
                ConnectVel_reshape2{1,i}=[ConnectVel_reshape2{1,i} ConnectVel1{1,j}(:,i)];
            end
        end
        for j=1:ConnectNum2
            if j==1
                ConnectDis_reshape3{1,i}=ConnectDis2{1,j}(:,i);
                ConnectVel_reshape3{1,i}=ConnectVel2{1,j}(:,i);
            else
                ConnectDis_reshape3{1,i}=[ConnectDis_reshape3{1,i} ConnectDis2{1,j}(:,i)];
                ConnectVel_reshape3{1,i}=[ConnectVel_reshape3{1,i} ConnectVel2{1,j}(:,i)];
            end
        end
    end
    ConnectDis1Double=cell2mat(ConnectDis_reshape1);
    ConnectVel1Double=cell2mat(ConnectVel_reshape1);
    ConnectDis2Double=cell2mat(ConnectDis_reshape2);
    ConnectVel2Double=cell2mat(ConnectVel_reshape2);
    ConnectDis3Double=cell2mat(ConnectDis_reshape3);
    ConnectVel3Double=cell2mat(ConnectVel_reshape3);
    % 刚体部分ODE损失
    Du_rigid1 = M1*ddU_rigid1+K1*(U_rigid1-ConnectDis1Double)+C1*(dU_rigid1-ConnectVel1Double);
    Du_rigid2 = M2*ddU_rigid2+K2*(U_rigid2-ConnectDis2Double+U_rigid2-ConnectDis3Double)+C2*(dU_rigid2-ConnectVel2Double+dU_rigid2-ConnectVel3Double);
    % 柔性体部分ODE损失
    ForceElementOutput1A = K1*(U_rigid1-ConnectDis1Double)+C1*(dU_rigid1-ConnectVel1Double);
    ForceElementOutput1B = K2*(U_rigid2-ConnectDis2Double)+C2*(dU_rigid2-ConnectVel2Double);
    Fvec1(:,TrackDOFs1(:,1)) = [ForceElementOutput1A(:,1:3) ForceElementOutput1B(:,1:3)];
    Fvec1(:,TrackDOFs1(:,2)) = [ForceElementOutput1A(:,4:6) ForceElementOutput1B(:,4:6)];
    Fvec1(:,TrackDOFs1(:,3)) = [ForceElementOutput1A(:,7:9) ForceElementOutput1B(:,7:9)];
    Gvec1 = Fvec1 * W21;
    Du_flex1 = ddU_flex1-Gvec1+kron(ones(dimension,1),Eigens21').*U_flex1+(kron(ones(dimension,1),Eigens21').*alpha_k+alpha_c).*dU_flex1;
    ForceElementOutput2 = K2*(U_rigid2-ConnectDis3Double)+C2*(dU_rigid2-ConnectVel3Double);
    Fvec2(:,TrackDOFs2(:,1)) = [ForceElementOutput2(:,1:3)];
    Fvec2(:,TrackDOFs2(:,2)) = [ForceElementOutput2(:,4:6)];
    Fvec2(:,TrackDOFs2(:,3)) = [ForceElementOutput2(:,7:9)];
    Gvec2 = Fvec2 * W22;
    Du_flex2 = ddU_flex2-Gvec2+kron(ones(dimension,1),Eigens22').*U_flex2+(kron(ones(dimension,1),Eigens22').*alpha_k+alpha_c).*dU_flex2;
    % 激励部分ODE项
    ExcitationForce=K_Contact.*(E-U_rigid1(:,7:9));
    Du_rigid1(:,7:9)=Du_rigid1(:,7:9)-ExcitationForce;
    Du=[Du_flex1 Du_flex2 Du_rigid1 Du_rigid2];
    for i=1:size(Du,2)
        Weights_Tem(i,1)=max(Du(5:end,i));
    end
    Weights(DataIndex,:)=Weights_Tem';
    A=1;
end
SaveFileName=['Weights_Medium_',num2str(n),'_V',num2str(version),'.mat'];SaveFileName=[Path,'\',SaveFileName];
save(SaveFileName,'Weights');