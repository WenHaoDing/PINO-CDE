function [Input,Output,Flag]=DataGeneratorV2(outside_faces,Nodes,ModeNumMax,W2,Eigens2,TrackDOFs,Nloc,index)
%% 柔性体例2-多模态叠加求解地震荷载激励下建筑物振动
%% 强制采用位移荷载控制
%% Part1 几何构建柔性信息导入
SectionNumber=1;
% [K,M,Mappings,MMat,MatA,Eigens,outside_faces,Nodes,eps,...
%  ModeNumMax,W2,Eigens2,u_bridge]=Subroutine_FlexBodyImport(SectionNumber);
%% Part2 寻找柔性构建上的受力结点信息
% epis=1e-3;
% [TrackDOFs,Nloc]=Subroutine_InteractionSearch(epis,Nodes);
% [TrackDOFS_Spy,Nloc_Spy]=Subroutine_SpyInteractionSearch(epis,Nodes);
% ForceNumber=size(TrackDOFs,1);

% /临时/-暂时生成一个随机信号用于激励
% for i=1:2
%     [g] =Generator_Irre();
%     Signal(:,i)=g;
% end
% 
% Signal=1e-3.*Signal;
sample=3;
% Signal=Simulation_SeismicWave(sample);
Signal=Simulation_SeismicWave_Uniform(sample,index);
Signal(:,1)=0.85.*Signal(:,1);
Signal(:,2)=1.*Signal(:,2);
Signal(:,3)=0.65.*Signal(:,3);
Trans=15000;TransMap=[linspace(0,1,Trans).^1.1]';%每一个线型需要单独调整此处
Signal(1:Trans,:)=kron(ones(1,size(Signal,2)),TransMap).*Signal(1:Trans,:);

Tstep=1e-4;
T=5+2*Tstep;
Nt=round(T/Tstep);
local_count = 0;

%% 激励插值
Time=[Tstep:Tstep:T]';
TimeSignal=[1e-4:1e-4:1e-4*size(Signal,1)]';
if Tstep~=1e-4
    for i=1:size(Signal,2)
        Signal_Spline(:,i)=spline(TimeSignal,Signal(:,i),Time);
    end
    Signal=Signal_Spline;
else
    Signal=Signal(1:length(Time),:);
end
for i=1:size(Signal,2)
    [disint,~] = IntFcn(Signal(:,i)', Time', Tstep, 2);
    Signal_Dis(:,i)=disint'-disint(1);
%     Signal_Vel(:,i)=velint;
end
Signal_Vel=(Signal_Dis(3:end,:)-Signal_Dis(1:end-2,:))./(2*Tstep);
Signal_Acl=(Signal_Dis(3:end,:)-2.*Signal_Dis(2:end-1,:)+Signal_Dis(1:end-2,:))./(Tstep^2);
Signal_Dis=Signal_Dis(2:end-1,:);
Time=Time(2:end-1);
T=T-2*Tstep;
DPI=2;
Tstep=Tstep*DPI;
Nt=round(T/Tstep);
Signal_Dis=Signal_Dis(1:DPI:end,:);
Signal_Vel=Signal_Vel(1:DPI:end,:);
Signal_Acl=Signal_Acl(1:DPI:end,:);
Time=Time(1:DPI:end,:);
% clear Signal_Spline
K=1e4;
C=1e2;
M_Vector=ones(1,9);
K_Contact=1e4;
Vc=250/3.6;
X0=10;
Step0=round(X0/Vc/Tstep);
Step=round(0.5/Vc/Tstep);
alpha=0.125;
beta=0.25;
ModeX=zeros(Nt,ModeNumMax);
ModeV=zeros(Nt,ModeNumMax);
ModeA=zeros(Nt,ModeNumMax);
DisRecord=zeros(length(Nloc),3);
VelRecord=zeros(length(Nloc),3);
AclRecord=zeros(length(Nloc),3);
SpyDisRecord_Interaction=zeros(Nt,length(Nloc),3);
SpyVelRecord_Interaction=zeros(Nt,length(Nloc),3);
SpyAclRecord_Interaction=zeros(Nt,length(Nloc),3);
Component=pinv(W2);
Flag=0;
for ii=1:Nt
% tempo=strcat(num2str(ii/Nt*100),'%');
% disp(tempo);
%% 不收敛预警系统
if ii==100
    if max(ModeA)>1e10
        Flag=1;
    end
end
%% Part3 系统Zhai积分
if ii==1
elseif ii==2        
        ModeX(ii,:)=ModeX(ii-1,:)+ModeV(ii-1,:)*Tstep+((1/2)+alpha)*ModeA(ii-1,:)*Tstep^2;
        ModeV(ii,:)=ModeV(ii-1,:)+(1+beta)*ModeA(ii-1,:)*Tstep;
else
        ModeX(ii,:)=ModeX(ii-1,:)+ModeV(ii-1,:)*Tstep+((1/2)+alpha)*ModeA(ii-1,:)*Tstep^2-alpha*ModeA(ii-2,:)*Tstep^2;
        ModeV(ii,:)=ModeV(ii-1,:)+(1+beta)*ModeA(ii-1,:)*Tstep-beta*ModeA(ii-2,:)*Tstep;
end
%% Part4 从桥梁模态信息中提取桥梁结构的振动响应
InteractionDis=W2*ModeX(ii,:)';                                
InteractionVel=W2*ModeV(ii,:)';
if ii>=2
    InteractionAcl=W2*ModeA(ii-1,:)';
else
    InteractionAcl=W2*ModeA(ii,:)';
end
for BS=1:SectionNumber
    for NS=1:length(Nloc)
        DisRecord(NS,:)=(InteractionDis((Nloc(NS)-1)*3+(1:3),BS))';
        VelRecord(NS,:)=(InteractionVel((Nloc(NS)-1)*3+(1:3),BS))';
        AclRecord(NS,:)=(InteractionAcl((Nloc(NS)-1)*3+(1:3),BS))';
        SpyDisRecord_Interaction(ii,NS,:)=DisRecord(NS,:);
        SpyVelRecord_Interaction(ii,NS,:)=VelRecord(NS,:);
        SpyAclRecord_Interaction(ii,NS,:)=AclRecord(NS,:);
    end
%     for NS=1:length(Nloc_Spy)
%         SpyDisRecord(ii,NS,:)=(InteractionDis((Nloc_Spy(NS)-1)*3+(1:3),BS))';
%         SpyVelRecord(ii,NS,:)=(InteractionVel((Nloc_Spy(NS)-1)*3+(1:3),BS))';
%         SpyAclRecord(ii,NS,:)=(InteractionAcl((Nloc_Spy(NS)-1)*3+(1:3),BS))';
%     end
    DisRecord_Store=DisRecord;
    VelRecord_Store=VelRecord;
    AclRecord_Store=AclRecord;
end                                       
%% Part5 力元出力计算
SignalPack=[Signal_Dis(ii,:) Signal_Vel(ii,:) Signal_Acl(ii,:)];
if ii>=500
    Flag=1;
else
    Flag=0;
end
[Force_X,Force_Y,Force_Z]=Force_KV_ForceElementV2(SignalPack,DisRecord_Store,VelRecord_Store,AclRecord_Store,Flag);
%% STEP7 柔性体响应计算
%% SEC1.Computation response for Flexible Body
if ii==1
    Fvec=zeros(length(W2),1);
end
for BI=1:SectionNumber
    Fvec(TrackDOFs(:,1),1)=-Force_X;
    Fvec(TrackDOFs(:,2),1)=-Force_Y;
    Fvec(TrackDOFs(:,3),1)=-Force_Z;
    alpha_k=0.01;
    alpha_c=0.01;
    Gvec=W2'*Fvec;
    ModeA(ii,:)=Gvec'-Eigens2'.*ModeX(ii,:)-(Eigens2'.*alpha_k+alpha_c).*ModeV(ii,:); 
end
%% SEC2.Visualization for Flexible Body
Switch_Visualization='Off';
if strcmp(Switch_Visualization,'On')==1
u_bridge=[ModeX(ii,:)';ModeV(ii,:)'];
Dfai_c = W2 * u_bridge(1:ModeNumMax); 
% Dfai_c = W2 * ModeA(ii,:)';
    scale_fcator = 1000;       
    if ~rem(ii*Tstep,0.05)
        vi = reshape(Dfai_c,3,[])';
        local_count = local_count+1;
        if local_count == 1
%             scale_fcator = 1e-2/max(Dfai_c(:))*1e-1;
%             scale_fcator = 2e5;
            [hfig,p] = plot3_DisplaceModel2(vi,outside_faces,Nodes,scale_fcator,0,0);
        elseif local_count < 10
%             scale_fcator = 1e-1/max(Dfai_c(:))*1e-1;
%             scale_fcator = 2e5;
            plot3_DisplaceModel_Version2(hfig,p,vi,Nodes,scale_fcator,0)
        else
%             scale_fcator = Dfai_c*1e3;
%             scale_fcator = 1e3/max(Dfai_c(:));
            plot3_DisplaceModel_Version2(hfig,p,vi,Nodes,scale_fcator,0)
        end
        pause(0.1)
    end
end
end
%% 测试位移荷载控制是否有效
% Target=SpyDisRecord_Interaction;
% figure(1)
% Target=SpyDisRecord;
% plot(squeeze(Target(:,:,3)));hold on;
% plot(Signal_Vel(1:Nt,3));hold off;
% figure(2)
% Target=SpyAclRecord_Interaction;
% plot(squeeze(Target(:,1:200,3)));hold on;
% plot(Signal_Acl(1:Nt,3));hold off;
%% 数据整合
DataPack=[Time(1:Nt,:) Signal_Dis(1:Nt,:) Signal_Vel(1:Nt,:) ModeX ModeV ModeA];
DPI=5;
DataPack=DataPack(1:DPI:size(DataPack,1),:);
Input=DataPack(:,1:7);
Output=DataPack(:,8:end);
end