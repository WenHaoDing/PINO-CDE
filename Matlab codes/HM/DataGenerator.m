function [Input,Output]=DataGenerator()
%% 柔性体例1
%% Part1 几何构建柔性信息导入
SectionNumber=1;
[K1,M1,Mappings1,MMat1,MatA1,Eigens1,outside_faces1,Nodes1,eps,ModeNumMax1,W21,Eigens21,u_bridge1,...
 K2,M2,Mappings2,MMat2,MatA2,Eigens2,outside_faces2,Nodes2,ModeNumMax2,W22,Eigens22,u_bridge2]=Subroutine_FlexBodyImport(SectionNumber);
%% Part2 寻找柔性构建上的受力结点信息
epis=1e-3;
[TrackDOFs1,Nloc1]=Subroutine_InteractionSearch(epis,Nodes1);
[TrackDOFs2,Nloc2]=Subroutine_InteractionSearchV2(epis,Nodes2);
Nodes2(:,end)=Nodes2(:,end)-0.1;

for i=1:3
    [g] =Generator_Irre(250);
    Signal(:,i)=g;
end
Signal=1e-3.*Signal;
Trans=1000;TransMap=[linspace(0,1,Trans).^1.1]';%每一个线型需要单独调整此处
Signal(1:Trans,:)=kron(ones(1,3),TransMap).*Signal(1:Trans,:);

Tstep=1e-4;
T=1;
Nt=round(T/Tstep);
local_count = 0;
X=zeros(Nt,3*3*2);
V=zeros(Nt,3*3*2);
A=zeros(Nt,3*3*2);
% 随机修改1，刚度和阻尼以及小球的质量为变化参数
% K=1e4;
% C=1e2;
K1=0.5e4+(1.5e4)*rand(1);
C1=0.5e2+1.5e2*rand(1);
K2=0.5e4+(1.5e4)*rand(1);
C2=0.5e2+1.5e2*rand(1);
% K1=1e4+(0.5e4)*rand(1);
% C1=1e2+0.5e2*rand(1);
% K2=1e4+(0.5e4)*rand(1);
% C2=1e2+0.5e2*rand(1);
K=[K1 K2];
C=[C1 C2];
% MPack = 0.8+0.4*rand(1,2);
MPack = 0.4+1.6*rand(1,2);
M_Vector=[MPack(1).*ones(1,9) MPack(2).*ones(1,9)];
K_Contact=1e4;
Vc=250/3.6;
X0=10;
Step0=round(X0/Vc/Tstep);
Step=round(0.5/Vc/Tstep);
alpha=0.5;
beta=0.5;
ModeX1=zeros(Nt,ModeNumMax1);
ModeV1=zeros(Nt,ModeNumMax1);
ModeA1=zeros(Nt,ModeNumMax1);
ModeX2=zeros(Nt,ModeNumMax2);
ModeV2=zeros(Nt,ModeNumMax2);
ModeA2=zeros(Nt,ModeNumMax2);
StoreX=zeros(Nt,9,3);
StoreV=zeros(Nt,9,3);
for ii=1:Nt
tempo=strcat(num2str(ii/Nt*100),'%');
% disp(tempo);
%% Part3 刚体部分Zhai积分
if ii==1
%         X(ii,:)=zeros(1,Fum_Total);V(ii,:)=zeros(1,Fum_Total);A(ii,:)=zeros(1,Fum_Total);
elseif ii==2
        X(ii,:)=X(ii-1,:)+V(ii-1,:)*Tstep+((1/2)+alpha)*A(ii-1,:)*Tstep^2;
        V(ii,:)=V(ii-1,:)+(1+beta)*A(ii-1,:)*Tstep;
        
        ModeX1(ii,:)=ModeX1(ii-1,:)+ModeV1(ii-1,:)*Tstep+((1/2)+alpha)*ModeA1(ii-1,:)*Tstep^2;
        ModeV1(ii,:)=ModeV1(ii-1,:)+(1+beta)*ModeA1(ii-1,:)*Tstep;
        ModeX2(ii,:)=ModeX2(ii-1,:)+ModeV2(ii-1,:)*Tstep+((1/2)+alpha)*ModeA2(ii-1,:)*Tstep^2;
        ModeV2(ii,:)=ModeV2(ii-1,:)+(1+beta)*ModeA2(ii-1,:)*Tstep;
else
        X(ii,:)=X(ii-1,:)+V(ii-1,:)*Tstep+((1/2)+alpha)*A(ii-1,:)*Tstep^2-alpha*A(ii-2,:)*Tstep^2;
        V(ii,:)=V(ii-1,:)+(1+beta)*A(ii-1,:)*Tstep-beta*A(ii-2,:)*Tstep;
        
        ModeX1(ii,:)=ModeX1(ii-1,:)+ModeV1(ii-1,:)*Tstep+((1/2)+alpha)*ModeA1(ii-1,:)*Tstep^2-alpha*ModeA1(ii-2,:)*Tstep^2;
        ModeV1(ii,:)=ModeV1(ii-1,:)+(1+beta)*ModeA1(ii-1,:)*Tstep-beta*ModeA1(ii-2,:)*Tstep;
        ModeX2(ii,:)=ModeX2(ii-1,:)+ModeV2(ii-1,:)*Tstep+((1/2)+alpha)*ModeA2(ii-1,:)*Tstep^2-alpha*ModeA2(ii-2,:)*Tstep^2;
        ModeV2(ii,:)=ModeV2(ii-1,:)+(1+beta)*ModeA2(ii-1,:)*Tstep-beta*ModeA2(ii-2,:)*Tstep;
end
X_moment=X(ii,:);
V_moment=V(ii,:);
Force_Tem=0.5.*Signal(ii);
%% Part4 从两块钢片的模态信息中提取耦合节点信息（耦合位置是相同的）
InteractionDis1=W21*ModeX1(ii,:)';                                
InteractionVel1=W21*ModeV1(ii,:)';
InteractionDis2=W22*ModeX2(ii,:)';                                
InteractionVel2=W22*ModeV2(ii,:)';
for BS=1:SectionNumber
    for NS=1:length(Nloc1)
        DisRecord1{NS,1}=(InteractionDis1((Nloc1(NS)-1)*3+(1:3),BS))';
        VelRecord1{NS,1}=(InteractionVel1((Nloc1(NS)-1)*3+(1:3),BS))';
    end
    for NS=1:length(Nloc2)
        DisRecord2{NS,1}=(InteractionDis2((Nloc2(NS)-1)*3+(1:3),BS))';
        VelRecord2{NS,1}=(InteractionVel2((Nloc2(NS)-1)*3+(1:3),BS))';
    end
    % 第一块钢片
    DisRecord1=cell2mat(DisRecord1);
    DisRecord_Store1=DisRecord1;
    DisRecord1=cell(length(Nloc1),1);
    VelRecord1=cell2mat(VelRecord1);
    VelRecord_Store1=VelRecord1;
    VelRecord1=cell(length(Nloc1),1);
    % 第二块钢片
    DisRecord2=cell2mat(DisRecord2);
    DisRecord_Store2=DisRecord2;
    DisRecord2=cell(length(Nloc1),1);
    VelRecord2=cell2mat(VelRecord2);
    VelRecord_Store2=VelRecord2;
    VelRecord2=cell(length(Nloc1),1);
end
DisRecord_Store=[DisRecord_Store1;DisRecord_Store2];
VelRecord_Store=[VelRecord_Store1;VelRecord_Store2];
StoreX(ii,:,:)=DisRecord_Store;                                            %接触点振动位移
StoreV(ii,:,:)=VelRecord_Store;                                            %接触点振动速度                                           
%% Part5 力元出力计算
[Force_X1,Force_Y1,Force_Z1,...
 Force_X2,Force_Y2,Force_Z2,...
 Force_X3,Force_Y3,Force_Z3]=Force_KV_ForceElement(K,C,X_moment,V_moment,DisRecord_Store,VelRecord_Store);
%% STEP7 柔性体响应计算
%% SEC1.Computation response for Flexible Body
if ii==1
    Fvec1=zeros(length(W21),1);
    Fvec2=zeros(length(W22),1);
end
for BI=1:SectionNumber
    Fvec1(TrackDOFs1(1:6,1),1)=[-Force_X1;-Force_X2];
    Fvec1(TrackDOFs1(1:6,2),1)=[-Force_Y1;-Force_Y2];
    Fvec1(TrackDOFs1(1:6,3),1)=[-Force_Z1;-Force_Z2];
    alpha_k=0.001;
    alpha_c=0.001;
    Gvec1=W21'*Fvec1;
    ModeA1(ii,:)=Gvec1'-Eigens21'.*ModeX1(ii,:)-(Eigens21'.*alpha_k+alpha_c).*ModeV1(ii,:); 
    Fvec2(TrackDOFs2(1:3,1),1)=[-Force_X3];
    Fvec2(TrackDOFs2(1:3,2),1)=[-Force_Y3];
    Fvec2(TrackDOFs2(1:3,3),1)=[-Force_Z3];
    Gvec2=W22'*Fvec2;
    ModeA2(ii,:)=Gvec2'-Eigens22'.*ModeX2(ii,:)-(Eigens22'.*alpha_k+alpha_c).*ModeV2(ii,:); 
end
% GvecStore(ii,:)=Gvec';
%% SEC2.Visualization for Flexible Body
Switch_Visualization='Off';
if strcmp(Switch_Visualization,'On')==1
u_bridge1=[ModeX1(ii,:)';ModeV1(ii,:)'];
u_bridge2=[ModeX2(ii,:)';ModeV2(ii,:)'];
Dfai_c1 = W21 * u_bridge1(1:ModeNumMax1); 
Dfai_c2 = W22 * u_bridge2(1:ModeNumMax2); 
    scale_fcator = 50;       
    if ~rem(ii*Tstep,0.001)
        vi1 = reshape(Dfai_c1,3,[])';
        vi2 = reshape(Dfai_c2,3,[])';
        local_count = local_count+1;
        if local_count == 1
%             scale_fcator = 1e-2/max(Dfai_c(:))*1e-1;
%             scale_fcator = 2e5;
%             [hfig1,p1] = plot3_DisplaceModel2(vi1,outside_faces1,Nodes1,scale_fcator,0,0,1);
%             [hfig1,p1] = plot3_DisplaceModel2([vi1;vi2],[outside_faces1;outside_faces2],[Nodes1;Nodes2],scale_fcator,0,0,1);
            [p1,p2,hfig] = plot3_DisplaceModel2V2(vi1,outside_faces1,Nodes1,scale_fcator,0,0,1,vi2,outside_faces2,Nodes2);
        elseif local_count < 10
%             scale_fcator = 1e-1/max(Dfai_c(:))*1e-1;
%             scale_fcator = 2e5;
%             plot3_DisplaceModel_Version2(hfig1,p1,vi1,Nodes1,scale_fcator,0)
            plot3_DisplaceModel_Version2V2(hfig,p1,vi1,Nodes1,scale_fcator,0,p2,vi2,Nodes2)
%             plot3_DisplaceModel_Version2(hfig2,p2,vi2,Nodes2,scale_fcator,0)
        else
%             scale_fcator = Dfai_c*1e3;
%             scale_fcator = 1e3/max(Dfai_c(:));
%             plot3_DisplaceModel_Version2(hfig1,p1,vi1,Nodes1,scale_fcator,0)
            plot3_DisplaceModel_Version2V2(hfig,p1,vi1,Nodes1,scale_fcator,0,p2,vi2,Nodes2)
%             plot3_DisplaceModel_Version2(hfig2,p2,vi2,Nodes2,scale_fcator,0)
        end
        pause(0.0001)
    end
end
% figure(3); plot(squeeze(StoreX(:,1,3)));hold on;plot(squeeze(StoreX(:,7,3)));hold off;
%% Part8. Computation Response for Rigid Body
% Irre=[Signal(ii+Step0+2*Step,1);Signal(ii+Step0+Step,1);Signal(ii+Step0,1)];
Irre=[Signal(ii,1);Signal(ii,2);Signal(ii,3)];
Excitation=K_Contact.*(Irre-X_moment(7:9)');
GeneralForce1=[Force_X1' Force_Y1' (Force_Z1+Excitation)'];
A(ii,1:9)=GeneralForce1./M_Vector(1:9);
GeneralForce2=[Force_X2'+Force_X3' Force_Y2'+Force_Y3' Force_Z2'+Force_Z3'];
A(ii,10:end)=GeneralForce2./M_Vector(10:end);
end
% FlexDStore_X=squeeze(StoreX(:,:,1));
% FlexDStore_Y=squeeze(StoreX(:,:,2));
% FlexDStore_Z=squeeze(StoreX(:,:,3));
% FlexVStore_X=squeeze(StoreV(:,:,1));
% FlexVStore_Y=squeeze(StoreV(:,:,2));
% FlexVStore_Z=squeeze(StoreV(:,:,3));
% FlexDStore=[FlexDStore_X FlexDStore_Y FlexDStore_Z];
% FlexVStore=[FlexVStore_X FlexVStore_Y FlexVStore_Z];
Time=[1e-4:1e-4:Nt*1e-4]';
% figure(1);
% set(gcf,'unit','normalized','position',[0.1,0.1,0.6,0.6]);
% subplot(2,1,1)
% plot(Time,FlexDStore_Z(:,1),'b','LineWidth',1.2);hold on;
% plot(Time,FlexDStore_Z(:,2),'r','LineWidth',1.2,'LineStyle','--');hold on;
% plot(Time,FlexDStore_Z(:,3),'k','LineWidth',1.2);hold off;
% title('柔性体垂向振动变形');
% subplot(2,1,2)
% plot(Time,FlexVStore_Z(:,1),'b','LineWidth',1.2);hold on;
% plot(Time,FlexVStore_Z(:,2),'r','LineWidth',1.2,'LineStyle','--');hold on;
% plot(Time,FlexVStore_Z(:,3),'k','LineWidth',1.2);hold off;
% title('柔性体垂向振动速度');
% 
% figure(2);
% set(gcf,'unit','normalized','position',[0.1,0.1,0.6,0.6]);
% subplot(2,1,1)
% plot(Time,X(:,7),'b','LineWidth',1.2);hold on;
% plot(Time,X(:,8),'r','LineWidth',1.2,'LineStyle','--');hold on;
% plot(Time,X(:,9),'k','LineWidth',1.2);hold off;
% title('刚体垂向振动变形');
% subplot(2,1,2)
% plot(Time,V(:,7),'b','LineWidth',1.2);hold on;
% plot(Time,V(:,8),'r','LineWidth',1.2,'LineStyle','--');hold on;
% plot(Time,V(:,9),'k','LineWidth',1.2);hold off;
% title('刚体垂向振动速度');
%% 数据整合
DataPack=[Time Signal(1:Nt,:) kron(ones(Nt,1),K) kron(ones(Nt,1),C) kron(ones(Nt,1),MPack) ModeX1 ModeX2 ModeV1 ModeV2 ModeA1 ModeA2 X V A];
DataPack=DataPack(1:end-1,:);
DPI=10;
DataPack=DataPack(1:DPI:size(DataPack,1),:);
Input=DataPack(:,1:10);
Output=DataPack(:,11:end);
end

