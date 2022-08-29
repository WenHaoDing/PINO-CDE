clear;
clc;
%% 结构参数
E=3.55e10;
NU=0.2;
[K,M,Mappings,MMat,MatA,Eigens,outside_faces,Nodes,eps,...
 ModeNumMax,W2,Eigens2,u_bridge]=Subroutine_FlexBodyImport(1);
Elist=load('Elist.txt');                                                   %有限元模型的单元和结点对应信息文件
%% 导入位移场
index=473;                                                                 %选取的PINO-MBD训练步
Path='~';
load('~\150.mat','MeanV','StdV');
Data=importdata([Path,'\','Performance',num2str(index),'.txt']);
Dis_flex=Data(:,1:200);
GTDis_flex=Data(:,201:end);
for i=1:size(Dis_flex,2)
    Dis_flex(:,i)=Dis_flex(:,i)*StdV(7+i)+MeanV(7+i);
    GTDis_flex(:,i)=GTDis_flex(:,i)*StdV(7+i)+MeanV(7+i);
end
TPick=round(2.65/5*1000);                                                                 %动力问题选取的时刻
%% 真实位移场重构
Resource = W2 * Dis_flex(TPick,:)';                                        
GTResource = W2 * GTDis_flex(TPick,:)';
StressStore=zeros(size(Nodes,1),6);                                        %应力信息预分配
% h=waitbar(0,'please wait');
%% 提取需要监视的单元
SPYE=load('Info_Element_Node.txt');                                        %默认选取了所有的单元
EList=SPYE(:,1);                                                           %所有要监视的单元编号
NodeList=SPYE(:,7:14);                                                     %所有单元对应的节点编号
u1=zeros(24,1);
u2=u1;
StressStore=zeros(length(EList),12);                                       %用于存储所有要监视应力的单元
StressLoc=zeros(length(EList),3);                                          %存储所有单元正中心的坐标
%% 导入所有单元的几何变形矩阵
load('EBStoreV2.mat'); 
%EXStore存储了所有不同形状单元的几何矩阵
%% 默认监视单元中心位置的应力
for i=1:length(EBStoreV2)
    tempo=strcat(num2str(i/length(EBStoreV2)*100),'%');
    disp(tempo);
    for j=1:8
        NodeNum=NodeList(i,j);
        u1(3*(j-1)+1:3*j,1)=Resource((NodeNum-1)*3+1:NodeNum*3);
        u2(3*(j-1)+1:3*j,1)=GTResource((NodeNum-1)*3+1:NodeNum*3);
    end
    w1=squeeze(EBStoreV2(EList(i),:,:))*u1;
    w2=squeeze(EBStoreV2(EList(i),:,:))*u2;
    StressStore(i,:)=[(w1)' (w2)'];
end
StressStore=1E-6.*StressStore;
%% 提取所有节点的坐标
AllNodes=importdata('FigureNode.txt');
AllNodes=reshape(AllNodes,size(AllNodes,1)*size(AllNodes,2),1);
FNodeCoord=Nodes(AllNodes,:);
for i=1:length(EList)
    % 提取单元正中心的位置
    StressLoc(i,1)=mean(Nodes(NodeList(i,:),1));
    StressLoc(i,2)=mean(Nodes(NodeList(i,:),2));
    StressLoc(i,3)=mean(Nodes(NodeList(i,:),3));
end
%% 施工-尝试进行三维插值
X=StressLoc(:,1);
Y=StressLoc(:,2);
Z=StressLoc(:,3);
V1=StressStore(:,1);
V2=StressStore(:,2);
V3=StressStore(:,3);
Xq=Nodes(1:end,1);
Yq=Nodes(1:end,2);
Zq=Nodes(1:end,3);
Vq1 = griddata(X,Y,Z,V1,Xq,Yq,Zq);
Vq2 = griddata(X,Y,Z,V2,Xq,Yq,Zq);
Vq3 = griddata(X,Y,Z,V3,Xq,Yq,Zq);
Vq=[Vq1 Vq2 Vq3];
Vq(isnan(Vq)) = 0;
%% 施工-尝试进行三维热图绘制
scale_fcator=1e-3;
[hfig,p] = plot3_DisplaceModel2_V3(Vq(:,1),outside_faces,Nodes,scale_fcator,0,0,1);
% [hfig2,p2] = plot3_DisplaceModel2_V2(vi_GT,outside_faces,Nodes,scale_fcator,0,0,2);

