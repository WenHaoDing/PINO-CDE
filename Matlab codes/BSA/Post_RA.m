clear;
clc;
%% Import maximum stress field (from python generation)
Path='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\Upload\post processing';
load([Path,'\MaxStress.mat']);
MaxStress=double(MaxStress);
MeanStress=mean(MaxStress);
%% Computing Structure Damage Probability
CaseNum=size(MaxStress,1);
Damage = MaxStress;
Threshold = 1.7;
Damage((MaxStress>=Threshold))=1;
Damage((MaxStress<Threshold))=0;
Inspection=sum(Damage);
DamagePro=Inspection./CaseNum;
plot(DamagePro);
%% Generating corresponding 4-D data
[K,M,Mappings,MMat,MatA,Eigens,outside_faces,Nodes,eps,...
 ModeNumMax,W2,Eigens2,u_bridge]=Subroutine_FlexBodyImport(1);
AllNodes=importdata('FigureNode.txt');                                     %
SPYE=load('Info_Element_Node.txt');                                        %默认选取了所有的单元
EList=SPYE(:,1);                                                           %所有要监视的单元编号
NodeList=SPYE(:,7:14);                                                     %所有单元对应的节点编号
StressLoc=zeros(length(EList),3);                                          %存储所有单元正中心的坐标
for i=1:length(EList)
    % 提取单元正中心的位置
    StressLoc(i,1)=mean(Nodes(NodeList(i,:),1));
    StressLoc(i,2)=mean(Nodes(NodeList(i,:),2));
    StressLoc(i,3)=mean(Nodes(NodeList(i,:),3));
end
X=StressLoc(:,1);
Y=StressLoc(:,2);
Z=StressLoc(:,3);
Xq=Nodes(1:end,1);
Yq=Nodes(1:end,2);
Zq=Nodes(1:end,3);
DamageField = griddata(X,Y,Z,DamagePro,Xq,Yq,Zq);
MeanMaxStressField = griddata(X,Y,Z,MeanStress,Xq,Yq,Zq);
DamageField(isnan(DamageField)) = 0;
MeanMaxStressField(isnan(DamageField)) = 0;
scale_fcator=1e-5;
[hfig,p] = plot3_DisplaceModel2_V3(DamageField,outside_faces,Nodes,scale_fcator,0,0,1);
[hfig,p] = plot3_DisplaceModel2_V3(MeanMaxStressField,outside_faces,Nodes,scale_fcator,0,0,2);
hold on;
x=[13.6 13.6 29.5 29.5 13.6];
y=[-9.5 -19 -19 -9.5 -9.5];
z=[1.00 1.00 1.00 1.00 1.00];
l1=plot3(x,y,z);hold on;
x=[13.6 13.6 29.5 29.5 13.6];
y=[-9.5 -19 -19 -9.5 -9.5];
z=[4 4 4 4 4];
l2=plot3(x,y,z);hold on;
x=[13.6 13.6];y=[-9.5 -9.5];z=[1.00 4];
l3=plot3(x,y,z);hold on;
x=[13.6 13.6];y=[-19 -19];z=[1.00 4];
l4=plot3(x,y,z);hold on;
x=[29.5 29.5];y=[-9.5 -9.5];z=[1.00 4];
l5=plot3(x,y,z);hold on;
x=[29.5 29.5];y=[-19 -19];z=[1.00 4];
l6=plot3(x,y,z);hold off;
for i=1:6
    eval(['l',num2str(i),'.LineWidth=0.75;']);
    eval(['l',num2str(i),'.LineStyle='':'';']);
    eval(['l',num2str(i),'.Color=[1 0.498 0 1];']);
end

%% Create Sliced Damage Probability Field
loc = 'F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\Upload\FEM files\BSA\';
model_file = [loc 'Building.cdb'];
[Nodes, Solid] = load_model_from_ANSYS(model_file);

Path='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\Upload\FEM files\BSA';
SliceE=importdata([Path,'\','SliceElements.txt']);
SliceE=SliceE(:,1);
Solid(:, 1:11) = [];    
Solid=Solid(SliceE,:);
Sliceoutside_faces = load_4mesh_from_ANSYS(Solid);
SliceNodes=reshape(Solid,size(Solid,1)*size(Solid,2),1);
SliceNodes=unique(SliceNodes);
SliceNodesCoord=[SliceNodes Nodes(SliceNodes,2:end)];
Dummy=Nodes(:,1);Dummy(SliceNodes)=[];
SliceDamageField=DamageField;SliceDamageField(Dummy,:)=0.*SliceDamageField(Dummy,:);
[hfig,p] = plot3_DisplaceModel2_V3(SliceDamageField,Sliceoutside_faces,Nodes,scale_fcator,0,0,3);
hold on;
x=[13.6 13.6 30.0 30.0 13.6];
y=[-8 -19.5 -19.5 -8 -8];
z=[1.00 1.00 1.00 1.00 1.00];
l1=plot3(x,y,z);hold on;
x=[13.6 13.6 30.0];
y=[-19.5 -8 -8];
z=[6 6 6];
l2=plot3(x,y,z);hold on;
x=[13.6 13.6];y=[-8 -8];z=[1.00 6];
l3=plot3(x,y,z);hold on;
x=[13.6 13.6];y=[-19.5 -19.5];z=[1.00 6];
l4=plot3(x,y,z);hold on;
x=[30.0 30.0];y=[-8 -8];z=[1.00 6];
l5=plot3(x,y,z);hold on;
% x=[30.0 30.0];y=[-19.5 -19.5];z=[1.00 6];
% l6=plot3(x,y,z);hold off;
for i=1:5
    eval(['l',num2str(i),'.LineWidth=1.5;']);
    eval(['l',num2str(i),'.LineStyle='':'';']);
    eval(['l',num2str(i),'.Color=[1 0.498 0 1];']);
end