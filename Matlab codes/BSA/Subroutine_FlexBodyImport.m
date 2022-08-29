function [K,M,Mappings,MMat,MatA,Eigens,outside_faces,Nodes,eps,...
          ModeNumMax,W2,Eigens2,u_bridge]=Subroutine_FlexBodyImport(SectionNumber)
%% /子程序/-柔性体结构文件导入
%% 本程序用于综合ANSYS有限元模型导入Matlab生成的所有文件
%% PART0.文件内容定义
% loc = 'G:\Project1\ANSYS_Building\';                     %定义ANSYS文件路径
% [K,M,Mappings,MMat,MatA,Eigens,outside_faces,Nodes]=load_ANSYS_model(loc,1);%读取ANSYS生成的所有结构参数
% save('Building_200.mat','K','M','Mappings','MMat','MatA','Eigens','outside_faces','Nodes');
load('Building_200.mat');

%% /可视化模块/-选择柔性体模型的可视化
Switch_Plot='Off';
if strcmp(Switch_Plot,'On')==1
plot_modal(MatA,outside_faces,Nodes,7);                                    %选择-可视化模型模态（第一阶）
end
N_DoF=3;                                                                   %每一个单元节点的自由度数量
if size(Nodes,2) == 4
    Nodes(:,1) = [];                                                       % [节点数，3]  1-3列为x,y,z坐标
end
eps=5e-4;                                                                 %节点搜索容差
%% PART1.桥梁结构模态信息定义
ModeNumMax =length(Eigens);                                                           %柔性体结构选取模态数量
W2 = MMat(:,1:ModeNumMax);                                                 %几何体节点的数量
Eigens2 = Eigens(1:ModeNumMax);                                            %特征向量对应的特征值平方
u_bridge = zeros(2*ModeNumMax,SectionNumber);                                          %桥梁结构模态空间坐标（初始预分配）
% u_bridge=reshape(u_bridge,2*ModeNumMax,[]);
% Cheat:The first 6 modes must be considered, otherwise the structure
% cannot be able to maintain correct dynamic respose because the reamining
% modes have difficulty (especially those with low frequencies)
% compensating the interaction displacement)
end

