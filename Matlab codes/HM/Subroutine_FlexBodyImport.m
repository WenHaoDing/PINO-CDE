function [K1,M1,Mappings1,MMat1,MatA1,Eigens1,outside_faces1,Nodes1,eps,ModeNumMax1,W21,Eigens21,u_bridge1,...
          K2,M2,Mappings2,MMat2,MatA2,Eigens2,outside_faces2,Nodes2,ModeNumMax2,W22,Eigens22,u_bridge2]=Subroutine_FlexBodyImport(SectionNumber)
%% /子程序/-柔性体结构文件导入
%% 本程序用于综合ANSYS有限元模型导入Matlab生成的所有文件
%% PART0.文件内容定义
load('IS.mat');
%% /可视化模块/-选择柔性体模型的可视化
Switch_Plot='Off';
if strcmp(Switch_Plot,'On')==1
plot_modal(MatA1,outside_faces1,Nodes1,15);                                    %选择-可视化模型模态（第一阶）
end
N_DoF=3;                                                                   %每一个单元节点的自由度数量
if size(Nodes1,2) == 4
    Nodes1(:,1) = [];                 % [节点数，3]  1-3列为x,y,z坐标
    Nodes2(:,1) = [];
end
eps=5e-4;                                                                 %节点搜索容差
%% PART1.桥梁结构模态信息定义
ModeNumMax1 = 15;                                                          
ModeNumMax2 =10;
W21 = MMat1(:,1:ModeNumMax1);                                                
W22 = MMat2(:,1:ModeNumMax2);
Eigens21 = Eigens1(1:ModeNumMax1);
Eigens22 = Eigens2(1:ModeNumMax2);
u_bridge1 = zeros(2*ModeNumMax1,SectionNumber);          
u_bridge2 = zeros(2*ModeNumMax2,SectionNumber);    
end

