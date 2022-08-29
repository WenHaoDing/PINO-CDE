function [TrackDOFs,Nloc] =Subroutine_InteractionSearch(epis,Nodes)
%% /耦合结点搜索/-整个面域上的结点
% xrange=[0.0-epis 0.0+epis];
% yrange=[0.01-epis 0.01+epis];
% zrange=[-0.02-epis -0.02+epis];
xrange=[-500 500];
yrange=[-500 500];
zrange=[1.56-epis 1.56+epis];
%% 搜索耦合结点
NnodeLoc = [xrange(1) yrange(1) zrange(1);...
            xrange(2) yrange(2) zrange(2)];
ConNode_num = get_nodes_by_selection(Nodes,NnodeLoc);
Nloc = ConNode_num(:);

TrackDOFs=zeros(numel(Nloc),3);
for itr=1:numel(Nloc)
    TrackDOFs(itr,1) = (Nloc(itr)-1)*3+1;                                  %   1-耦合点数目*2个耦合点 横桥向 x方向自由度 对应W2中的行数
    TrackDOFs(itr,2) = (Nloc(itr)-1)*3+2;                                  %   1-耦合点数目*2个耦合点 横桥向 x方向自由度 对应W2中的行数
    TrackDOFs(itr,3) = (Nloc(itr)-1)*3+3;                                  %   1-耦合点数目*2个耦合点 垂直 竖直向上 y方向自由度   对应W2中的行数
end
end

