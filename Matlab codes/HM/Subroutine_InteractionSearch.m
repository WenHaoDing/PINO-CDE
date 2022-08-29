function [TrackDOFs,Nloc] =Subroutine_InteractionSearch(epis,Nodes)
%% /耦合结点搜索/-整个面域上的结点
r=0.12;
angle = (pi/6) + [0 (2/3)*pi (4/3)*pi];
angle = [angle angle+(pi/3)];
for i=1:length(angle)
    if i<=3
        IL{i,1}=[r*cos(angle(i)) r*sin(angle(i)) 0.02];
    else
        IL{i,1}=[r*cos(angle(i)) r*sin(angle(i)) 0];
    end
end
IL=cell2mat(IL);

%% 搜索耦合结点
for itr = 1:size(IL,1)
    tnode = IL(itr,:);
    NnodeLoc = [tnode(1)-epis tnode(2)-epis (tnode(3))-epis; ...
        tnode(1)+epis tnode(2)+epis (tnode(3))+epis];
    ConNode_num = get_nodes_by_selection(Nodes,NnodeLoc);
    Nloc(itr,1) = ConNode_num(1);
end
TrackDOFs=zeros(numel(Nloc),3);
for itr=1:numel(Nloc)
    TrackDOFs(itr,1) = (Nloc(itr)-1)*3+1;                                  %   1-耦合点数目*2个耦合点 横桥向 x方向自由度 对应W2中的行数
    TrackDOFs(itr,2) = (Nloc(itr)-1)*3+2;                                  %   1-耦合点数目*2个耦合点 横桥向 x方向自由度 对应W2中的行数
    TrackDOFs(itr,3) = (Nloc(itr)-1)*3+3;                                  %   1-耦合点数目*2个耦合点 垂直 竖直向上 y方向自由度   对应W2中的行数
end
end

