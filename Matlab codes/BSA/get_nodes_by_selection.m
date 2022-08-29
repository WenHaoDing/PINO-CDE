function OutNode_num = get_nodes_by_selection(Nodes,locs)
% locs 坐标向量，每一行为一个三元素坐标  预留耦合点坐标加减误差限
% Nodes 所有节点三维坐标


if size(Nodes,2) == 4
    Nodes(:,1) = [];   % 1-3列为x,y,z坐标
end

%% 本程序用于搜索节点，节点的空间坐标位置必须位于搜索节点的空间位置之中（并非按照最短距离进行搜索）

bools = zeros(size(Nodes,1),1) == 0;   % [耦合点个数，1]
for col = 1:3
    bools = bools & (Nodes(:,col) >= locs(1,col)) & (Nodes(:,col) <= locs(2,col));
end

OutNode_num = find(bools); % 寻找非零

% N = size(Nodes,1);
% OutNode_num = zeros(size(locs,1),1);
% for itr = 1:size(locs,1)
%     dist = sum((Nodes-ones(N,1)*locs(itr,1:3)).^2,2);
%     [~,inds] = min(dist);
%     OutNode_num(itr) = inds;
% end

% function cal_dist(Vec)
% val = Vec(:,1).^2

%% 
% locs = [0 0 0; 1 1 1];
% get_node_by_loc(Nodes,locs)


% Nodes(433,:)
