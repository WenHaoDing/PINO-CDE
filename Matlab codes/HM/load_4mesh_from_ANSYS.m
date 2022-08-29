%% 计算外边面片-------------- 四边形网格 
 
function [outside_faces] = load_4mesh_from_ANSYS(Solid)

node_order = [...
    1 2 3 4;             % 底面
    1 2 6 5;             % 正面
    2 3 7 6;             % 右侧
    1 4 8 5;             % 左侧
    3 4 8 7;             % 背面
    5 6 7 8];            % 顶面
node_order = node_order(:)';

% 收集所有的面
Faces = Solid(:,node_order);
% face1 = Faces(:,1:6:24);
Faces = [Faces(:,1:6:24);Faces(:,2:6:24);Faces(:,3:6:24);...    % 仅仅Solid45单元是这样
    Faces(:,4:6:24);Faces(:,5:6:24);Faces(:,6:6:24)];  % 所有单元的底面，正面，右侧，左侧，背面，顶面 [单元数*6，4]

tempFaces = Faces;
for jtr = 1:size(tempFaces,2)-1                  % 1-3循环 把每个面的四个节点编号从小到大排序
    Bo = tempFaces(:,2:end) < tempFaces(:,1:end-1);
    for itr = 1:size(tempFaces,2)-1    % 1-3循环
        tempFaces(Bo(:,itr),itr:itr+1) = tempFaces(Bo(:,itr),[itr+1 itr]); 
    end
end
% tempFaces 1-4列根据节点编号大小升序排列
IndexNumber = 0;
cols = size(tempFaces,2);  % 4列
for itr = 1:cols
    IndexNumber = IndexNumber+tempFaces(:,itr)*(1e5)^(cols-itr);   % ？？？？？？？？
end

[v,inds] = sort(IndexNumber);   % B = sort(A) 按升序对 A 的元素进行排序。
Faces = Faces(inds,:);
% tempFaces = tempFaces(inds,:);

% sumFaces = sum(Faces,2);

bo = diff([0;v]) == 0;
bo([bo(2:end); 0] == 1) = 1;
 
outside_faces = Faces(~bo,:);
sum(~bo)  % 数组元素总和

