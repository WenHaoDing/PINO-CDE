
function [K,M,Mappings,MMat,MatA,Eigens,outside_faces,Nodes] = load_ANSYS_model(loc,opt,name)

% model n.模型
% opt = 1 : 读取所有参数
% opt = 0 : 除了刚度矩阵与质量矩阵

% Mappings -------- 
% MMat ------------ 
% Eigens ---------- Eigens = (Freqs*2*pi).^2; 特征值
% outside_faces --- 
% Nodes ----------- 
  

if nargin == 1     % 函数输入参数数目
    opt = 0;
end

% loc = 'E:\ANSYS\Modal4\';
% loc = 'H:\Yuzhuangzhuang\Software Data\ANSYS19.0\N03Tower_Tongling\';
input_file1 = [loc 'STIFF_MATRIX.TXT'];                % 刚度矩阵
input_file2 = [loc 'MASS_MATRIX.TXT'];
model_file = [loc name];
mapping_file = [loc 'STIFF_MATRIX.mapping'];
modal_file = [loc 'modefile.txt'];

%% 读取结构刚度矩阵和质量矩阵（稀疏状态便于进行数据存储）
if opt
    K = HBMatrixRead(input_file1);                                         %通过功能函数读取结构的刚度矩阵
    M = HBMatrixRead(input_file2);                                         %通过功能函数读取结构的质量矩阵
else
    K = [];
    M = [];
end

%% 读取有限元的几何关系

[Nodes, Solid] = load_model_from_ANSYS(model_file);  % -------------------------->1  

Solid(:, 1:11) = [];    % 剩下的每一行的1-8列为定义这个solid45单元的8个节点编号

%% 读取有限元的几何关系

% fname = model_file;
% nodes_str = '%d %d %d %e %e %e';
% solid_str = '';
% for itr = 1:19
%     solid_str = [solid_str '%d ']; %#ok<AGROW>
% end
% 
% % Geo = importdata(fname);
% fid = fopen(fname,'r');
% 
% tline = fgetl(fid);
% tline2 = [];
% while ischar(tline)
%     if strcmp(tline,'(3i8,6e20.13)')
%         num = str2double(tline2(end-6:end));
%         Nodes = zeros(num,6);
%         tline = fgetl(fid);
%         for itr = 1:num
%             vec = sscanf(tline,nodes_str);
%             Nodes(itr,1:length(vec)) = vec;
%             tline = fgetl(fid);
%         end
%         Nodes(:,[2 3]) = [];
%     elseif strcmp(tline,'(19i8)')
%         num = str2double(tline2(end-6:end));
%         Solid = zeros(num,9)*NaN;
%         tline = fgetl(fid);
%         for itr = 1:num
%             vec = sscanf(tline,solid_str);
%             if length(vec) < 19
%                 Solid(itr,1:length(vec)-10) = vec(11:end);
%                 tline = fgetl(fid);
%             else
%                 Solid(itr,:) = vec(11:19);
%                 tline = fgetl(fid);
%             end 
%         end
%         break;
%     else
%         tline2 = tline;
%         tline = fgetl(fid);
%     end
% end
% fclose(fid);
% Solid(:,1) = [];
% 
% Solid(isnan(sum(Solid,2)),:) = [];

% pMMat = pinv(MMat);
% mesh(pMMat*MMat)
%% Mapping 寻找节点对应关系，刚度矩阵的行列与节点对应关系

% MappingCell = importdata('STIFF_MATRIX100.mapping');
MappingCell = importdata(mapping_file);
Mapping = zeros(length(MappingCell)-1,2);
for itr = 2:length(MappingCell)
    vecs = sscanf(MappingCell{itr},'%d %d');
    Mapping(itr-1,1:length(vecs)) = vecs;    % 读入STIFF_MATRIX.mapping数据 行数=矩阵维度，第二列为节点编号
end

degreeFredom_num = 3;
vec123 = [1;2;3]*ones(1,length(Mapping)/degreeFredom_num);
Mapping(:,2) = (Mapping(:,2)-1)*degreeFredom_num+vec123(:);
%此时Mapping矩阵的第二列存储的是每一个矩阵编号对应的实际节点自由度 需要注意的是mapping文件中不包含已经锁定自由度的约束点的自由度
%Nodes中和Mapping文件中都不会显示已经被约束的节点的自由度
invMapping = Mapping;
[~,b] = sort(Mapping(:,2));  % B = sort(A) 按升序对 A 的元素进行排序。     ？？？？？？？？？
invMapping(:,2) = b;                       
% plot(Mapping(:,1),Mapping(:,2),'o')
% Mapping(invMapping)
% b中存储的是按照系统自由度进行升序排列以后对应的结构刚度矩阵的位置序列

Mapping = Mapping(:,2);  %  （每个节点编号-1）*3+（1，2，3）
invMapping = invMapping(:,2);

Mappings = [Mapping invMapping];
% Mapping中存储的第一列数据为原始的mapping信息，即在进行升序排列之前的txt中的信息指代的自由度的排列方式
% Mapping中存储的第二列数据为排序之后的mapping信息，即在进行升序排列之后的txt中的信息，即为如果希望进行自由度的升序排列，排列的顺序将会为Mapping数组中的第二列数据
% spy(K)
% spy(K(invMapping,invMapping))


%% 读取ANSYS模态计算结果

[MMat,MatA,Eigens] = load_modal_from_ANSYS(modal_file);  % -------------------------->2  


% Matp = MMat(Mapping,:);
% 
% mesh(full(Matp'*K*Matp))
% 
% figure(12)
% hold on;
% plot(diag(Matp'*K*Matp));
% plot(Eigens,'r');
% 
% plot(diag(Matp'*K*Matp)-Eigens)


%% 使用刚度矩阵与质量矩阵计算模态数据

% Lm = chol(M,'lower');            % 11124x11124 sparse double
% % mesh(full(Lm*Lm'-M))
% % invLm = inv(Lm);
% % spy(invLm)
% 
% mMat = (Lm\K)/Lm';                         % 11124x11124 sparse double
% % [Vm,D] = eigs(mMat,length(mMat));
% [Vm,D] = eig(full(mMat));   % 将稀疏矩阵转换为满矩阵  [V,D] = eig(A)  D为特征值矩阵（对角阵）V为对应的特征向量矩阵
% W = Lm'\Vm;
% % W2 = Lm'\Vm;
% 
% diag_D = diag(D);  % x = diag(A) 返回 A 的主对角线元素的列向量   [11124,1]
% diag_D = abs(diag_D);   % 绝对值
% [sort_diag_D,ind] = sort(diag_D);   % 圆频率 特征值 omega^2    B = sort(A) 按升序对 A 的元素进行排序。
% sort_diag_D(sort_diag_D < 0) = 0;
% sort_diag_D2 = sqrt(sort_diag_D)/2/pi;  % 固有频率 f = omega/(2*pi)
% Wmat = W(:,ind);
% W2mat = W2(:,ind);
% Freqs = sort_diag_D2;   % 固有频率 f = omega/(2*pi)
% % VV = VV(:,ind);
% % figure(11)
% % hold on;
% % plot(sort_diag_D2);
% % plot(Freqs,'r');
% 
% % mesh(full(W'*K*W-D))
% % mesh(full(Vm'*D*Vm-mMat))
% % mesh(full(Vm*mMat*Vm'-D))
% % mesh(full(D))
% % mesh(full(W'*K*W))
% % mesh(full(W'*M*W))
% % 
% % mesh(full(Wmat'*K*Wmat))
% % 
% % plot(W(:,1))
% % 
% % plot(diag(D)-diag(W'*K*W))
% % plot(diag(D))
% % 
% % plot(sum(W.^2))
% % plot(sum(Vm.^2))


%% 根据mapping数据，将模态变量与刚度矩阵对应

% Modal_result = Wmat(invMapping,:);
% 
% W = MMat(:,:); % ----------- 与刚度矩阵对应起来的阵型向量组
% invW = pinv(W);

%% 整理模态矩阵

% ModalSum = 100;
% mode_num = 3;
% 
% Modes = 1:ModalSum;

% Modal_result = Wmat(invMapping,:);
% 
% vi = reshape(Modal_result(:,8),3,[])';
% % 
% ModalValueMat = zeros(size(vi,1),size(vi,2),length(Modes));
% for itr = 1:length(Modes)
%     vi = reshape(Modal_result(:,Modes(itr)),3,[])';
%     ModalValueMat(:,:,itr) = vi;
% end
% 
% plot(ModalValueMat(:,3,100))
% 
% plot(MatA(:,4,11))

% ModalValueMat = Modal_result(:,2:end,:);

%% 直接读取模态数据 ----------------

% ModalValueMat = MatA(:,2:end,:);   % [节点数，3，阶数]
% EigValues = Freqs;


%% 计算外边面片 -------------- 三角形网格

% outside_faces = load_3mesh_from_ANSYS(Solid);

%% 计算外边面片-------------- 四边形网格

outside_faces = load_4mesh_from_ANSYS(Solid);  % -------------------------->3


%% 

% 
% % sumFaces = sum(outside_faces,2);
% 
% % tempF = Faces(inds,:);
% 
% 
% % 确定内部点
% [a,b] = hist(Solid(:),1:max(Solid(:)));
% inside_nodes = b(a >= 8);
% 
% bools = zeros(size(Faces,1),1) == 1;
% for itr = 1:length(inside_nodes)
%     bools(Faces(:,1) == inside_nodes(itr)) = 1;
%     bools(Faces(:,2) == inside_nodes(itr)) = 1;
%     bools(Faces(:,3) == inside_nodes(itr)) = 1;
%     bools(Faces(:,4) == inside_nodes(itr)) = 1;
% end
% outside_faces1 = Faces(~bools,:);
% inFaces = Faces(bools,:);
% length(bools)-sum(bools)
%  

%% 赋值

% verts = Nodes(:,2:4);

%% 绘图
% plot3(Nodes(:,2),Nodes(:,3),Nodes(:,4),'.')      % nplot
% axis equal;
% axis([0 5 0 3 0 32]+[-2 2 -2 2 -2 2]*2);
% view(-35,53)

% Mode_num = 20;       % 准备绘制的模态阵型
% % plot3_DisplaceModel(ModalValueMat(:,:,Mode_num),outside_faces,Nodes,0)
% plot3_DisplaceModel(ModalValueMat(:,:,Mode_num),outside_faces,Nodes,100,1000,0)

% figure(10);clf
% hold on;
% col = 1;
% Mode_num = 7;
% plot(ModalValueMat(:,col,Mode_num))
% plot(-MatA(:,col+1,Mode_num),'r')

%% 可视化
% TPhi = ModalValueMat(:,:,1);
% faces = outside_faces;
% verts = Nodes;
% 
% if size(verts,2) == 4
%     verts(:,1) = [];
% end

% modeNums = ones(1,3)*20;
% LengthBeam = 32.5+0.2;
% 
% clear BeamObject
% for itr = 1:length(modeNums)
%     BeamObject(itr).TPhi = ModalValueMat(:,:,modeNums(itr)); 
%     BeamObject(itr).coeff_Plot = 0;
%     BeamObject(itr).faces = outside_faces;
%     BeamObject(itr).verts = Nodes(:,2:4);
%     BeamObject(itr).verts(:,3) = BeamObject(itr).verts(:,3)+LengthBeam*(itr);
% end
% 
%  
% [hfig, BeamObject] = get_muti_objects3D(BeamObject);
% 
% DefScale = 200;
% dt = 0.1;
% % coeff_Plot = 0;
% for itr = 1:1000
% %     disp(sin(ii)) 
% %     coeff_Plot = itr*dt;
%     for jtr = 1:length(BeamObject)
%         BeamObject(jtr).coeff_Plot = rem(itr*dt,2*pi);
%     end
%     update_figure3D(BeamObject,DefScale,0)
% end













