
function [Nodes, Solids] = load_model_from_ANSYS(file_name)
%% 

% file_name = 'D:\ANSYS\五峰山\file2.cdb';  
% [Nodes, Solids] = load_model_from_ANSYS(file_name);
% Nodes ------ 1-4列：节点编号，xyz坐标

%% 读取有限元的几何关系

fname = file_name;                        % ENGINEERINGCALCULATION.cdb
NBLOCK_str = 'NBLOCK,%d,SOLID,%d,%d';     % ANSYS中NBLOCK语句的格式      分别为字段数6，定义的最大节点数，写入的节点数
EBLOCK_str = 'EBLOCK,%d,SOLID,%d,%d';     % ？？？
solid_str = get_str_format(19);           % format--格式   为什么是19个'%d'

% Geo = importdata(fname);
fid = fopen(fname,'r');             % 打开文件 filename 以便以二进制读取形式进行访问，并返回等于或大于 3 的整数文件标识符。 如果 fopen 无法打开文件，则 fileID 为 -1。

tline = fgetl(fid);                 % 将 fid 传递给 fgetl 函数以从文件读取一行。，并删除换行符。
while ischar(tline)          % 确定输入是否为字符数组,是，返回逻辑值1
    if strcmp(tline(1:min([length(tline), 6])),'NBLOCK')   % 节点数据块儿  字段数是6  tf = strcmp(s1,s2) 比较 s1 和 s2，如果二者相同，则返回 1 (true)
        sizeNodes = sscanf(tline, NBLOCK_str); % 获得'NBLOCK,%d,SOLID,%d,%d'分别为字段数6，定义的最大节点数，写入的节点数
        num = sizeNodes(2);                    % 定义的最大节点数
        Nodes = zeros(num,sizeNodes(1));
        tline = fgetl(fid); %#ok<NASGU>  % 跳跃一行
        tline = fgetl(fid);
        for itr = 1:num
            vec = textscan(tline,'%f');   % 从文本文件或字符串读取格式化数据 读cdb文件中的数据
            Nodes(itr,1:length(vec{1})) = vec{1};
            tline = fgetl(fid);
        end
        Nodes(:,[2 3]) = [];     % cdb中第2-3列数据为0   原来的4-6列为节点三维坐标
        
    elseif strcmp(tline(1:min([length(tline), 6])),'EBLOCK')
        sizeNodes = sscanf(tline, EBLOCK_str);
        num = sizeNodes(2);                   % 定义的最大单元数
        Solids = zeros(num,19)*NaN;
        tline = fgetl(fid); %#ok<NASGU>  % 跳跃一行
        tline = fgetl(fid);
        for itr = 1:num
            vec = sscanf(tline,solid_str);
            Solids(itr,1:length(vec)) = vec;
            tline = fgetl(fid);
        end
        break;
    else
        tline = fgetl(fid);
    end
end
fclose(fid);                    % 关闭文件

% Solid(:,1) = [];

% Solid(isnan(sum(Solid,2)),:) = [];
AAA=1;

%% 
function format_str = get_str_format(n)      % n = 19
format_str = '';
for itr = 1:n
    format_str = [format_str '%d ']; %#ok<AGROW>
end










