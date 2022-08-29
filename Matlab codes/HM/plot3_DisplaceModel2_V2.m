function [hfig,p] = plot3_DisplaceModel2_V2(TPhi,faces,verts,DefScale,isATime,ifColorBar,FigNum)

if nargin < 5
    ifColorBar = false;
    if nargin < 4
        isATime = false;
    end
end
if size(verts,2) == 4
    verts(:,1) = [];
end

%% 
% 动画显示
hfig = figure(FigNum);
clf
dt = 0.1;
edges = max(verts(:))*5;
% 使用的patch命令
p = patch( 'Faces', faces, 'Vertices', verts );
% axsz = 0.3 ;
set( gca, 'XLim', [min(verts(:,1)) max(verts(:,1))]+[-1 1].*edges, ...
    'YLim', [min(verts(:,2)) max(verts(:,2))]+[-1 1].*edges, ...
    'ZLim', [min(verts(:,3)) max(verts(:,3))]+[-1 1].*edges );
camlookat( p );
alpha( 0.6 );
axis square equal auto off;

% view( 67,-56 )
view( 156,22 )              % 调整视角

% Init userdata，将需要用的数据和控件句柄保存下来

%% 
if nargout == 0    
end

cameratoolbar('Show') 

data.FaceVertexItem = 1;
data.DefScale = DefScale;
data.dt = dt;
data.pause = false;
data.ModeChanged = false;

set( hfig, 'userdata', data );  % init displayed mode

if ~isATime
    dat = get( hfig, 'userdata' );  % 获取数据
    u = TPhi.* ( dat.DefScale * 1 ); % 
    set( p, 'vertices', verts + u ); % 改变节点坐标
    dispVector = dat.DefScale * sqrt( TPhi( :, 1 ) .^ 2 + TPhi( :, 2 ) .^ 2 + TPhi( :, 3 ) .^ 2 ); % 计算变形矢量
    faceVertexCData = GetVertCData2( dat.FaceVertexItem, dat.DefScale * TPhi, u, dispVector ) ; % 计算当前变形量对应的颜色值与最大变形量
    set( p, 'FaceVertexCData', faceVertexCData , 'EdgeColor', 'interp' ); % 更新颜色显示
    if ifColorBar
        maxDef = max(u(:));
        maxDef = maxDef / dat.DefScale;
        tickLabel = 0 : maxDef / 10 : maxDef; % 计算colorbar刻度并显示
        colorbar( 'YTickLabel', { num2str( tickLabel( 1 ) ), num2str( tickLabel( 2 ) ), num2str( tickLabel( 3 ) ), ...
            num2str( tickLabel( 4 ) ), num2str( tickLabel( 5 ) ), num2str( tickLabel( 6 ) ), num2str( tickLabel( 7 ) ), ...
            num2str( tickLabel( 8 ) ), num2str( tickLabel( 9 ) ), num2str( tickLabel( 10 ) ), num2str( tickLabel( 11 ) ) } );
    end
%     axis square equal auto off;
    return;
end

% main loop
ii = 0;
count = 0; % MaxCount = 900;
while count < isATime                     % ishandle( p )  % 一个死循环
    count = count+1;
    
    dat = get( hfig, 'userdata' );  % 获取数据
    ii = mod( ii + dat.dt, 2 * pi );
    
    u = TPhi.* ( dat.DefScale * sin( ii ) ); % 对模态矩阵进行插值，求的当前时刻变形量
    set( p, 'vertices', verts + u ); % 改变节点坐标，这是动画关键
    dispVector = dat.DefScale * sqrt( TPhi( :, 1 ) .^ 2 + TPhi( :, 2 ) .^ 2 + TPhi( :, 3 ) .^ 2 ); % 计算变形矢量
    faceVertexCData = GetVertCData2( dat.FaceVertexItem, dat.DefScale * TPhi, u, dispVector ) ; % 计算当前变形量对应的颜色值与最大变形量
    set( p, 'FaceVertexCData', faceVertexCData , 'EdgeColor', 'interp' ); % 更新颜色显示
    if ifColorBar
        maxDef = max(u(:));
        maxDef = maxDef / dat.DefScale;
        tickLabel = 0 : maxDef / 10 : maxDef; % 计算colorbar刻度并显示
        colorbar( 'YTickLabel', { num2str( tickLabel( 1 ) ), num2str( tickLabel( 2 ) ), num2str( tickLabel( 3 ) ), ...
            num2str( tickLabel( 4 ) ), num2str( tickLabel( 5 ) ), num2str( tickLabel( 6 ) ), num2str( tickLabel( 7 ) ), ...
            num2str( tickLabel( 8 ) ), num2str( tickLabel( 9 ) ), num2str( tickLabel( 10 ) ), num2str( tickLabel( 11 ) ) } );
    end
    drawnow
    pause(0.0001);
end % while










