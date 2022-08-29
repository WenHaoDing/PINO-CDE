function plot3_DisplaceModel_Version2(hfig,p,TPhi,verts,DefScale,ifColorBar)

if nargin < 5
    ifColorBar = false;
end
if size(verts,2) == 4
    verts(:,1) = [];
end

%%

data.FaceVertexItem = 1;
data.DefScale = DefScale;
data.pause = false;
data.ModeChanged = false;

set( hfig, 'userdata', data );  % init displayed mode

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




