function plot3_DisplaceModel_Version2V2(hfig,p1,TPhi1,verts1,DefScale,ifColorBar,p2,TPhi2,verts2)

if nargin < 5
    ifColorBar = false;
end
if size(verts1,2) == 4
    verts1(:,1) = [];
end

%%

data.FaceVertexItem = 1;
data.DefScale = DefScale;
data.pause = false;
data.ModeChanged = false;

set( hfig, 'userdata', data );  % init displayed mode

dat = get( hfig, 'userdata' );  % 获取数据
u1 = TPhi1.* ( dat.DefScale * 1 ); %
u2 = TPhi2.* ( dat.DefScale * 1 ); %
set( p1, 'vertices', verts1 + u1 ); % 改变节点坐标
set( p2, 'vertices', verts2 + u2 ); % 改变节点坐标
dispVector1 = dat.DefScale * sqrt( TPhi1( :, 1 ) .^ 2 + TPhi1( :, 2 ) .^ 2 + TPhi1( :, 3 ) .^ 2 ); % 计算变形矢量
dispVector2 = dat.DefScale * sqrt( TPhi2( :, 1 ) .^ 2 + TPhi2( :, 2 ) .^ 2 + TPhi2( :, 3 ) .^ 2 ); % 计算变形矢量
faceVertexCData1 = GetVertCData2( dat.FaceVertexItem, dat.DefScale * TPhi1, u1, dispVector1 ) ; % 计算当前变形量对应的颜色值与最大变形量
faceVertexCData2 = GetVertCData2( dat.FaceVertexItem, dat.DefScale * TPhi2, u2, dispVector2 ) ; % 计算当前变形量对应的颜色值与最大变形量
set( p1, 'FaceVertexCData', faceVertexCData1 , 'EdgeColor', 'interp' ); % 更新颜色显示
set( p2, 'FaceVertexCData', faceVertexCData2 , 'EdgeColor', 'interp' ); % 更新颜色显示
disp([num2str(1000 * max(u1(:)) / dat.DefScale)]);
disp([num2str(1000 * max(u2(:)) / dat.DefScale)]);
if ifColorBar
    maxDef = max(u1(:));
    maxDef = maxDef / dat.DefScale;
    maxDef=1000*maxDef;
        tickLabel = roundn(0 : maxDef / 5 : maxDef, -3); % 计算colorbar刻度并显示
        colorbar( 'YTickLabel', { num2str( tickLabel( 1 ) ), num2str( tickLabel( 2 ) ), num2str( tickLabel( 3 ) ), ...
            num2str( tickLabel( 4 ) ), num2str( tickLabel( 5 ) ), num2str( tickLabel( 6 ) ), num2str( tickLabel( 7 ) ), ...
            num2str( tickLabel( 8 ) ), num2str( tickLabel( 9 ) ), num2str( tickLabel( 10 ) ), num2str( tickLabel( 11 ) ) } );
end
%     maxDef = max(u1(:));
%     maxDef = maxDef / dat.DefScale;
%     maxDef=1000*maxDef;
%     disp([num2str(maxDef)]);
end
%     axis square equal auto off;




