function  faceVertexCData = GetVertCData2( markflag, TPhi, u, dispVector ,Reference )

m = length( dispVector );
faceVertexCData = zeros( m, 3 );
ToOne = 1;
switch markflag
    case 1
        Vector = dispVector;
        udisp = sqrt( u( :, 1 ) .^ 2 + u( :, 2 ) .^ 2 + u( :, 3 ) .^ 2 );
        if ToOne==1
            udisp=udisp-min(udisp);
%             udisp=0.1*udisp/(max(udisp)-min(udisp));
        end
    case 2
        Vector = TPhi( :, 1 );
        udisp = u( :, 1 );
    case 3
        Vector = TPhi( :, 2 );
        udisp = u( :, 2 );
    case 4
        Vector = TPhi( :, 3 );
        udisp = u( :, 3 );
    case 5
        Vector = max(TPhi,[],1);
        udisp = u(:,1);
    case 6
        Vector = TPhi(:,1)./max(TPhi).*0.9;
        udisp = u(:,1);
        
end
if nargin <= 4
    VectorMax = max( Vector );
else
    VectorMax = Reference;
end

Vector = abs( udisp / VectorMax );
if ToOne==1
Vector = Vector/(max(Vector)-min(Vector));
Vector = Vector-min(Vector);
end

Ntemp = 20;
colors = jet(Ntemp);
% colors = parula(Ntemp);
% colors = hsv(Ntemp);cool
scal = linspace(0,1,Ntemp+1);
scal(end) = inf;
for itr = 1:Ntemp
    bo = Vector >= scal(itr) & Vector <= scal(itr+1);
    faceVertexCData(bo,:) = ones(sum(bo),1)*colors(itr,:);
end