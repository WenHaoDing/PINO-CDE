function [Force_X,Force_Y,Force_Z] = Force_KV_ForceElementV2(Signal,DisRecord_Store,VelRecord_Store,AclRecord_Store,Flag)
%% Relative Displacement
Static=zeros(size(DisRecord_Store,1),1);
Dis_X=Signal(1).*ones(size(DisRecord_Store,1),1);
Dis_Y=Signal(2).*ones(size(DisRecord_Store,1),1);
Dis_Z=Signal(3).*ones(size(DisRecord_Store,1),1);
Vel_X=Signal(4).*ones(size(DisRecord_Store,1),1);
Vel_Y=Signal(5).*ones(size(DisRecord_Store,1),1);
Vel_Z=Signal(6).*ones(size(DisRecord_Store,1),1);
Acl_X=Signal(7).*ones(size(DisRecord_Store,1),1);
Acl_Y=Signal(8).*ones(size(DisRecord_Store,1),1);
Acl_Z=Signal(9).*ones(size(DisRecord_Store,1),1);
RD_X=DisRecord_Store(:,1)-Dis_X;
RD_Y=DisRecord_Store(:,2)-Dis_Y;
RD_Z=DisRecord_Store(:,3)-Dis_Z;
RV_X=VelRecord_Store(:,1)-Vel_X;
RV_Y=VelRecord_Store(:,2)-Vel_Y;
RV_Z=VelRecord_Store(:,3)-Vel_Z;
RA_X=AclRecord_Store(:,1)-Acl_X;
RA_Y=AclRecord_Store(:,2)-Acl_Y;
RA_Z=AclRecord_Store(:,3)-Acl_Z;
%% Forced Contact Element Parameters
KD_X=5e6;
KD_Y=5e6;
KD_Z=5e6;
KDV_X=1e6;
KDV_Y=1e6;
KDV_Z=1e6;
% if Flag==1
%     KDV_X=1.5e6;
%     KDV_Y=1.5e6;
%     KDV_Z=1.5e6;
% end
%% Force element output
% Force_X=KD_X*RD_X;
% Force_Y=KD_Y*RD_Y;
% Force_Z=KD_Z*RD_Z;
Force_X=KD_X*RD_X+KDV_X*RV_X;
Force_Y=KD_Y*RD_Y+KDV_Y*RV_Y;
Force_Z=KD_Z*RD_Z+KDV_Z*RV_Z;
end

