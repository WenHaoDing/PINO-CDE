function [Force_X1,Force_Y1,Force_Z1,...
          Force_X2,Force_Y2,Force_Z2,...
          Force_X3,Force_Y3,Force_Z3] = Force_KV_ForceElement(K,C,X_moment,V_moment,DisRecord_Store,VelRecord_Store)
%% Relative Displacement For Fisrt Neck
X_moment1=X_moment(1:9);
V_moment1=V_moment(1:9);
X_moment2=X_moment(10:end);
V_moment2=V_moment(10:end);
RD_X1=DisRecord_Store(1:3,1)-X_moment1(1:3)';
RD_Y1=DisRecord_Store(1:3,2)-X_moment1(4:6)';
RD_Z1=DisRecord_Store(1:3,3)-X_moment1(7:9)';
RV_X1=VelRecord_Store(1:3,1)-V_moment1(1:3)';
RV_Y1=VelRecord_Store(1:3,2)-V_moment1(4:6)';
RV_Z1=VelRecord_Store(1:3,3)-V_moment1(7:9)';
%% RD For Second Neck-Inverse
RD_X2=DisRecord_Store(4:6,1)-X_moment2(1:3)';
RD_Y2=DisRecord_Store(4:6,2)-X_moment2(4:6)';
RD_Z2=DisRecord_Store(4:6,3)-X_moment2(7:9)';
RV_X2=VelRecord_Store(4:6,1)-V_moment2(1:3)';
RV_Y2=VelRecord_Store(4:6,2)-V_moment2(4:6)';
RV_Z2=VelRecord_Store(4:6,3)-V_moment2(7:9)';
%% RD For Third Neck
RD_X3=DisRecord_Store(7:9,1)-X_moment2(1:3)';
RD_Y3=DisRecord_Store(7:9,2)-X_moment2(4:6)';
RD_Z3=DisRecord_Store(7:9,3)-X_moment2(7:9)';
RV_X3=VelRecord_Store(7:9,1)-V_moment2(1:3)';
RV_Y3=VelRecord_Store(7:9,2)-V_moment2(4:6)';
RV_Z3=VelRecord_Store(7:9,3)-V_moment2(7:9)';
%% Force element output
K1=K(1);C1=C(1);
K2=K(2);C2=C(2);
Force_X1=K1.*RD_X1+C1.*RV_X1;
Force_Y1=K1.*RD_Y1+C1.*RV_Y1;
Force_Z1=K1.*RD_Z1+C1.*RV_Z1;
Force_X2=K2.*RD_X2+C2.*RV_X2;
Force_Y2=K2.*RD_Y2+C2.*RV_Y2;
Force_Z2=K2.*RD_Z2+C2.*RV_Z2;
Force_X3=K2.*RD_X3+C2.*RV_X3;
Force_Y3=K2.*RD_Y3+C2.*RV_Y3;
Force_Z3=K2.*RD_Z3+C2.*RV_Z3;
end

