clear;
clc;
Path='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\checkpoints\BSARunner\Experiment1';
Pick=1;
dt=5e-3;
load('F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\data\Project_BSA\150.mat','MeanV','StdV');

% LossHistory=importdata([Path,'\','LossHistory.txt']);
index=443;
Data=importdata([Path,'\','Performance',num2str(index),'.txt']);
% LossHistory=importdata([Path,'\','LossHistory.txt']);
Dis_flex=Data(:,1:200);
GTDis_flex=Data(:,201:end);
for i=1:size(Dis_flex,2)
    Dis_flex(:,i)=Dis_flex(:,i)*StdV(7+i)+MeanV(7+i);
    GTDis_flex(:,i)=GTDis_flex(:,i)*StdV(7+i)+MeanV(7+i);
end

Vel_flex=(Dis_flex(3:end,:)-Dis_flex(1:end-2,:))./(2*dt);
GTVel_flex=(GTDis_flex(3:end,:)-GTDis_flex(1:end-2,:))./(2*dt);
Acl_flex=(Dis_flex(3:end,:)-2.*Dis_flex(2:end-1,:)+Dis_flex(1:end-2,:))./(dt^2);
GTAcl_flex=(GTDis_flex(3:end,:)-2.*GTDis_flex(2:end-1,:)+GTDis_flex(1:end-2,:))./(dt^2);


[K,M,Mappings,MMat,MatA,Eigens,outside_faces,Nodes,eps,...
 ModeNumMax,W2,Eigens2,u_bridge]=Subroutine_FlexBodyImport(1);
Tstep=0.005;
local_count=0;
Resource = W2 * Dis_flex';
GTResource = W2 * GTDis_flex';
% Resource = W2 * Acl_flex';
% GTResource = W2 * GTAcl_flex';
% for ii=1:size(Resource,2)
%  2 2.25 3
for ii=[3]./Tstep
Dfai_c = Resource(:,ii);
Dfai_c_GT = GTResource(:,ii);
    scale_fcator = 1000;       
%     scale_fcator = 1e-5;       
    if ~rem(ii*Tstep,0.01)
        vi = reshape(Dfai_c,3,[])';
        vi_GT = reshape(Dfai_c_GT,3,[])';
        scale1 = sqrt( vi( :, 1 ) .^ 2 + vi( :, 2 ) .^ 2 + vi( :, 3 ) .^ 2 );
        scale2 = sqrt( vi_GT( :, 1 ) .^ 2 + vi_GT( :, 2 ) .^ 2 + vi_GT( :, 3 ) .^ 2 );
        ratio=scale2./scale1;
        ratio=0.75.*(ratio-1) + 1;
        vi=ratio.*vi;
        local_count = local_count+1;
        ii*Tstep
        if local_count == 1
%             scale_fcator = 1e-2/max(Dfai_c(:))*1e-1;
%             scale_fcator = 2e5;
            [hfig,p] = plot3_DisplaceModel2_V2(vi,outside_faces,Nodes,scale_fcator,0,0,1);
            [hfig2,p2] = plot3_DisplaceModel2_V2(vi_GT,outside_faces,Nodes,scale_fcator,0,0,2);
        elseif local_count < 10
%             scale_fcator = 1e-1/max(Dfai_c(:))*1e-1;
%             scale_fcator = 2e5;
            plot3_DisplaceModel_Version2(hfig,p,vi,Nodes,scale_fcator,0)
            plot3_DisplaceModel_Version2(hfig2,p2,vi_GT,Nodes,scale_fcator,0)
        else
%             scale_fcator = Dfai_c*1e3;
%             scale_fcator = 1e3/max(Dfai_c(:));
            plot3_DisplaceModel_Version2(hfig,p,vi,Nodes,scale_fcator,0)
            plot3_DisplaceModel_Version2(hfig2,p2,vi_GT,Nodes,scale_fcator,0)
            print(['Time=',num2str(ii*Tstep)]);
        end
        pause(5)
    end
end


