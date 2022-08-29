clear;
clc;
Path='';
Pick=1;
dt=1e-3;
load('F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\data\Project_HM\5000_V3.mat','MeanV','StdV');

% LossHistory=importdata([Path,'\','LossHistory.txt']);
index=525;
Data=importdata([Path,'\','Performance',num2str(index),'.txt']);
Dis_flex1=Data(:,1:15);
Dis_flex2=Data(:,16:25);
GTDis_flex1=Data(:,44:58);
GTDis_flex2=Data(:,59:68);
for i=1:size(Dis_flex1,2)
    Dis_flex1(:,i)=Dis_flex1(:,i)*StdV(10+i)+MeanV(10+i);
    GTDis_flex1(:,i)=GTDis_flex1(:,i)*StdV(10+i)+MeanV(10+i);
end
for i=1:size(Dis_flex2,2)
    Dis_flex2(:,i)=Dis_flex2(:,i)*StdV(10+15+i)+MeanV(10+15+i);
    GTDis_flex2(:,i)=GTDis_flex2(:,i)*StdV(10+15+i)+MeanV(10+15+i);
end    
    
    
Vel_flex1=(Dis_flex1(3:end,:)-Dis_flex1(1:end-2,:))./(2*dt);
GTVel_flex1=(GTDis_flex1(3:end,:)-GTDis_flex1(1:end-2,:))./(2*dt);

Vel_flex2=(Dis_flex2(3:end,:)-Dis_flex2(1:end-2,:))./(2*dt);
GTVel_flex2=(GTDis_flex2(3:end,:)-GTDis_flex2(1:end-2,:))./(2*dt);

Acl_flex1=(Dis_flex1(3:end,:)-2.*Dis_flex1(2:end-1,:)+Dis_flex1(1:end-2,:))./(dt^2);
GTAcl_flex1=(GTDis_flex1(3:end,:)-2.*GTDis_flex1(2:end-1,:)+GTDis_flex1(1:end-2,:))./(dt^2);

Acl_flex2=(Dis_flex2(3:end,:)-2.*Dis_flex2(2:end-1,:)+Dis_flex2(1:end-2,:))./(dt^2);
GTAcl_flex2=(GTDis_flex2(3:end,:)-2.*GTDis_flex2(2:end-1,:)+GTDis_flex2(1:end-2,:))./(dt^2);
[K1,M1,Mappings1,MMat1,MatA1,Eigens1,outside_faces1,Nodes1,eps,ModeNumMax1,W21,Eigens21,u_bridge1,...
 K2,M2,Mappings2,MMat2,MatA2,Eigens2,outside_faces2,Nodes2,ModeNumMax2,W22,Eigens22,u_bridge2]=Subroutine_FlexBodyImport(1);
Nodes2(:,end)=Nodes2(:,end)-0.1;

Tstep=0.001;
local_count=0;
% Resource1 = W21 * Dis_flex1';
% GTResource1 = W21 * GTDis_flex1';
% Resource2 = W22 * Dis_flex2';
% GTResource2 = W22 * GTDis_flex2';

% Resource1 = W21 * Vel_flex1';
% GTResource1 = W21 * GTVel_flex1';
% Resource2 = W22 * Vel_flex2';
% GTResource2 = W22 * GTVel_flex2';

Resource1 = W21 * Acl_flex1';
GTResource1 = W21 * GTAcl_flex1';
Resource2 = W22 * Acl_flex2';
GTResource2 = W22 * GTAcl_flex2';
% for ii=1:size(Resource1,2)
%     if ii==529
%         A=1;
%     end
% Dfai_c1 = Resource1(:,ii); 
% Dfai_c_GT1 = GTResource1(:,ii);
% Dfai_c2 = Resource2(:,ii); 
% Dfai_c_GT2 = GTResource2(:,ii);
%     scale_fcator = 50;       
% %     scale_fcator = 0.0001;       
%     if ~rem(ii*Tstep,0.001)
%         vi1 = reshape(Dfai_c1,3,[])';
%         vi1_GT = reshape(Dfai_c_GT1,3,[])';
%         vi2 = reshape(Dfai_c2,3,[])';
%         vi2_GT = reshape(Dfai_c_GT2,3,[])';
%         local_count = local_count+1;
%         if local_count == 1
% %             scale_fcator = 1e-2/max(Dfai_c(:))*1e-1;
% %             scale_fcator = 2e5;
%             [p1,p2,hfig] = plot3_DisplaceModel2V2(vi1,outside_faces1,Nodes1,scale_fcator,0,0,1,vi2,outside_faces2,Nodes2);
%             [p1_GT,p2_GT,hfig_GT] = plot3_DisplaceModel2V2(vi1_GT,outside_faces1,Nodes1,scale_fcator,0,0,2,vi2_GT,outside_faces2,Nodes2);
%         elseif local_count < 10
%             plot3_DisplaceModel_Version2V2(hfig,p1,vi1,Nodes1,scale_fcator,0,p2,vi2,Nodes2)
%             plot3_DisplaceModel_Version2V2(hfig_GT,p1_GT,vi1_GT,Nodes1,scale_fcator,0,p2_GT,vi2_GT,Nodes2)
%         else
% %             scale_fcator = Dfai_c*1e3;
% %             scale_fcator = 1e3/max(Dfai_c(:));
%             plot3_DisplaceModel_Version2V2(hfig,p1,vi1,Nodes1,scale_fcator,0,p2,vi2,Nodes2)
%             plot3_DisplaceModel_Version2V2(hfig_GT,p1_GT,vi1_GT,Nodes1,scale_fcator,0,p2_GT,vi2_GT,Nodes2)
%         end
%         pause(0.0001)
%     end
% end

for ii=1:size(Resource1,2)
    if ii==529
        A=1;
    end
Dfai_c1 = Resource1(:,ii); 
Dfai_c_GT1 = GTResource1(:,ii);
Dfai_c2 = Resource2(:,ii); 
Dfai_c_GT2 = GTResource2(:,ii);
    scale_fcator = 1e-4;       
    if ~rem(ii*Tstep,0.001)
        vi1 = reshape(Dfai_c1,3,[])';
        vi1_GT = reshape(Dfai_c_GT1,3,[])';
        vi2 = reshape(Dfai_c2,3,[])';
        vi2_GT = reshape(Dfai_c_GT2,3,[])';
        local_count = local_count+1;
        if local_count == 1
%             scale_fcator = 1e-2/max(Dfai_c(:))*1e-1;
%             scale_fcator = 2e5;
            [p1,p2,hfig] = plot3_DisplaceModel2V2(vi1,outside_faces1,Nodes1,scale_fcator,0,0,1,vi2,outside_faces2,Nodes2);
            [p1_GT,p2_GT,hfig_GT] = plot3_DisplaceModel2V2(vi1_GT,outside_faces1,Nodes1,scale_fcator,0,0,2,vi2_GT,outside_faces2,Nodes2);
        elseif local_count < 10
            plot3_DisplaceModel_Version2V2(hfig,p1,vi1,Nodes1,scale_fcator,0,p2,vi2,Nodes2)
            plot3_DisplaceModel_Version2V2(hfig_GT,p1_GT,vi1_GT,Nodes1,scale_fcator,0,p2_GT,vi2_GT,Nodes2)
        else
%             scale_fcator = Dfai_c*1e3;
%             scale_fcator = 1e3/max(Dfai_c(:));
            plot3_DisplaceModel_Version2V2(hfig,p1,vi1,Nodes1,scale_fcator,0,p2,vi2,Nodes2)
            plot3_DisplaceModel_Version2V2(hfig_GT,p1_GT,vi1_GT,Nodes1,scale_fcator,0,p2_GT,vi2_GT,Nodes2)
        end
        pause(0.0001)
    end
end
