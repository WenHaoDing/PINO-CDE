clear;
clc;
%% PART0.控制面板
%% SEC1.文件信息
InputPath='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\checkpoints\BSARunner\Experiment1\eval';
FileName='eval_499.mat';FileName=[InputPath,'\',FileName];
load(FileName);
load('F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\data\Project_BSA\150.mat','MeanV','StdV');
output=double(output);
Path='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\Upload\Matlab codes\BSA\';
load([Path,'Building_200.mat'],'MMat','Nodes');
W2=MMat(:,1:200);
InEle=[37717 37634 31245 30085 40374 1769 43729];
SNumber=length(InEle);
load([Path,'EBStoreV2.mat']);
EBStoreV2=EBStoreV2(InEle,:,:);
SPYE=load([Path,'Info_Element_Node.txt']);                                       
EList=SPYE(:,1);                                                           
NodeList=SPYE(InEle,7:14);
% Data normalization
for i=1:size(output,3)
    output(:,:,i)=output(:,:,i)*StdV(7+i)+MeanV(7+i);
end
% output=output(:,51:end,:);
CaseNumber=size(output,1);
j=1;
PTensor=cell(CaseNumber,size(output,3));
% Inform=zeros(CaseNumber,200,3);
%% PART1.Probability evolution operation
Batchsize=50;
Switch_SpanComputation='Off';
if strcmp(Switch_SpanComputation,'On')==1
for nC=1:499
    nC
    XSequence=squeeze(output(nC,:,:))';
%     VSequence=(XSequence(:,3:end)-XSequence(:,1:end-2))/(2*0.005);
%     XSequence=XSequence(:,2:end-1);
    %% 位移场重构
    Resource = W2 * XSequence;
    StressStore=zeros(size(XSequence,2),length(InEle));
    u1=zeros(24,size(XSequence,2));
    for ie=1:size(EBStoreV2,1)
        for ijn=1:8
            NodeNum=NodeList(ie,ijn);
            u1(3*(ijn-1)+1:3*ijn,:)=Resource((NodeNum-1)*3+1:NodeNum*3,:);
        end
        w1=(squeeze(EBStoreV2(ie,:,:))*u1)';
        w1=max(w1(:,1:3),[],2);
        StressStore(:,ie)=w1;
    end
XSequence=StressStore';
SpanL(nC,:)=min(XSequence');
SpanR(nC,:)=max(XSequence');
end
SpanL=min(SpanL);
SpanR=max(SpanR);
save('Span_499V2.mat','SpanL','SpanR');
else
 load('Span_499V2.mat');
end
RSpan=SpanR-SpanL;
SpanL=SpanL-0.2.*RSpan;
SpanR=SpanR+0.2.*RSpan;
for nC=446:499
    nC
    %% SEC1.数据导入及预处理
    XSequence=squeeze(output(nC,:,:))';
%     VSequence=(XSequence(:,3:end)-XSequence(:,1:end-2))/(2*0.005);
%     XSequence=XSequence(:,2:end-1);
    %% 位移场重构
    Resource = W2 * XSequence;
    StressStore=zeros(size(XSequence,2),length(InEle));
    u1=zeros(24,size(XSequence,2));
    for ie=1:size(EBStoreV2,1)
        for ijn=1:8
            NodeNum=NodeList(ie,ijn);
            u1(3*(ijn-1)+1:3*ijn,:)=Resource((NodeNum-1)*3+1:NodeNum*3,:);
        end
        w1=(squeeze(EBStoreV2(ie,:,:))*u1)';
        w1=max(w1(:,1:3),[],2);
        StressStore(:,ie)=w1;
    end
XSequence=StressStore';
 VSequence=(XSequence(:,3:end)-XSequence(:,1:end-2))/(2*0.005);
 XSequence=XSequence(:,2:end-1);
%     VSequence=squeeze(output(nC,:,j+200))';
    %% 有限差分的差分计算范围决定了计算效率，通过切割进行计算范围的缩减
%% SEC2.有限差分控制
% K1.设置参数
% Span=max(XSequence,[],2)-min(XSequence,[],2);
% Center=0.5*(max(XSequence,[],2)+min(XSequence,[],2));
% Xmin=Center-(0.5+0.1)*Span;
% Xmax=Center+(0.5+0.1)*Span;
Xmax=SpanR';
Xmin=SpanL';
DPI=1500;
dx=(Xmax-Xmin)/DPI;
dt=5e-5;                                                                   %时间网格                    
Time=957*0.005;
XDimension=DPI;                                          %空间网格纬度
TDimension=round(Time/dt);                                                 %时间网格纬度
% K2.差分变量参数
rL=dt./dx;                                                                  %网格比
RDPI=10;
PTimeSequence=[(0.205+dt):dt:(0.205+dt*TDimension)]';                                      %概率空间时间网格切分序列
for p=1:SNumber
    PXSequence(:,p)=[Xmin(p):dx(p):Xmin(p)+dx(p)*(XDimension-1)]';
end
for p=1:SNumber
    InMax=max(StressStore(:,p));
    [~,Index(p)]=min(abs(PXSequence(:,p)-InMax));
    Index(p)=Index(p)+200;
    if Index(p)>=DPI
        Index(p)=DPI-1;
    end

end
SDPI=max(Index);
SXSequence=PXSequence(1:Index(p),:);
Stitch=DPI-SDPI;
XDimension=SDPI;                                                            %空间网格纬度
PMatrix=zeros(SNumber,round(TDimension/RDPI),DPI);                  %概率空间矩阵预分配
% PXSequence=[Xmin:dx:Xmin+dx*(XDimension-1)]';                              %概率空间空间网格切分序列
%% PART1.数据处理
% SEC1.数据预处理
% FileName=['工况',num2str(nC),'.txt'];FileName=[InputPath,'\',FileName];
% Data_Cycle=importdata(FileName);
% Dimension=length(Data_Cycle);
% Factor=1000;                                                               %单位转换为mm
TimeSequence=[0.205:0.005:(0.205+Time)]';
X=XSequence;
V=VSequence;
% A=Data_Cycle(:,4);A=Factor.*A;
% SEC2.流场信息插值
TimeRatio=round(length(PTimeSequence)/length(TimeSequence));
X=spline(TimeSequence,X,PTimeSequence);
V=spline(TimeSequence,V,PTimeSequence);
% A=spline(TimeSequence,A,PTimeSequence);
%% PART2.TVD格式有限差分
% SEC1.边界条件处理及初始条件赋值
X_inital=X(:,1);
Memory_This=zeros(length(InEle),XDimension);
Memory_Next=zeros(length(InEle),XDimension);
for p=1:SNumber
[~,index]=min(abs(SXSequence(:,p)-X_inital(p)));
Memory_This(p,index)=1;
Memory_Next(p,index)=1;
% PMatrix(p,1,index)=1;
% PMatrix(p,2,index)=1;
end
for it=2:TDimension
%     it/TDimension
    Memory_This=Memory_Next;
    for ix=3:XDimension-2
        %ix
        %% 变量抽取
%         pj2k=PMatrix(:,it,ix+2);
%         pj1k=PMatrix(:,it,ix+1);
%         pjk=PMatrix(:,it,ix);
%         pjm1k=PMatrix(:,it,ix-1);
%         pjm2k=PMatrix(:,it,ix-2);
        pj2k=Memory_This(:,ix+2);
        pj1k=Memory_This(:,ix+1);
        pjk=Memory_This(:,ix);
        pjm1k=Memory_This(:,ix-1);
        pjm2k=Memory_This(:,ix-2);
        
        rjp_p12=zeros(1,SNumber);
        rjn_p12=zeros(1,SNumber);
        rjp_n12=zeros(1,SNumber);
        rjn_n12=zeros(1,SNumber);
        for p=1:SNumber
        if pj1k(p)==pjk(p)
            if pj2k(p)==pj1k(p)
                rjp_p12(p)=1;
            else
                rjp_p12(p)=2;
            end
            if pjk(p)==pjm1k(p)
                rjn_p12(p)=1;
            else
                rjn_p12(p)=2;
            end   
        else
        rjp_p12(p)=(pj2k(p)-pj1k(p))/(pj1k(p)-pjk(p));
        rjn_p12(p)=(pjk(p)-pjm1k(p))/(pj1k(p)-pjk(p));
        end

%         if pj1k==pjk
%             if pjk==pjm1k
%                 rjn_p12=1;
%             else
%                 rjn_p12=2;
%             end
%         else
%         rjn_p12=(pjk-pjm1k)/(pj1k-pjk);
%         end
        if pjk(p)==pjm1k(p)
            if pj1k(p)==pjk(p)
                rjp_n12(p)=1;
            else
                rjp_n12(p)=2;
            end
            if pjm1k(p)==pjm2k(p)
                rjn_n12(p)=1;
            else
                rjn_n12(p)=2;
            end
        else
        rjp_n12(p)=(pj1k(p)-pjk(p))/(pjk(p)-pjm1k(p));
        rjn_n12(p)=(pjm1k(p)-pjm2k(p))/(pjk(p)-pjm1k(p));
        end
        end
%         if pjk==pjm1k
%             if pjm1k==pjm2k
%                 rjn_n12=1;
%             else
%                 rjn_n12=2;
%             end
%         else
%         rjn_n12=(pjm1k-pjm2k)/(pjk-pjm1k);
%         end
        gk=0.5.*(V(:,it-1)+V(:,it));
        gka=abs(gk);                                                       %gk绝对值
        [Psi1]=Flux_Limiter_Psi_Batch(rjp_p12,rjn_p12,gk);                       %第一流动阀门
        [Psi2]=Flux_Limiter_Psi_Batch(rjp_n12,rjn_n12,gk);                       %第二流动阀门
        Project1=pjk;
        Project2=rL.*(0.5.*(gk+gka).*(pjk-pjm1k)+0.5.*(gk-gka).*(pj1k-pjk));
        Project3=0.5.*(1-abs(rL.*gk)).*abs(rL.*gk).*(Psi1.*(pj1k-pjk)-Psi2.*(pjk-pjm1k));
        FDM_Result=Project1-Project2-Project3;                             %本步有限差分法计算结果
        Memory_Next(:,ix)=FDM_Result;
%         PMatrix(:,it+1,ix)=FDM_Result;
    end
   if max(max(Memory_Next))>=1.1
       print('The result does not converge, please reduce the integration step size');
       break
   end
   % clear consumption
   if rem(it,RDPI)==0
       Filter=[Memory_This zeros(length(InEle),Stitch)];
       Filter(Filter<1e-3)=0;
       PMatrix(:,it/RDPI,:)=Filter;
   end
end
%% 存储PDEM分析结果
% 处理结果，5%最大值以下信息丢弃，存储于系数矩阵中
% Process=PMatrix(TimeRatio:TimeRatio:end,:);
% Process(Process<(max(Process)/200))=0;
% Process=sparse(Process);
% PTensor{nC,j}=Process;
% Inform(nC,j,:)=[Xmin dx Xmin+dx*(XDimension-1)];
Information=[Xmin dx Xmin+dx*(XDimension-1)];
SavePath='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\checkpoints\BSARunner\Experiment1\PDEM';
FileName=[SavePath,'\Evolution',num2str(nC),'.mat'];
save(FileName,'PMatrix','Information');
% FileName_Cycle=['Case',num2str(nC),'.mat'];FileName_Cycle=[SavePath,'\',FileName_Cycle];
% save(FileName_Cycle,'PMatrix');
end


% Inspection Pot
DPI=10;
SPYmode=1;
SPYPmatrix=squeeze(PMatrix(SPYmode,1:DPI:end,:));
mesh(SPYPmatrix);


