clear;
clc;
Path='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\checkpoints\BSARunner\Experiment1\PDEM';
CaseNumber=499;
mode='no';
if strcmp(mode,'eval')==1
for nC=1:CaseNumber
    nC
    FileName=[Path,'\Evolution',num2str(nC),'.mat'];
    load(FileName);
    if nC==1
        PMatrix_Fusion=PMatrix;
    else
        PMatrix_Fusion=PMatrix_Fusion+PMatrix;
        Spy=squeeze(PMatrix(1,:,:));
    end
end
save('FusedReliability.mat','PMatrix_Fusion');
end
load('FusedReliability.mat');
load('Span_499V2.mat');
PMatrix_Fusion=PMatrix_Fusion./CaseNumber;
RSpan=SpanR-SpanL;
SpanL=SpanL-0.2.*RSpan;
SpanR=SpanR+0.2.*RSpan;
SpaceDPI=size(PMatrix_Fusion,3);
for i=1:size(PMatrix_Fusion,1)
    SpaceTick(:,i)=linspace(SpanL(i),SpanR(i),SpaceDPI)';
    MapStep(i)=SpaceTick(2,i)-SpaceTick(1,i);
    PMatrix_Fusion(i,:,:)=PMatrix_Fusion(i,:,:)./MapStep(i);
end
TimeTick=linspace(0.2,5,size(PMatrix_Fusion,2));

In=5;
DPI=10;
DPITime=TimeTick(1:DPI:end);
[xx,yy]=meshgrid(SpaceTick(:,In)',DPITime);
figure(1)
plot1 = mesh(xx,yy,squeeze(PMatrix_Fusion(In,1:DPI:end,:)));
PMatrix_Fusion_PDEM=PMatrix_Fusion(:,1:DPI:end,:);


load('FusedReliability_PINO_MBD_49999.mat');
load('Span_499V2.mat');
RSpan=SpanR-SpanL;
SpanL=SpanL-0.2.*RSpan;
SpanR=SpanR+0.2.*RSpan;
SpaceDPI=size(PMatrix_Fusion,3);
for i=1:size(PMatrix_Fusion,1)
    SpaceTick(:,i)=linspace(SpanL(i),SpanR(i),SpaceDPI)';
end
TimeTick=linspace(0.2,5,size(PMatrix_Fusion,2));

DPI=1;
DPITime=TimeTick(1:DPI:end);
[xx,yy]=meshgrid(SpaceTick(:,In)',DPITime);
figure(2)
plot2 = mesh(xx,yy,squeeze(PMatrix_Fusion(In,1:DPI:end,:)));

TimeInspect=225;
figure(3)
% save('PostProcessedProbability.mat','PMatrix_Fusion','PMatrix_Fusion_PDEM','SpaceTick','-V7.3');
for TimeInspect=1:1
plot(SpaceTick(:,In)',squeeze(PMatrix_Fusion(In,TimeInspect,:)),'b','LineWidth',0.5);hold on;
plot(SpaceTick(:,In)',squeeze(PMatrix_Fusion_PDEM(In,TimeInspect,:)),'r','LineWidth',1);hold off;
pause(0.1);
end
