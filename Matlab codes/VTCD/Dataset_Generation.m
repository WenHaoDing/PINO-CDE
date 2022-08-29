clear;
clc;
Path='J:\';
Ruler=1;
DPI=1;
for Pick=1:5
    Pick
    FileName=['MCM',num2str(Pick),'.mat'];FileName=[Path,'\',FileName];
    load(FileName);
    EmitSe=find(FlagStore==1);
    Batch_Input(EmitSe,:,:)=[];
    Batch_Output(EmitSe,:,:)=[];
    TemNum=size(Batch_Input,1);
    input(Ruler:Ruler+TemNum-1,:,:)=Batch_Input(:,DPI:DPI:end,:);
    output(Ruler:Ruler+TemNum-1,:,:)=Batch_Output(:,DPI:DPI:end,:);
    Ruler=Ruler+TemNum;
end
%% 数据补充（训练大数）
% Fn=10000;
n=size(input,1);
% input(n+1:Fn,:,:)=input(1:Fn-n,:,:);
% output(n+1:Fn,:,:)=output(1:Fn-n,:,:);
%% 数据归一化
for i=1:size(input,3)
    MaxV_input(i)=max(max(input(:,:,i)));
    MinV_input(i)=min(min(input(:,:,i)));
    MeanV_input(i)=mean(mean(input(:,:,i)));
    StdV_input(i)=std(reshape(squeeze(input(:,:,i)),[size(input,1)*size(input,2),1]));
end
for i=1:size(output,3)
    MaxV_output(i)=max(max(output(:,:,i)));
    MinV_output(i)=min(min(output(:,:,i)));
    MeanV_output(i)=mean(mean(output(:,:,i)));
    StdV_output(i)=std(reshape(squeeze(output(:,:,i)),[size(output,1)*size(output,2),1]));
end
%% 取消时间的归一信息
MeanV_input(1)=0;
StdV_input(1)=1;

MaxV=[MaxV_input MaxV_output];
MinV=[MinV_input MinV_output];
MeanV=[MeanV_input MeanV_output];
StdV=[StdV_input StdV_output];


load('F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\data\Project_VTCD\10000V2.mat','MaxV','MinV','MeanV','StdV');
InDim=19;
MeanV_input=MeanV(1:InDim);MeanV_output=MeanV(InDim+1:end);
StdV_input=StdV(1:InDim);StdV_output=StdV(InDim+1:end);
for i=1:size(input, 3)
    input(:,:,i)=(input(:,:,i)-MeanV_input(i))/StdV_input(i);
end
for i=1:size(output, 3)
    output(:,:,i)=(output(:,:,i)-MeanV_output(i))/StdV_output(i);
end
%% 数据维度转换
% input=permute(input,[1, 3, 2]);
% output=permute(output,[1, 3, 2]);
%% Visualization
figure(1)
for pick=1:34
for i=1:100:size(input, 1)
    Tem=output(i,:,pick);
    Tem=squeeze(Tem);
    plot(Tem');hold on;
end
pause(1);
hold off;
end
Path='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\data\Project_VTCD';
% Path='L:\Data\IrreOnlyData';
FileName='2000V2.mat';FileName=[Path,'\',FileName];
save(FileName,'input','output','MeanV','MaxV','MinV','StdV','-v7.3');

%% The effect of DPI on the FFT result
% Tix=27;
% Tem=output(100,:,Tix);
% Tem=squeeze(Tem);
% DPI=4;Tstep=DPI*1e-3;
% Time=[Tstep:Tstep:length(Tem)*1e-3]';
% DPIData = Tem(DPI:DPI:end);
% [TrueFFT,~] = FFT_Case2([[1e-3:1e-3:length(Tem)*1e-3]' Tem'],1,1);
% [DPI_FFT,~] = FFT_Case2([Time DPIData'],1,1);
% loglog(TrueFFT(:,1),TrueFFT(:,2));hold on;
% loglog(DPI_FFT(:,1),DPI_FFT(:,2));hold on;
%一个似乎非常有意义的结论：
%DPI对FFT的效果有影响，似乎采用偶数分辨率的效果优于奇数分辨率，同时
%分辨率如果大于等于5，也会对频谱造成比较可观的效果破坏
%然而，在当前这种情况下似乎不会影响，因为我本来就是在用dpi后的数据做训练
clearvars -except input output MeanV MaxV MinV StdV