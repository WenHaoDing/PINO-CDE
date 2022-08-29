clear;
clc;
Path='H:\Pack4\V4';
Ruler=1;
DPI=1;
version=4;
for Pick=9:9
    Pick
    FileName=['Batch',num2str(Pick),'.mat'];FileName=[Path,'\',FileName];
    load(FileName);
    TemNum=size(Batch_Input,1);
    input(Ruler:Ruler+TemNum-1,:,:)=Batch_Input(:,DPI:DPI:end,:);
    output(Ruler:Ruler+TemNum-1,:,:)=Batch_Output(:,DPI:DPI:end,[1:25 76:93 26:50 94:111 51:75 112:129]);
    % Stack all physical quantities together (X V A)
    Ruler=Ruler+TemNum;
end
%% 数据补充（训练大数）
Fn=500;
n=size(input,1);
input(n+1:Fn,:,:)=input(1:Fn-n,:,:);
output(n+1:Fn,:,:)=output(1:Fn-n,:,:);
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


load('F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\data\Project_HM\5000_V3.mat','MaxV','MinV','MeanV','StdV');
InDim=10;
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
for i=1:100:size(input, 1)
    Tix=21;
    Tem=output(i,:,Tix);
    Tem=squeeze(Tem);
    plot(Tem');hold on;
end
Path='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\data\Project_HM';
% Path='L:\Data\IrreOnlyData';
FileName=[num2str(n),'_V',num2str(version),'.mat'];FileName=[Path,'\',FileName];
save(FileName,'input','output','MeanV','MaxV','MinV','StdV','-v7.3');

%% The effect of DPI on the FFT result
clearvars -except input output MeanV MaxV MinV StdV