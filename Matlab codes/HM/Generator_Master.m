clear;
clc;
% [~,~,~,~,~,~,~,~,~,...
%  ModeNumMax,~,~,~]=Subroutine_FlexBodyImport(1);
for Pick=2:2
n=1000;
Batchsize=500;
N_Start=1:Batchsize:n;
N_End=N_Start+Batchsize;N_End(end)=n;
Batch_Input=zeros(Batchsize,1000,1+9);
Batch_Output=zeros(Batchsize,1000,129);
for ib=1:Batchsize
tempo=strcat(num2str(ib/Batchsize*100),'%');
disp(tempo);
Index=N_Start(Pick)+ib-1;
[Input,Output]=DataGenerator();
Batch_Input(ib,:,:)=Input;
Batch_Output(ib,:,:)=Output;
end
Path='G:';
FileName=['Batch',num2str(Pick),'.mat'];FileName=[Path,'\',FileName];
save(FileName,'Batch_Input','Batch_Output');
end
