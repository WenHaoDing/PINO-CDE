clear;
clc;
[K,M,Mappings,MMat,MatA,Eigens,outside_faces,Nodes,eps,...
 ModeNumMax,W2,Eigens2,u_bridge]=Subroutine_FlexBodyImport(1);
epis=0.001;
[TrackDOFs,Nloc]=Subroutine_InteractionSearch(epis,Nodes);
for Pick=7:7
n=499;
Batchsize=50;
N_Start=1:Batchsize:n;
N_End=N_Start+Batchsize;N_End(end)=n;
Batch_Input=zeros(Batchsize,5000,1+3*2);
Batch_Output=zeros(Batchsize,5000,ModeNumMax*3);
for ib=1:Batchsize
tempo=strcat(num2str(ib/Batchsize*100),'%');
disp(tempo);
Index=N_Start(Pick)+ib-1;
% [Input,Output,flag]=DataGeneratorV2(outside_faces,Nodes,ModeNumMax,W2,Eigens2,TrackDOFs,Nloc);
[Input,Output,flag]=DataGeneratorV2(outside_faces,Nodes,ModeNumMax,W2,Eigens2,TrackDOFs,Nloc,Index);
Batch_Input(ib,:,:)=Input;
Batch_Output(ib,:,:)=Output;
Flag(ib)=flag;
end
Path='G:';
FileName=['Batch',num2str(Pick),'.mat'];FileName=[Path,'\',FileName];
save(FileName,'Batch_Input','Batch_Output');
end
