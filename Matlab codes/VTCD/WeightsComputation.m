clear;
clc;
% 认为：对于不同的参数PDE，权重应当对应设置
% 加载数据，反归一化，计算不同PDE的权重
Path='F:\Pycharm\ExistingPytorch\GNN_Series\Physics-Informed Neural Operator\PINO-Project1\data\Project_VTCD';
FileName='2000V2.mat';FileName=[Path,'\',FileName];
load(FileName,'input','output','MeanV','StdV','MaxV','MinV');
n=size(input,1);
dt=0.001;
Weights=zeros(n,10);    
for DataIndex=1:n
    DataIndex/n
    Data=[squeeze(input(DataIndex,:,:)) squeeze(output(DataIndex,:,:))];
    for i=1:size(MeanV, 2)
        Data(:,i)=Data(:,i)*StdV(i)+MeanV(i);
    end     

[Time, E1, E2, E3, E4, Mc, Jc, Mt, Jt, Mw, Kp, Cp, Ks, Cs, Lc, Lt, Vc, Kkj, Ckj,...
    uZc, uBc, uZt1, uBt1, uZt2, uBt2, uZw1, uZw2, uZw3, uZw4, WIrre1, WIrre2, WIrre3, WIrre4]=deal(Data(:,1),Data(:,2),Data(:,3),Data(:,4),Data(:,5),Data(:,6),...
    Data(:,7),Data(:,8),Data(:,9),Data(:,10),Data(:,11),Data(:,12),Data(:,13),Data(:,14),Data(:,15),Data(:,16),Data(:,17),Data(:,18),...
    Data(:,19),Data(:,20),Data(:,21),Data(:,22),Data(:,23),Data(:,24),Data(:,25),Data(:,26),Data(:,27),Data(:,28),Data(:,29),Data(:,50),Data(:,51),Data(:,52),Data(:,53));

g=9.8;

%% 制作差分张量
List={'uZc', 'uBc', 'uZt1', 'uBt1', 'uZt2', 'uBt2', 'uZw1', 'uZw2', 'uZw3', 'uZw4'};
for index=1:length(List)
    var=List{index};
    eval(['d',var,'=(',var,'(3:end)-',var,'(1:end-2))/(2*dt);']);
    eval(['dd',var,'=(',var,'(3:end)-2..*',var,'(2:end-1)+',var,'(1:end-2))/(dt^2);']);
end
%% 缩减普通变量维度
List={'Time', 'E1', 'E2', 'E3', 'E4', 'Mc', 'Jc', 'Mt', 'Jt', 'Mw', 'Kp', 'Cp', 'Ks', 'Cs', 'Lc', 'Lt', 'Vc', 'Kkj', 'Ckj',...
    'uZc', 'uBc', 'uZt1', 'uBt1', 'uZt2', 'uBt2', 'uZw1', 'uZw2', 'uZw3', 'uZw4','WIrre1', 'WIrre2', 'WIrre3', 'WIrre4'};
for index=1:length(List)
    var=List{index};
    eval([var,'=',var,'(2:end-1);']);
end
%% 检查差分计算的正确性
% plot(dduZw2);hold on;
% plot(Data(2:end-1,19+21+7));hold off;
%% 对每一个物理量加入白噪声（对应该物理量的数量级的百分比的噪声）
%% 观察不同方程的PDE损失变化情况
List={'uZc', 'uBc', 'uZt1', 'uBt1', 'uZt2', 'uBt2', 'uZw1', 'uZw2', 'uZw3', 'uZw4',...
    'duZc', 'duBc', 'duZt1', 'duBt1', 'duZt2', 'duBt2', 'duZw1', 'duZw2', 'duZw3', 'duZw4',...
    'dduZc', 'dduBc', 'dduZt1', 'dduBt1', 'dduZt2', 'dduBt2', 'dduZw1', 'dduZw2', 'dduZw3', 'dduZw4'};
Ratio=0.02;
dimension=length(uZc);
% plot(dduZc);hold on;
for index =1:length(List)
    var=List{index};
    eval(['Std=std(',var,');']);
    eval([var,'=',var,'+Ratio.*Std.*2.*(rand(dimension,1)-0.5);']);
end
% plot(dduZc);hold off;
%% 考量PDE损失数量级情况
Du1 = Mc .* g - 2 .* Cs .* duZc - 2 .* Ks .* uZc + Cs .* duZt1 + Ks .* uZt1 + Cs .* duZt2 + Ks .* uZt2;
Du2 = -2 .* Cs .* Lc .^ 2 .* duBc - 2 .* Ks .* Lc .^ 2 .* uBc - Cs .* Lc .* duZt1 + Cs .* Lc .* duZt2 - Ks .* Lc .* uZt1 + Ks .* Lc .* uZt2;
Du3 = Mt .* g - (2 .* Cp + Cs) .* duZt1 - (2 .* Kp + Ks) .* uZt1 + Cs .* duZc + Ks .* uZc + Cp .* duZw1 + Cp .* duZw2 + Kp .* uZw1 + Kp .* uZw2 - Cs .* Lc .* duBc - Ks .* Lc .* uBc;
Du4 = -2 .* Cp .* Lt .^ 2 .* duBt1 - 2 .* Kp .* Lt .^ 2 .* uBt1 - Cp .* Lt .* duZw1 + Cp .* Lt .* duZw2 - Kp .* Lt .* uZw1 + Kp .* Lt .* uZw2;
Du5 = Mt .* g - (2 .* Cp + Cs) .* duZt2 - (2 .* Kp + Ks) .* uZt2 + Cs .* duZc + Ks .* uZc + Cp .* duZw3 + Cp .* duZw4 + Kp .* uZw3 + Kp .* uZw4 + Cs .* Lc .* duBc + Ks .* Lc .* uBc;
Du6 = -2 .* Cp .* Lt .^ 2 .* duBt2 - 2 .* Kp .* Lt .^ 2 .* uBt2 - Cp .* Lt .* duZw3 + Cp .* Lt .* duZw4 - Kp .* Lt .* uZw3 + Kp .* Lt .* uZw4;
Du7 = -Kp .* uZw1 + Cp .* duZt1 + Kp .* uZt1 - Cp .* Lt .* duBt1 - Kp .* Lt .* uBt1 + Mw .* g;
Du8 = -Kp .* uZw2 + Cp .* duZt1 + Kp .* uZt1 + Cp .* Lt .* duBt1 + Kp .* Lt .* uBt1 + Mw .* g;
Du9 = -Kp .* uZw3 + Cp .* duZt2 + Kp .* uZt2 - Cp .* Lt .* duBt2 - Kp .* Lt .* uBt2 + Mw .* g;
Du10 = -Kp .* uZw4 + Cp .* duZt2 + Kp .* uZt2 + Cp .* Lt .* duBt2 + Kp .* Lt .* uBt2 + Mw .* g;
% 施加稳定力
SF_Pre = (2 .* Mt + Mc) .* g / 4;
SF_Sec = Mc .* g / 2;
Du1 = Du1 - 2 .* SF_Sec;
Du3 = Du3 + 1 .* SF_Sec - 2 .* SF_Pre;
Du5 = Du5+ 1 .* SF_Sec - 2 .* SF_Pre;
Du7 = Du7+1 .* SF_Pre;
Du8 = Du8+1 .* SF_Pre;
Du9 = Du9+1 .* SF_Pre;
Du10 = Du10+1 .* SF_Pre;
% 施加惯性力
Du1 = Du1-Mc .* dduZc;
Du2 = Du2-Jc .* dduBc;
Du3 = Du3-Mt .* dduZt1;
Du4 = Du4-Jt .* dduBt1;
Du5 = Du5-Mt .* dduZt2;
Du6 = Du6-Jt .* dduBt2;
Du7 = Du7-Mw .* dduZw1;
Du8 = Du8-Mw .* dduZw2;
Du9 = Du9-Mw .* dduZw3;
Du10 = Du10-Mw .* dduZw4;  
% 施加轮轨力
G = 4.57 * 0.43 ^ (-0.149) * 1e-8;
detZ1 = uZw1 - WIrre1 - E1;
detZ2 = uZw2 - WIrre2 - E2;
detZ3 = uZw3 - WIrre3 - E3;
detZ4 = uZw4 - WIrre4 - E4;
detZ1(find(detZ1<0))=0;
detZ2(find(detZ2<0))=0;
detZ3(find(detZ3<0))=0;
detZ4(find(detZ4<0))=0;

Du7 = Du7- ((1 / G) .* detZ1) .^ (3 / 2);
Du8 = Du8- ((1 / G) .* detZ2) .^ (3 / 2);
Du9 = Du9- ((1 / G) .* detZ3) .^ (3 / 2);
Du10 = Du10- ((1 / G) .* detZ4) .^ (3 / 2);

% for i=1:10
%     figure(i);
%     eval(['plot(Du',num2str(i),');']);
% end
for i=1:10
    eval(['Weights_Tem(',num2str(i),',1)=max(Du',num2str(i),'(5:end));']);
end
Weights(DataIndex,:)=Weights_Tem';
A=1;
end
SaveFileName='Weights_2000V2.mat';SaveFileName=[Path,'\',SaveFileName];
save(SaveFileName,'Weights','-v7.3');