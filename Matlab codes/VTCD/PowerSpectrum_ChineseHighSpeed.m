function [S]=PowerSpectrum_ChineseHighSpeed(SpaceFrequence,Type)
%% 中国高速铁路轨道谱参数
GaoDiSwitch=[0.0187 0.0474 0.1533];
GaoDiParameter_A=[1.0544e-5 3.5588e-3 1.9784e-2 3.9488e-4];
GaoDiParameter_n=[3.3891 1.9271 1.3643 3.4516];
GuiXiangSwitch=[0.045 0.1234];
GuiXiangParameter_A=[3.9513e-3 1.1047e-2 7.5633e-4];
GuiXiangParameter_n=[1.867 1.5354 2.8171];
ShuiPingSwitch=[0.0258 0.1163];
ShuiPingParameter_A=[3.6148e-3 4.3685e-2 4.5867e-3];
ShuiPingParameter_n=[1.7278 1.0461 2.0939];
GuiJuSwitch=[0.1090 0.2938];
GuiJuParameter_A=[5.4978e-2 5.0701e-3 1.8778e-4];
GuiJuParameter_n=[0.8282 1.9037 4.5948];
%% 中国高速铁路轨道功率谱值计算
if strcmp(Type,'高低')==1
    if SpaceFrequence<GaoDiSwitch(1)
        ComParameter_A=GaoDiParameter_A(1);
        ComParameter_n=GaoDiParameter_n(1);
    elseif SpaceFrequence>=GaoDiSwitch(1)&&SpaceFrequence<GaoDiSwitch(2)
        ComParameter_A=GaoDiParameter_A(2);
        ComParameter_n=GaoDiParameter_n(2);
    elseif SpaceFrequence>=GaoDiSwitch(2)&&SpaceFrequence<GaoDiSwitch(3)
        ComParameter_A=GaoDiParameter_A(3);
        ComParameter_n=GaoDiParameter_n(3);
    elseif SpaceFrequence>=GaoDiSwitch(3)
        ComParameter_A=GaoDiParameter_A(4);
        ComParameter_n=GaoDiParameter_n(4);
    end
elseif strcmp(Type,'轨向')==1
    if SpaceFrequence<GuiXiangSwitch(1)
        ComParameter_A=GuiXiangParameter_A(1);
        ComParameter_n=GuiXiangParameter_n(1);
    elseif SpaceFrequence>=GuiXiangSwitch(1)&&SpaceFrequence<GuiXiangSwitch(2)
        ComParameter_A=GuiXiangParameter_A(2);
        ComParameter_n=GuiXiangParameter_n(2);
    elseif SpaceFrequence>=GuiXiangSwitch(2)
        ComParameter_A=GuiXiangParameter_A(3);
        ComParameter_n=GuiXiangParameter_n(3);
    end
elseif strcmp(Type,'水平')==1
    if SpaceFrequence<ShuiPingSwitch(1)
        ComParameter_A=ShuiPingParameter_A(1);
        ComParameter_n=ShuiPingParameter_n(1);
    elseif SpaceFrequence>=ShuiPingSwitch(1)&&SpaceFrequence<ShuiPingSwitch(2)
        ComParameter_A=ShuiPingParameter_A(2);
        ComParameter_n=ShuiPingParameter_n(2);
    elseif SpaceFrequence>=ShuiPingSwitch(2)
        ComParameter_A=ShuiPingParameter_A(3);
        ComParameter_n=ShuiPingParameter_n(3);
    end
elseif strcmp(Type,'轨距')==1
    if SpaceFrequence<GuiJuSwitch(1)
        ComParameter_A=GuiJuParameter_A(1);
        ComParameter_n=GuiJuParameter_n(1);
    elseif SpaceFrequence>=GuiJuSwitch(1)&&SpaceFrequence<GuiJuSwitch(2)
        ComParameter_A=GuiJuParameter_A(2);
        ComParameter_n=GuiJuParameter_n(2);
    elseif SpaceFrequence>=GuiJuSwitch(2)
        ComParameter_A=GuiJuParameter_A(3);
        ComParameter_n=GuiJuParameter_n(3);
    end 
end
    S=ComParameter_A/(SpaceFrequence^(ComParameter_n));                    %计算功率谱密度/mm2/(1/m)
end
