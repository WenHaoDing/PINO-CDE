function [g] =Generator_Irre(v)
% v=100;
Speed=v;
V=v/3.6;
t=500/V;%
LANM_Min=2.32;
LANM_Max=7;
f_min=V/LANM_Max;%
f_1=V/1;
f_max=V/LANM_Min;%
Tc=1e-4;
N=t/Tc;
N_r=2^nextpow2(N);

delta_f=1/(N_r*Tc);%
N_f=round(f_max/delta_f);%
N_1=round(f_1/delta_f);
N_0=round(f_min/delta_f);%a

%% 额外自定义参数
PowerSpectrum_Type='高铁谱';
Type='高低';
for ii=N_0:N_f
    f=ii*delta_f;
%     if   f<f_1
        SpaceFrequency=f/V;                                                %当前空间频率
        SpaceWaveLength=f*2*pi/V;
     if strcmp(PowerSpectrum_Type,'常规')==1
        if   f<f_1
        omega1=f*2*pi/V;
        S_k(ii-N_0+1)=0.25*0.2095*0.8245^2/(omega1^2*(omega1^2+0.8245^2))*2*pi/V/2;
        elseif  f>=f_1
        omega2=f/V;
        S_k(ii-N_0+1)=0.036*omega2^(-3.15)/V/1e2/2;
        end
    elseif strcmp(PowerSpectrum_Type,'美国谱')==1
        if   f<f_1
            S_k(ii-N_0+1)=PSD_America(SpaceFrequency,Type,6)*1e4/V;
        elseif  f>=f_1
            S_k(ii-N_0+1)=0.036*SpaceFrequency^(-3.15)/V/100/2;
        end
    elseif strcmp(PowerSpectrum_Type,'干线谱')==1
        if strcmp(Type,'高低')==1
        S_k(ii-N_0+1)=PSD_China_GanXian(SpaceFrequency,1)/100/V;
        elseif strcmp(Type,'轨向')==1
        S_k(ii-N_0+1)=PSD_China_GanXian(SpaceFrequency,2)/100/V;
        end
    elseif strcmp(PowerSpectrum_Type,'高铁谱')==1
        if   f<f_1
        S_k(ii-N_0+1)=PowerSpectrum_ChineseHighSpeed(SpaceFrequency,Type);
        S_k(ii-N_0+1)=S_k(ii-N_0+1)/V/100;
        elseif  f>=f_1
        S_k(ii-N_0+1)=0.036*SpaceFrequency^(-3.15)/v/100/2;
        end
    elseif strcmp(PowerSpectrum_Type,'德国低干扰谱')==1
        S_k(ii-N_0+1)=PowerSpectrum_GermanDiGanRao(SpaceWaveLength,Type);  %注意德国谱的输入是空间波长
        S_k(ii-N_0+1)=S_k(ii-N_0+1)/V/100;
    elseif strcmp(PowerSpectrum_Type,'Sato谱')==1
        if   f<f_1
             S_k(ii-N_0+1)=PSD_America(SpaceFrequency,Type,5)*1e4/V;
        elseif  f>=f_1
%              A=4.15*1e-8;
             A=5*1e-7;
             S_k(ii-N_0+1)=A/(SpaceWaveLength^3);
             S_k(ii-N_0+1)=S_k(ii-N_0+1)/V/100;
        end

%         elseif  f>=f_1
     end
%     elseif  f>=f_1
%         omega2=f/V;
%         S_k(ii-N_0+1)=0.036*omega2^(-3.15)/V/1e2/2;
%     end
end
phi=2*pi*rand(size(S_k));
epsilon=exp(1i*phi);
b=N_r*epsilon.*sqrt(S_k*delta_f);
c=fliplr(b);
d=conj(c);
e=[zeros(1,N_0-1),b,zeros(1,N_r/2-N_f+1),zeros(1,N_r/2-N_f),d,zeros(1,N_0-2)]';
g=ifft(e);
g=10.*g;


end

