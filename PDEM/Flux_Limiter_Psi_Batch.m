function [Psi]=Flux_Limiter_Psi_Batch(rp,rn,gk)
% Roe-Sweby Flux Lmiter with relatively small dissipation
modeNumber=length(rp);
Psi_sb_rp=max([zeros(modeNumber,1) min(2*rp,1)' min(rp,2)'],[],2);
Psi_sb_rn=max([zeros(modeNumber,1) min(2*rn,1)' min(rn,2)'],[],2);
up_gk=zeros(modeNumber,1);
un_gk=zeros(modeNumber,1);
for p=1:modeNumber
if gk(p)>=0
    up_gk(p)=1;
else
    up_gk(p)=0;
end
if (-gk(p))>=0
    un_gk(p)=1;
else
    un_gk(p)=0;
end 
end
Psi=un_gk.*Psi_sb_rp+up_gk.*Psi_sb_rn;
end

