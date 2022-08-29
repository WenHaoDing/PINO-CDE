% 积分操作由加速度求位移，可选时域积分和频域积分
function [disint, velint] = IntFcn(acc, t, ts, flag)
if flag == 1
    % 时域积分
    [disint, velint] = IntFcn_Time(t, acc);
    velenergy = sqrt(sum(velint.^2));
    velint = detrend(velint);
    velreenergy = sqrt(sum(velint.^2));
    velint = velint/velreenergy*velenergy;  
    disenergy = sqrt(sum(disint.^2));
    disint = detrend(disint);
    disreenergy = sqrt(sum(disint.^2));
    disint = disint/disreenergy*disenergy; % 此操作是为了弥补去趋势时能量的损失

    % 去除位移中的二次项
    p = polyfit(t, disint, 2);
    disint = disint - polyval(p, t);
else
    % 频域积分
    velint =  iomega(acc, ts, 3, 2);
    velint = detrend(velint);
    disint =  iomega(acc, ts, 3, 1);
    % 去除位移中的二次项
    p = polyfit(t, disint, 2);
    disint = disint - polyval(p, t);
end
end