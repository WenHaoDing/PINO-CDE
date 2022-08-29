function dataout =  iomega(datain, dt, datain_type, dataout_type)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   IOMEGA is a MATLAB script for converting displacement, velocity, or
%   acceleration time-series to either displacement, velocity, or
%   acceleration times-series. The script takes an array of waveform data
%   (datain), transforms into the frequency-domain in order to more easily
%   convert into desired output form, and then converts back into the time
%   domain resulting in output (dataout) that is converted into the desired
%   form.
%
%   Variables:
%   ----------
%
%   datain       =   input waveform data of type datain_type
%
%   dataout      =   output waveform data of type dataout_type
%
%   dt           =   time increment (units of seconds per sample)
%
%                    1 - Displacement
%   datain_type  =   2 - Velocity
%                    3 - Acceleration
%
%                    1 - Displacement
%   dataout_type =   2 - Velocity
%                    3 - Acceleration
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Make sure that datain_type and dataout_type are either 1, 2 or 3
if (datain_type < 1 || datain_type > 3)
    error('Value for datain_type must be a 1, 2 or 3');
elseif (dataout_type < 1 || dataout_type > 3)
    error('Value for dataout_type must be a 1, 2 or 3');
end
%   Determine Number of points (next power of 2), frequency increment
%   and Nyquist frequency
N = 2^nextpow2(max(size(datain)));
df = 1/(N*dt);
Nyq = 1/(2*dt);
%   Save frequency array
iomega_array = 1i*2*pi*(-Nyq : df : Nyq-df);
iomega_exp = dataout_type - datain_type;
%   Pad datain array with zeros (if needed)
size1 = size(datain,1);
size2 = size(datain,2);
if (N-size1 ~= 0 && N-size2 ~= 0)
    if size1 > size2
        datain = vertcat(datain,zeros(N-size1,1));
    else
        datain = horzcat(datain,zeros(1,N-size2));
    end
end
%   Transform datain into frequency domain via FFT and shift output (A)
%   so that zero-frequency amplitude is in the middle of the array
%   (instead of the beginning)
A = fft(datain);
A = fftshift(A);
%   Convert datain of type datain_type to type dataout_type
for j = 1 : N
    if iomega_array(j) ~= 0
        A(j) = A(j) * (iomega_array(j) ^ iomega_exp);
    else
        A(j) = complex(0.0,0.0);
    end
end
%   Shift new frequency-amplitude array back to MATLAB format and
%   transform back into the time domain via the inverse FFT.
A = ifftshift(A);
datain = ifft(A);
%   Remove zeros that were added to datain in order to pad to next
%   biggerst power of 2 and return dataout.
if size1 > size2
    dataout = real(datain(1:size1,size2));
else
    dataout = real(datain(size1,1:size2));
end
return