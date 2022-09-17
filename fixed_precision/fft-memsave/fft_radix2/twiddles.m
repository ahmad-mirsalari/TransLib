clear all;
close all;
clc;

% SETTINGS
Radix = 2;                 % Base Radix
N = 2^11;                   % Number of Input Samples
Steps = log(N)/log(Radix); % Number of Steps to process
NumBTF = N/Radix;          % Number of Butterflies per single step
Lw = 2;                    % LineWidth for Plotting
ShowIndexFigs = 0;         % Show Other figures or not

% GENERATE DATA
Data = zeros(1,N);
Data = Data + cos(2*pi*3*linspace(0,1,N));
Data = Data + cos(2*pi*6*linspace(0,1,N));
Data = Data + 0.1*randn(1,N);
Freq = linspace(-N/2,N/2,N); % fs = N;
DataOrig = Data;

% BITREVERSE INPUT
Data = bitrevorder(Data);

% GENERATE Twiddles (WNs)
WNs = exp(-1j*2*pi*(0:N/2-1)/N)

% Salvare il testo così sul file atom twiddle_factor.h (se ti chiede dove
% copiarlo magari dirgli Gedit
fileID = fopen('twiddle_factor.h' , 'w');
    fprintf(fileID, '#ifndef TWIDDLE_FACT_H\n#define TWIDDLE_FACT_H\n\n');
    fprintf(fileID, 'PULP_L1_DATA Complex_type twiddle_factors[] = {\n');
    for tw = 0:WnIndex-1
                fprintf(fileID, '{%.6ff, %.6ff}, \n', real(WNs(tw+1)), imag(WNs(tw+1)));
    end
    fprintf(fileID, '};\n#endif');
    fclose(fileID);
