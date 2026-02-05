clear all;
close all;
clc;

% FFT DIT RADIX-4
DIF = 1;                % Decimation in Time or Frequnency  //devo metterlo a 2 o commentarlo? ma tanto non lo usa
N = 2^10;                % Number of input samples
Radix = 4;              % The radix
Steps = log(N)/log(4);  % Number of Steps
NumBFly = N/4;          % Number of Butterflies per step
WNs = [];               % Twiddle Factors
Lw = 2;                 % LineWidth for Plotting


for Stage = 1:Steps

    Dist = 4^(Steps - Stage);  //è il num. dei twiddles che va da 4^N-1 a 4^0

    % SAVE DATA INDEXES
    Dta = zeros(1,N);

    % CALCULATE BUTTERFLIES
    for i = 0:NumBFly-1

        % CALCULATE TWIDDLES
        if(Stage == Steps)
            WNx = ones(1,4);  //genera una matrice MxN formata da elementi uguali a 1
        else
            Vals = ([0 1 2 3])./N .* (4^(Stage - 1)) .* mod(i,Dist);
            WNx = exp(-1j*2*pi*Vals);
            for tw = 0:Radix-1
                fprintf(fileID, '{%.6ff, %.6ff}, \n', real(WNx(tw+1)), imag(WNx(tw+1)));
            end
        end
    end

end



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
