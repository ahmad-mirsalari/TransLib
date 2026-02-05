
N = 2048
BITS = log2(N)

bit_rev = (0:2047)

rev_index = bin2dec(fliplr(dec2bin(bit_rev,BITS)));

fileID = fopen('bit_reverse.h' , 'w'); % e mentre modifico la parte viola, quindi faccio run e mi appare l'aspetto del file bit_reverse.h 
fprintf(fileID, '#ifndef BIT_REVERSE_H\n#define BIT_REVERSE_H\n\n');
fprintf(fileID, '#ifdef FABRIC\n#define DATA_LOCATION\n#else\n#define DATA_LOCATION __attribute__((section(".data_l1")))\n#endif\n\n');
fprintf(fileID, 'DATA_LOCATION short bit_rev_LUT[] = {\n');
for i = 1:N
    fprintf(fileID, '%d, \n', rev_index(i));
end
fprintf(fileID, '};\n#endif');
fclose(fileID);
