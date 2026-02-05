#!/bin/bash
echo -e  "fadd.s  \c"
grep -o "fadd.s" trace.txt | wc -l
echo  -e "fmul.s  \c"
grep -o "fmul.s" trace.txt | wc -l
echo  -e "fmadd.s \c"
grep -o "fmadd.s" trace.txt | wc -l
echo  -e "fsub.s  \c"
grep -o "fsub.s" trace.txt | wc -l
echo  -e "fmsub.s \c"
grep -o "fmsub.s" trace.txt | wc -l
echo  -e "flw     \c"
grep -o "flw" trace.txt | wc -l
echo  -e "fsw     \c"
grep -o "fsw" trace.txt | wc -l
