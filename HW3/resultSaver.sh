#!/bin/bash

OUT_FILE="/tmp/result.txt"

#Checks file existence.
if [ ! -f $OUT_FILE ]; then
	touch $OUT_FILE
	echo "N       Iteration        Time">$OUT_FILE
fi
nvcc $1 -o $2
for i in {2..80}
do
	./$2 $i >>$OUT_FILE
done