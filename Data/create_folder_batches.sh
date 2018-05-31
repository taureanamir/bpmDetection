#!/bin/bash
# this file should be executed from the folder where the wav files are located
for i in {1..14} ;
do
    mkdir "batch"$i
    echo "Created directory:  " "batch"$i

    mv `ls *.wav | head -50` "batch"$i
    echo "Moved files to:  " "batch"$i

done
