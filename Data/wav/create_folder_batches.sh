#!/bin/bash
do
for i in {1..14} ;
do
    mkdir "batch"$i
    echo "Created directory:  " "batch"$i

    mv `ls *.wav | head -50` "batch"$i
    echo "Moved files to:  " "batch"$i

done
