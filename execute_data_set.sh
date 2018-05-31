#!/bin/bash
# this file should be executed from the folder where the wav files are located

echo "4th batch started at;$now"
python3 DataPreperation.py 4
echo "4th batch started at;$now"


echo "14th batch started at;$now"
python3 DataPreperation.py 14
echo "14th batch started at;$now"

echo "13th batch started at;$now"
python3 DataPreperation.py 13
echo "13th batch started at;$now"
