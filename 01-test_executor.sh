#!/bin/bash

input_mp3=$1
echo "-----------------------------------------"
output_ext="wav"
if [ ${input_mp3: -4} == ".wav" ]; then
  output_file="${input_mp3/.wav/"2."$output_ext}"
  echo "--------------------------------------------------------"
  echo "CROP wav file ("$input_mp3"), output as $output_file at: $(date +%d-%m-%Y" "%H:%M:%S) "
  echo "--------------------------------------------------------"

else
  output_file="${input_mp3/mp3/$output_ext}"
  echo "--------------------------------------------------------"
  echo "CROP mp3 file ("$input_mp3") and convert to WAV ("$output_file") at: $(date +%d-%m-%Y" "%H:%M:%S) "
  echo "--------------------------------------------------------"

fi

ffmpeg -ss 0 -t 120 -i $input_mp3 $output_file

rm data/wav/test/*.*
rm data/figures/test/*.*

mv $output_file data/wav/test/

echo "--------------------------------------------------------"
echo "WAV to spectogram conversion started at: $(date +%d-%m-%Y" "%H:%M:%S)"
echo "--------------------------------------------------------"

python3 04-testDataPrep.py

echo "--------------------------------------------------------"
echo "Process test started at: $(date +%d-%m-%Y" "%H:%M:%S)"
echo "--------------------------------------------------------"
python3 05-processTestData.py

echo "-------------- Processing Completed at $(date +%d-%m-%Y" "%H:%M:%S)---------------"
