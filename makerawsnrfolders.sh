#!/bin/bash

mkdir speech

basedir=$1
snr=$2
noisetype=$3

bdl=$basedir/noise_out_raw_bdl/$snr/$noisetype
jmk=$basedir/noise_out_raw_jmk/$snr/$noisetype
slt=$basedir/noise_out_raw_slt/$snr/$noisetype

echo "Processing: " $bdl
./createlinks.sh $bdl speech

echo "Processing: " $jmk
./createlinks.sh $jmk speech

echo "Processing: " $slt
./createlinks.sh $slt speech
