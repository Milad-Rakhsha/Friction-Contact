# =============================================================================
# SIMULATION-BASED ENGINEERING LAB (SBEL) - http://sbel.wisc.edu
# University of Wisconsin-Madison
#
# Copyright (c) 2020 SBEL
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# at https://opensource.org/licenses/BSD-3-Clause
#
# =============================================================================
# Contributors: Nic Olsen
# =============================================================================
#!/usr/bin/env bash

if [ $# -ne 4 ]; then
    echo "usage: $0 <data_dir> <startframe> <endframe> <fps>"
    exit 1
fi

data_dir=$1
start=$2
stop=$3
fps=$4

cd $data_dir
if [ $? -ne 0 ]; then
    echo "invalid data_dir"
fi

for i in $(seq $start $stop); do
    printf -v frame "%06d" $i
    echo $frame
    ~/ospray-tools/cospRender loader=conlain_raw_friction colorizer=constant scene=granular resources=/home/nic/SimpleDVI/Assets width=1920 height=1080 camera=0,-20,5 camdir=0,20,-5 bgcolor=a1a1aa mesh_input_file=step${frame}.csv output=step${frame} input=
done

# Render video
ffmpeg -r $fps -f image2 -i step%06d.ppm -vcodec libx264 -crf 25 -pix_fmt yuv420p movie.mp4
