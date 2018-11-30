#!/usr/bin/env bash

#
# This script generates all figures displayed in http://tygert.com/comps.pdf
#

#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the LICENSE file
# in the root directory of this source tree.
#


rm -rf brainsout
mkdir brainsout


# Process the main examples.
SUBSAMPLING_FACTOR=.25
SUBSAMPLING_FACTOR2=.1
N_RESAMPS=1000

python jackknife.py --filein brains/brain0.png \
    --fileout brainsout/jdefault.png \
    --subsampling_factor $SUBSAMPLING_FACTOR
python jackknife2.py --filein brains/brain0.png \
    --fileout brainsout/j2default.png \
    --subsampling_factor $SUBSAMPLING_FACTOR2
python bootstrap.py --filein brains/brain0.png \
    --fileout brainsout/bdefault.png \
    --subsampling_factor $SUBSAMPLING_FACTOR --n_resamps $N_RESAMPS
python bootstrap2.py --filein brains/brain0.png \
    --fileout brainsout/b2default.png \
    --subsampling_factor $SUBSAMPLING_FACTOR2 --n_resamps $N_RESAMPS


# Process the twenty brains.
SUBSAMPLING_FACTOR=.25
SUBSAMPLING_FACTOR2=.1
N_RESAMPS=1000

for ind in {1..20}
do
    echo
    echo 'jackknife' ${ind}
    python jackknife.py --filein brains/brain${ind}.png \
        --fileout brainsout/j_${ind}.png \
        --subsampling_factor $SUBSAMPLING_FACTOR
done

for ind in {1..20}
do
    echo
    echo 'jackknife2' ${ind}
    python jackknife2.py --filein brains/brain${ind}.png \
        --fileout brainsout/j2_${ind}.png \
        --subsampling_factor $SUBSAMPLING_FACTOR2
done

for ind in {1..20}
do
    echo
    echo 'bootstrap' ${ind}
    python bootstrap.py --filein brains/brain${ind}.png \
        --fileout brainsout/b_${ind}.png \
        --subsampling_factor $SUBSAMPLING_FACTOR --n_resamps $N_RESAMPS
done

for ind in {1..20}
do
    echo
    echo 'bootstrap2' ${ind}
    python bootstrap2.py --filein brains/brain${ind}.png \
        --fileout brainsout/b2_${ind}.png \
        --subsampling_factor $SUBSAMPLING_FACTOR2 --n_resamps $N_RESAMPS
done


# Process the twenty brains with twice the sampling factors.
SUBSAMPLING_FACTOR=.5
SUBSAMPLING_FACTOR2=.2
N_RESAMPS=1000

for ind in {1..20}
do
    echo
    echo 'jackknife' ${ind}
    python jackknife.py --filein brains/brain${ind}.png \
        --fileout brainsout/j_${ind}_2.png \
        --subsampling_factor $SUBSAMPLING_FACTOR
done

for ind in {1..20}
do
    echo
    echo 'jackknife2' ${ind}
    python jackknife2.py --filein brains/brain${ind}.png \
        --fileout brainsout/j2_${ind}_2.png \
        --subsampling_factor $SUBSAMPLING_FACTOR2
done

for ind in {1..20}
do
    echo
    echo 'bootstrap' ${ind}
    python bootstrap.py --filein brains/brain${ind}.png \
        --fileout brainsout/b_${ind}_2.png \
        --subsampling_factor $SUBSAMPLING_FACTOR --n_resamps $N_RESAMPS
done

for ind in {1..20}
do
    echo
    echo 'bootstrap2' ${ind}
    python bootstrap2.py --filein brains/brain${ind}.png \
        --fileout brainsout/b2_${ind}_2.png \
        --subsampling_factor $SUBSAMPLING_FACTOR2 --n_resamps $N_RESAMPS
done


# Crop the main examples.
convert brainsout/jdefault.png -crop x50\% brainsout/jdefault_\%d.png
convert brainsout/j2default.png -crop x50\% brainsout/j2default_\%d.png
convert brainsout/bdefault.png -crop x50\% brainsout/bdefault_\%d.png
convert brainsout/b2default.png -crop x50\% brainsout/b2default_\%d.png

for j in {0..1}
do
    convert brainsout/jdefault_${j}.png -trim brainsout/jdefault_${j}.png
    convert brainsout/j2default_${j}.png -trim brainsout/j2default_${j}.png
    convert brainsout/bdefault_${j}.png -trim brainsout/bdefault_${j}.png
    convert brainsout/b2default_${j}.png -trim brainsout/b2default_${j}.png
done


# Crop the twenty brains.
for k in {1..20}
do
    convert brainsout/j_${k}.png -crop x50\% brainsout/j_${k}_\%d.png
    convert brainsout/j2_${k}.png -crop x50\% brainsout/j2_${k}_\%d.png
    convert brainsout/b_${k}.png -crop x50\% brainsout/b_${k}_\%d.png
    convert brainsout/b2_${k}.png -crop x50\% brainsout/b2_${k}_\%d.png
done

for k in {1..20}
do
    for j in {0..1}
    do
        convert brainsout/j_${k}_${j}.png -trim brainsout/j_${k}_${j}.png
        convert brainsout/j2_${k}_${j}.png -trim brainsout/j2_${k}_${j}.png
        convert brainsout/b_${k}_${j}.png -trim brainsout/b_${k}_${j}.png
        convert brainsout/b2_${k}_${j}.png -trim brainsout/b2_${k}_${j}.png
    done
done


# Crop the twenty brains with twice the sampling factors.
for k in {1..20}
do
    convert brainsout/j_${k}_2.png -crop x50\% brainsout/j_${k}_2_\%d.png
    convert brainsout/j2_${k}_2.png -crop x50\% brainsout/j2_${k}_2_\%d.png
    convert brainsout/b_${k}_2.png -crop x50\% brainsout/b_${k}_2_\%d.png
    convert brainsout/b2_${k}_2.png -crop x50\% brainsout/b2_${k}_2_\%d.png
done

for k in {1..20}
do
    for j in {0..1}
    do
        convert brainsout/j_${k}_2_${j}.png -trim brainsout/j_${k}_2_${j}.png
        convert brainsout/j2_${k}_2_${j}.png -trim brainsout/j2_${k}_2_${j}.png
        convert brainsout/b_${k}_2_${j}.png -trim brainsout/b_${k}_2_${j}.png
        convert brainsout/b2_${k}_2_${j}.png -trim brainsout/b2_${k}_2_${j}.png
    done
done
