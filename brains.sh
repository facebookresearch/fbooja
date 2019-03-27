#!/usr/bin/env bash

#
# This script generates figures for the paper in subdirectory "tex".
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the LICENSE file
# in the root directory of this source tree.
#


rm -rf brainsout
mkdir brainsout


N_RESAMPS=1000


# Process the twenty brains.
SUBSAMPLING_FACTOR=.25
SUBSAMPLING_FACTOR2=.1

for ind in {1..20}
do
    echo
    echo 'bootstrap' ${ind}
    python bootstrap.py --filein brains/brain${ind}.png \
        --fileout brainsout/h${ind}.png \
        --subsampling_factor $SUBSAMPLING_FACTOR --n_resamps $N_RESAMPS
done

for ind in {1..20}
do
    echo
    echo 'bootstrap2' ${ind}
    python bootstrap2.py --filein brains/brain${ind}.png \
        --fileout brainsout/r${ind}.png \
        --subsampling_factor $SUBSAMPLING_FACTOR2 --n_resamps $N_RESAMPS
done

for ind in {1..20}
do
    echo
    echo 'jackknife' ${ind}
    python jackknife.py --filein brains/brain${ind}.png \
        --fileout brainsout/h${ind}j.png \
        --subsampling_factor $SUBSAMPLING_FACTOR
done

for ind in {1..20}
do
    echo
    echo 'jackknife2' ${ind}
    python jackknife2.py --filein brains/brain${ind}.png \
        --fileout brainsout/r${ind}j.png \
        --subsampling_factor $SUBSAMPLING_FACTOR2
done


# Process the twenty brains with twice the sampling factors.
SUBSAMPLING_FACTOR=.5
SUBSAMPLING_FACTOR2=.2

for ind in {1..20}
do
    echo
    echo 'bootstrap' ${ind} '2x'
    python bootstrap.py --filein brains/brain${ind}.png \
        --fileout brainsout/h${ind}x2.png \
        --subsampling_factor $SUBSAMPLING_FACTOR --n_resamps $N_RESAMPS
done

for ind in {1..20}
do
    echo
    echo 'bootstrap2' ${ind} '2x'
    python bootstrap2.py --filein brains/brain${ind}.png \
        --fileout brainsout/r${ind}x2.png \
        --subsampling_factor $SUBSAMPLING_FACTOR2 --n_resamps $N_RESAMPS
done

for ind in {1..20}
do
    echo
    echo 'jackknife' ${ind} '2x'
    python jackknife.py --filein brains/brain${ind}.png \
        --fileout brainsout/h${ind}jx2.png \
        --subsampling_factor $SUBSAMPLING_FACTOR
done

for ind in {1..20}
do
    echo
    echo 'jackknife2' ${ind} '2x'
    python jackknife2.py --filein brains/brain${ind}.png \
        --fileout brainsout/r${ind}jx2.png \
        --subsampling_factor $SUBSAMPLING_FACTOR2
done


# Process two of the twenty brains for visualization.
SUBSAMPLING_FACTOR=.25
SUBSAMPLING_FACTOR2=.1
inds='3 10'

for ind in ${inds}
do
    echo
    echo 'bootstrap' ${ind}
    python bootstrap.py --filein brains/brain${ind}.png \
        --fileout brainsout/h${ind}.png \
        --subsampling_factor $SUBSAMPLING_FACTOR --n_resamps $N_RESAMPS --viz
done

for ind in ${inds}
do
    echo
    echo 'bootstrap2' ${ind}
    python bootstrap2.py --filein brains/brain${ind}.png \
        --fileout brainsout/r${ind}.png \
        --subsampling_factor $SUBSAMPLING_FACTOR2 --n_resamps $N_RESAMPS --viz
done
