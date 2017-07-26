#!/bin/bash

dest=/data/opesci/experimentation/raw/endeavour

mkdir -p $dest

echo "Copying from richter to $dest"
scp -r fl1612@ese-richter.ese.ic.ac.uk:/home/fl1612/endeavour-results/results/* $dest
