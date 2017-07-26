#!/bin/bash

if [ -z ${DEVITO_OUTPUT+x} ]; then
	echo "Please, set DEVITO_OUTPUT to the root results directory"
	exit
fi

dest=$DEVITO_OUTPUT/endeavour

mkdir -p $dest

echo "Copying from richter to $dest"
scp -r fl1612@ese-richter.ese.ic.ac.uk:/home/fl1612/endeavour-results/endeavour/* $dest
