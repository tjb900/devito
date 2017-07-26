#!/bin/bash

mkdir -p ~/endeavour-results
echo "Copying from Endeavour..."
scp -r Xflupor@207.108.8.122:/panfs/projects/External/opesci/results/edv ~/endeavour-results/
