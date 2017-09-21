#!/bin/bash

if [ -z ${DEVITO_HOME+x} ]; then
	echo "Please, set DEVITO_HOME to the root Devito directory"
	exit
fi

cd $DEVITO_HOME

git branch > version.txt
git log -n 1 --pretty=format:"%H" >> version.txt
sed -i -e '$a\' version.txt

scp -r devito/ examples/ version.txt Xflupor@207.108.8.122:/panfs/projects/External/opesci/devito/

cd ..
