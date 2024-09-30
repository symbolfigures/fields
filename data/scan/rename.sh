#!/bin/bash

# Renames all files in folder dpi$1,
# where $1 is the dpi.
# $1 is entered as positional parameters.

# "$02d" pads the filename so it has 2 digits.
# If there are more than 99 files, update accordingly.

count=0

for file in $(ls $1 | sort -V)
do
	n=$(printf "%02d" $count)
	mv $1/$file $1/$n.png
	((count++))
done


