#!/bin/bash

# Converts all files from JPEG to PNG in folder dpi$1,
# where $1 is the dpi.
# $1 is entered as a positional parameter.


for jpg in $1/*.jpg
do
	png=${jpg%.jpg}.png
	magick $jpg $png
	rm $jpg
done


# Reverse
: '
for png in $1/*.png
do
	jpg=${png%.png}.jpg
	magick $png $jpg
	rm $png
done
'