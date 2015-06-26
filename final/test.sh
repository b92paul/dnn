#!/bin/bash
IFS=$'\n'
for l in `cat $1`;
do
	echo $l | ./run.sh -
done

