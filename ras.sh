#!/bin/bash
j=1000
for i in {0..9};
    do
        a=$(./api-usage $i $j)
        echo $a
    done