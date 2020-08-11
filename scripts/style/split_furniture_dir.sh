#!/bin/bash
styleFolder="data/style"
class_to_fix="Drinkware"
for filename in $styleFolder/$class_to_fix/train/*.obj; do
  class=$(echo $filename | cut -d'_' -f 2)
  [ -d $styleFolder/$class ] || { mkdir $styleFolder/$class; mkdir $styleFolder/$class/train; }
  mv $filename $styleFolder/$class/train/$(basename "$(echo $filename | cut -d'_' -f 1)")_$(echo $filename | cut -d'_' -f 3)
done