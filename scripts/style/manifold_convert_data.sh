#!/bin/bash
outputFolder="simplifiedMore"
simplificationScript="data/simplification0.02.mlx"
for category in data/style/*; do
  if [ -d $category ]; then
    [ -d $category/manifold ] || mkdir $category/manifold
    [ -d $category/$outputFolder ] || mkdir $category/$outputFolder
    for filename in $category/train/*.obj; do
      echo $(basename "$filename") "$filename"
      [ -f "$category/manifold/$(basename "$filename")" ] || ./ManifoldPlus/build/manifold --input "$filename" --output "$category/manifold/$(basename "$filename")" --depth 8
      [ -f "$category/$outputFolder/$(basename "$filename")" ] || snap run meshlab.meshlabserver -i "$category/manifold/$(basename "$filename")" -o "$category/$outputFolder/$(basename "$filename")" -s "$simplificationScript"
    done
  fi
done