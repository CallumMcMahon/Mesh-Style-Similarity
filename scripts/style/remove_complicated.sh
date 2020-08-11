#!/bin/bash
# name of class subfolder where remeshed files will be saved
outputFolder="remeshed"
meshInfoScript='data/getMeshInfo.mlx'
# regex to extract number of edges from script output
regex='E:\s+([0-9]+)'

function runMeshLabScript {
  # input location:   $1
  # output location:  $2
  # script to use:    $3
  # return:           number of edges
  mapfile -t output < <(snap run meshlab.meshlabserver -i "$1" -o "$2" -s "$3" 2>/dev/null |
          while IFS= read -r line; do
            if [[ $line =~ $regex ]]; then
              name="${BASH_REMATCH[1]}"
              echo $name
            fi
          done)
}

for category in data/style/*; do
  if [ -d $category ]; then
    # make sub-folder if it doesn't not already exist
    [ -d $category/$outputFolder ] || mkdir $category/$outputFolder
    for filename in $category/$outputFolder/*.obj; do
      echo $category "$(basename "$filename")"
      # check if file has already been processed
      # if mesh has too many faces to begin with, give up
      runMeshLabScript $filename $category/$outputFolder/"$(basename "$filename")" $meshInfoScript
      echo "mesh has ${output} edges"

      if ((output > 20000)); then
        echo "to be deleted"
        rm "$category/$outputFolder/$(basename "$filename")"
      fi
    done
  fi
done

