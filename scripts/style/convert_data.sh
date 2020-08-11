#!/bin/bash
# name of class subfolder where remeshed files will be saved
outputFolder="remeshed"
meshInfoScript='data/getMeshInfo.mlx'
# link to meshLab script
simplificationScript='data/remesh-simp.mlx'
# regex to extract number of faces from script output
regex='([0-9]+) fn'

function runMeshLabScript {
  # input location:   $1
  # output location:  $2
  # script to use:    $3
  # return:           array {input#faces, output#faces}
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
    for filename in $category/train/*.obj; do
      echo $(basename "$filename") "$filename"
      # check if file has already been processed
      if [ ! -f "$category/$outputFolder/$(basename "$filename")" ]; then
        # if mesh has too many faces to begin with, give up
        runMeshLabScript $filename $category/$outputFolder/$(basename "$filename") $meshInfoScript
        echo "mesh has ${output[0]} faces"

        if ((output[1] > 90000)); then
          echo "to be deleted"
          rm "$category/$outputFolder/$(basename "$filename")"
        else
          # parse output of script for input and processed mesh number of faces
          runMeshLabScript "$filename" "$category/$outputFolder/$(basename "$filename")" "$simplificationScript"
          echo "remeshed and simplified to ${output[1]} faces"
          # throw away mesh if it is above a certain number of faces. Too computationally expensive to process by model
          if ((output[1] > 20000)); then
            echo "to be deleted"
            rm "$category/$outputFolder/$(basename "$filename")"
          fi
        fi
      fi
    done
  fi
done

