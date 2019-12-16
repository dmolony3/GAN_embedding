#!/bin/sh

fake_directory=/home/microway/Documents/SPADE/results/IVUS_45MHz/inference_latest/images/synthesized_image_45MHz/
real_directory=/home/microway/Documents/IVUS/Data/

output_directory=/home/microway/Documents/IVUS/Segmentation2.0/Embedding_GAN/Images/

count=0
total_images=2048
for file in $fake_directory*
do
  if (( $count>=$total_images )); then
    break
  fi
  image=${file#"$fake_directory"}
  image=${image%.png}
  
  patient=${image%_*}
  frame=${image##*_}
  real_path=$real_directory$patient"/"$frame".jpg"
  cp $real_path $output_directory"Real_"${patient}_$frame".jpg"
  cp $file $output_directory"Fake_"${patient}_$frame".jpg"

  ((count++))
  echo $count
done
