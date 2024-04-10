#!/bin/bash

# define a download function
function google_drive_download()
{
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

# mesh
mkdir ./meshes
google_drive_download 1u8GJ7cT7_5hQlplj-5_pYh16m86GdF99 hammer.obj
mv hammer.obj ./meshes

# prepare directories
mkdir ./demo
mkdir ./demo/hammer

# per-vertex encoder features
mkdir ./demo/hammer/encoder
google_drive_download 13bhW6FDzLs4UAQAyaR6N6w1efK41Z8M3 pred_f.pth
mv pred_f.pth ./demo/hammer/encoder

# decoder checkpoint
mkdir ./demo/hammer/decoder
google_drive_download 1WWu0NO1pZpS39_tNCAhFD77RSotyq5E4 decoder_checkpoint.pth
mv decoder_checkpoint.pth ./demo/hammer/decoder
