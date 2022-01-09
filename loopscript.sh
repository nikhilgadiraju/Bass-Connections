#!/bin/sh
# Change below 'cd' command to include same path as 'fpath_norm' in the normalization python file
cd /Users/nikhilgadiraju/Desktop/2021-2022/Extracurriculurs/'Bass Connections'/'Manganese Image Processing'/'Water Tube-Normalized Brain Volumes'

for i in *.*; do
fname=${i%%.*}
newname_bfc=${fname}_corrected.nii.gz
newname_field=${fname}_biasfield.nii.gz

# Change first entry in list following '-o' to be 'fpath_bfc_img/$newname_bfc' where fpath_bfc_img is indicated in the nomalization python file
# Change second entry in list following '-o' to be 'fpath_bfield/$newname_bfc' where fpath_bfield is indicated in the nomalization python file
N4BiasFieldCorrection -i $i -o [/Users/nikhilgadiraju/Desktop/2021-2022/Extracurriculurs/'Bass Connections'/'Manganese Image Processing'/'Bias Field Corrected (BFC) Brain Volumes'/$newname_bfc,/Users/nikhilgadiraju/Desktop/2021-2022/Extracurriculurs/'Bass Connections'/'Manganese Image Processing'/'BFC Bias Fields'/$newname_field]

echo $newname_bfc
done

