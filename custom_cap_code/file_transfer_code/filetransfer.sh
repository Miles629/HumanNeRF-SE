sub="mcy2"
rootpath="/root/workspace-NerfHuman/InstantAvatar/data/custom/$sub"
aimpath="/root/workspace-NerfHuman/CVPRversionAnimateHuman/dataset/custom/$sub"
mkdir $aimpath
ln -s "$rootpath/images"  $aimpath
ln -s "$rootpath/masks"  $aimpath
python filetrans.py $rootpath --outputpath $aimpath