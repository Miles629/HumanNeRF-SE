sub="iftime"
rootpath="/root/workspace-NerfHuman/InstantAvatar/data/custom/$sub"
aimpath="/root/workspace-NerfHuman/CVPRversionAnimateHuman/dataset/drive/$sub"
aimshape="/root/workspace-NerfHuman/CVPRversionAnimateHuman/dataset/zju_mocap/386_41_train/mesh_infos.pkl"
mkdir $aimpath
ln -s "$rootpath/images"  $aimpath
ln -s "$rootpath/masks"  $aimpath
python filetrans_aimshape.py $rootpath --outputpath $aimpath --shapepath $aimshape