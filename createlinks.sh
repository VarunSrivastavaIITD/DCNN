srcdir=$1
dstdir=$2

for file in $srcdir/*
do
ln -s $(realpath $file) $dstdir/$(basename $file)
done
