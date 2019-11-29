d="$(date +%F)"
dir_name="plot_${d}"
mkdir $dir_name
for filename in `find -iname '*.obj'`
do
     cp $filename $dir_name
done