# for run in {1..30}; do echo -e ${run}; ./benchmark_mpi.sh; done
for scale in $(seq 1 10)
do 	
	ffmpeg -n -loglevel quiet -i test.png -vf scale=iw*${scale}:ih*${scale} test${scale}x.png
	declare -ia X=(299 1096 1197 84)
	declare -ia Y=(134 57 768 592)
	declare -ia 'X1=("${X[@]/%/*scale}")'
	declare -ia 'Y1=("${Y[@]/%/*scale}")'
	echo -e "\tDoing ${scale}..."
	mpiexec ../x64/Release/ImageHomography test${scale}x.png 0.5625 "${X1[@]}" "${Y1[@]}" >> test${scale}x_mpi.txt
	echo -e "\tDone."
done
