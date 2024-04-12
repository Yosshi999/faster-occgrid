docker run -it --rm --name nerfacc --gpus all \
	-v $PWD/experiments:/workspace/experiments \
	-v $PWD/data:/workspace/data \
	faster-occgrid bash
