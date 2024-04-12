set -uex
echo saving to $1

scenes=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

for scene in ${scenes[@]}; do
	python src/render_ngp_nerf_vdb.py --scene $scene --data_root /workspace/data/nerf_synthetic/ --db $1/ngp_nerf_vdb.db
done
echo "# VDB+HDDA on NeRF-Synthetic" >> $1/result.md
python src/show_table.py $1/ngp_nerf_vdb.db >> $1/result.md

scenes=("garden" "bicycle" "bonsai" "counter" "kitchen" "room" "stump")
scenes+=("flowers" "treehill")

for scene in ${scenes[@]}; do
	python src/render_ngp_nerf_vdb.py --scene $scene --data_root /workspace/data/mipnerf360_v2/ --db $1/mip_nerf_vdb.db
done
echo "# VDB+HDDA on Mip-NeRF 360" >> $1/result.md
python src/show_table.py $1/mip_nerf_vdb.db >> $1/result.md

