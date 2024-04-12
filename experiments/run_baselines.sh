set -uex

scenes=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

for scene in ${scenes[@]}; do
	python src/train_ngp_nerf_occ.py --scene $scene --data_root /workspace/data/nerf_synthetic/
done
echo "# NerfAcc on NeRF-Synthetic" >> baseline.md
python src/show_table.py ngp_nerf_occ.db >> baseline.md

scenes=("garden" "bicycle" "bonsai" "counter" "kitchen" "room" "stump")
scenes+=("flowers" "treehill")

for scene in ${scenes[@]}; do
	python src/train_ngp_nerf_occ.py --scene $scene --data_root /workspace/data/mipnerf360_v2/
done
echo "# NerfAcc on Mip-NeRF 360" >> baseline.md
python src/show_table.py mip_nerf_occ.db >> baseline.md
