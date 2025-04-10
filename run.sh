  export PYTHONPATH=.

torchrun  --nproc_per_node=4 dinov2/train/train.py --config-file dinov2/configs/train/vitb16_ego4d.yaml
torchrun  --nproc_per_node=8 dinov2/train/train.py --config-file dinov2/configs/train/vitb16_ego4d.yaml --no-resume --output_dir /pfss/mlde/workspaces/mlde_wsp_PI_Roig/schaumloeffel/logs/ego4d/dinov2