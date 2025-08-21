huggingface-cli download --repo-type dataset  yucy207/robopanoptes_test12 --cache-dir ./d

python process_data_from_lerobot.py 
python train.py --config-name=train_diffusion_transformer_snake_workspace task.dataset_path=./dataset.zarr.zip 
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes 2 train.py --config-name=train_diffusion_transformer_snake_workspace task.dataset_path=sweep.zarr.zip


HUGGINGFACE_TOKEN='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'
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
HF_USER=$(huggingface-cli whoami | head -n 1)
echo "$HF_USER"

dataset_id="_0818"
local_dir="/data/yuchenyang/snake_traj"
huggingface-cli download --repo-type dataset  yucy207/robopanoptes_test${dataset_id} --local-dir ${local_dir}