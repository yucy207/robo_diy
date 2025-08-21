huggingface-cli download --repo-type dataset  yucy207/robopanoptes_test12 --cache-dir ./d

python process_data_from_lerobot.py 
python train.py --config-name=train_diffusion_transformer_snake_workspace task.dataset_path=./dataset.zarr.zip 

dataset_id="_0818"
cache_dir="/data/yuchenyang/snake_traj"
huggingface-cli download --repo-type dataset  yucy207/robopanoptes_test${dataset_id} --cache-dir ${cache_dir}