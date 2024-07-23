#!/bin/bash
export DIR="$(dirname "$(pwd)")"
export PYTHONPATH=${PYTHONPATH}:${DIR}

# Add smiles, latent vectors and targets path here
smiles_path='../data/VEGFR2_6_smiles.txt'
z_path='../data/VEGFR2_6_latent_features.txt'
y_path='../data/VEGFR2_6_dual.txt'

# Path to JTVAE checkpoint, vocab and configurations
path_to_vae_statedict='../JTVAE-GA/fast_molvae/moses_h450z56_prop/model.iter-730000'
vocab_path='../JTVAE-GA/moses/vocab.txt'
hidden_size=450
latent_size=56
depthT=20
depthG=3
beta=0.1

# task_id is type of target function
task_id="pic50"
seed=2023

# True if track with wandb, modify wandb configurations accordingly
track_with_wandb=True
read -p "Please type your wandb api key (empty for logged in user): " wandb_key
wandb_entity='buchijw'
wandb_project_name='LOLBO-JTVAE'
wandb_run_name=

# Target function direction
minimize=False

max_n_oracle_calls=5000
num_initialization_points=1360
learning_rte=0.001
acq_func='ts'
# ['ts','ei']
bsz=60

init_n_update_epochs=20
num_update_epochs=3
e2e_freq=3
update_e2e=True

k=50

verbose=True

WANDB_API_KEY=$wandb_key python molecule_optimization_jtvae.py \
            --smiles_path $smiles_path \
            --z_path $z_path \
            --y_path $y_path \
            --path_to_vae_statedict $path_to_vae_statedict \
            --vocab_path $vocab_path \
            --hidden_size $hidden_size \
            --latent_size $latent_size \
            --depthT $depthT \
            --depthG $depthG \
            --beta $beta \
            --task_id $task_id \
            --seed $seed \
            --track_with_wandb $track_with_wandb \
            --wandb_entity $wandb_entity \
            --wandb_project_name $wandb_project_name \
            --wandb_run_name $wandb_run_name \
            --minimize $minimize \
            --max_n_oracle_calls $max_n_oracle_calls \
            --learning_rte $learning_rte \
            --acq_func $acq_func \
            --bsz $bsz \
            --num_initialization_points $num_initialization_points \
            --init_n_update_epochs $init_n_update_epochs \
            --num_update_epochs $num_update_epochs \
            --e2e_freq $e2e_freq \
            --update_e2e $update_e2e \
            --k $k \
            --verbose $verbose - run_lolbo - done
