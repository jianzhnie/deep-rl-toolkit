python3 atari_dqn.py --env_id "SpaceInvadersNoFrameskip-v4" --test-num 100 --logger wandb
sleep 1h
python3 atari_dqn.py --env_id "BreakoutNoFrameskip-v4" --test-num 100  --logger wandb
sleep 1h
python3 atari_dqn.py --env_id "EnduroNoFrameskip-v4 "  --test-num 100 --logger wandb
sleep 1h
python3 atari_dqn.py --env_id "QbertNoFrameskip-v4" ---test-num 100 --logger wandb
sleep 1h
python3 atari_dqn.py --env_id "MsPacmanNoFrameskip-v4" --test-num 100 --logger wandb
sleep 1h
python3 atari_dqn.py --env_id "SeaquestNoFrameskip-v4" --test-num 100 --logger wandb
sleep 1h
python atari_dqn.py --env_id "PongNoFrameskip-v4" --batch-size 64 --logger wandb
