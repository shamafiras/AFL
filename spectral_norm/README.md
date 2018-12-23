# AFL
train baseline:
python main.py --train --loop_count 1 --checkpoint_dir checkpoints/myCP/ --a1 1 --a2 1 --a3 1
train AFL:
python main.py --train --loop_count 2 --checkpoint_dir checkpoints/myCP/ --load_epoch 199 --a1 1 --a2 1 --a3 1

test pretrained:
python main.py --loop_count 2 --checkpoint_dir checkpoints/dualMiddle/ --dual_input --load_epoch 379 --a1 0 --a2 0.2 --a3 0
