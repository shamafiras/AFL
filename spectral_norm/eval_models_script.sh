out_file="eval_dualMid_a0a01a0_out.txt"

for loops in 1 2 3 4
do
	echo "======= evaluating with loop $loops ======================="
	python main.py --checkpoint_dir checkpoints/dualMiddle/ --loop_count $loops --a1 0 --a2 0.2 --a3 0 --load_epoch 379 >> $out_file
	echo "==========================================================="
done

