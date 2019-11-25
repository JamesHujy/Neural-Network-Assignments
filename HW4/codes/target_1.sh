python train.py --targeted_attack \
				--alpha 1 \
				--beta 0 \
				--gamma 0 \
				--nr_epoch 4000 \
				--device 4 

python train.py --targeted_attack \
				--alpha 1 \
				--beta 0.01 \
				--gamma 0 \
				--nr_epoch 4000 \
				--device 4

python train.py --targeted_attack \
				--alpha 1 \
				--beta 0 \
				--gamma 1 \
				--nr_epoch 4000 \
				--device 4 

python train.py --targeted_attack \
				--alpha 1 \
				--beta 0.01 \
				--gamma 1 \
				--nr_epoch 4000 \
				--device 4