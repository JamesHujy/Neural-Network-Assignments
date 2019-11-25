python train.py --targeted_attack \
				--use_cross_entropy_loss \
				--alpha 1 \
				--beta 0 \
				--gamma 0 \
				--nr_epoch 4000 \
				--device 1 

python train.py --targeted_attack \
				--use_cross_entropy_loss \
				--alpha 1 \
				--beta 0.0001 \
				--gamma 0 \
				--nr_epoch 4000 \
				--device 1

python train.py --targeted_attack \
				--use_cross_entropy_loss \
				--alpha 1 \
				--beta 0 \
				--gamma 0.01 \
				--nr_epoch 4000 \
				--device 1 

python train.py --use_cross_entropy_loss \
				--targeted_attack \
				--alpha 1 \
				--beta 0.0001 \
				--gamma 0.01 \
				--nr_epoch 4000 \
				--device 1 