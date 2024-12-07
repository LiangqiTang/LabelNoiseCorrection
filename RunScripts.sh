#!/bin/bash
# Running things

# ####### Joint Mixup and Bootstrapping #######

python train.py --h

### Cross entropy (CE)
python train.py --Mixup 'None'  --experiment-name 'CE' \
	--epochs 300 --M 100 250 --noise-level 0  --dataset CIFAR10

### Mixup (M)
python train.py --Mixup 'Static' --BootBeta 'None' --experiment-name 'M' \
	--epochs 300 --M 100 250 --noise-level 0 --dataset CIFAR10

### Static mixup with hard bootstrapping (M-DYR-H-BMM)
python train.py --Mixup 'Static' --BootBeta 'Hard' --experiment-name 'M-DYR-H-BMM' \
	--epochs 300 --M 100 250 --noise-level 80 --reg-term 1.0 --dataset CIFAR10

### Static mixup with soft bootstrapping (M-DYR-S-BMM)
python train.py --Mixup 'Static' --BootBeta 'Soft' --experiment-name 'M-DYR-S-BMM' \
	--epochs 300 --M 100 250 --noise-level 80 --reg-term 1.0 --dataset CIFAR10

### Static mixup with hard bootstrapping (M-DYR-H-GMM)
python train.py --Mixup 'Static' --BootBeta 'Hard' --experiment-name 'M-DYR-H-GMM' \
	--epochs 300 --M 100 250 --noise-level 80 --reg-term 1.0 --dataset CIFAR10 --MixtureModel 'GMM'

### Static mixup with soft bootstrapping (M-DYR-S-GMM)
python train.py --Mixup 'Static' --BootBeta 'Soft' --experiment-name 'M-DYR-S-GMM' \
	--epochs 300 --M 100 250 --noise-level 80 --reg-term 1.0 --dataset CIFAR10 --MixtureModel 'GMM'

### Dynamic mixup with hard bootstrapping (MD-DYR-H)
python train.py --Mixup 'Dynamic' --experiment-name 'MD-DYR-H' \
	--epochs 300 --M 100 250 --noise-level 90 --reg-term 1.0 --dataset CIFAR10


### Dynamic mixup with soft to hard bootstrapping (MD-DYR-SH)
python train.py --Mixup 'Dynamic' --experiment-name 'MD-DYR-SH' \
	--epochs 300 --M 100 250 --noise-level 90 --reg-term 1.0 --dataset CIFAR10

### Mixup for tinyimagenet (M-imagenet)
python train.py --Mixup 'Static' --BootBeta 'None' --experiment-name 'M-imagenet' --epochs 100 --M 30 80 --noise-level 0 --dataset TinyImageNet --batch-size 256 --lr 0.001

### Static mixup with hard bootstrapping for tinyimagenet(M-DYR-H-BMM-imagenet)
python train.py --Mixup 'Static' --BootBeta 'Hard' --experiment-name 'M-DYR-H-BMM-imagenet' --epochs 100 --M 30 80 --noise-level 0 --reg-term 1.0 --dataset TinyImageNet --batch-size 256 --lr 0.001

