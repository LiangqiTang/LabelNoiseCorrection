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

### Dynamic mixup with soft to hard bootstrapping (MD-DYR-SH)
python train.py --Mixup 'Dynamic' --experiment-name 'MD-DYR-SH' \
	--epochs 300 --M 100 250 --noise-level 90 --reg-term 1.0 --dataset CIFAR10



