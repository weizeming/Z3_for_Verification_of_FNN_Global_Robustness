# Z3_for_Verification_of_FNN_Global_Robustness
This is the repository for our paper "Using Z3 for formal modeling and verification of FNN global robustness".

## For training SDNs
+ `train.py` is used for SDN training. To train a SDN, for example, use<code>python train.py --dataset 'MNIST'</code>
+ `model.py` is used for design SDN architecture.
+ `utils.py` is used for load the datasets.

## For finding ADRs
+ `find_adr.py` is used for create boundary and adversarial examples. For example, use<code>python find_adr.py --dataset 'MNIST' --mode 'all'</code>
+ `AE.py` is used to train autoencoders and save prototypes.
+ `z3_utils` is used to translate the SDN model parameters into z3-readable lists, and other utils.
