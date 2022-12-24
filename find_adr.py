from z3 import *
from z3_utils import load_model, generate_APs, load_prototype
import argparse
import numpy as np
import matplotlib.pyplot as plt
from time import time
from func_timeout import func_set_timeout


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST', choices=['MNIST','FashionMNIST'])
    parser.add_argument('--mode', default='all', choices=['all', 'customized'])
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--target', default=0, type=int)
    parser.add_argument('--boundary', default=1, type=int)
    parser.add_argument('--r', default=0.2, type=float)
    parser.add_argument('--adv', action='store_true')

    return parser.parse_args()

d = 784 # dimension of input
m = 3   # layers of SDN

@func_set_timeout(600)
def generate_example(dataset, target, boundary, adversarial=False):
    start_time = time()
    N, W, B, g, a = load_model(dataset)
    Input = [Real(f"x_{i}") for i in range(d)]
    Hidden = [[Real(f"h_{i}_{j}") for j in range(N[i])] for i in range(m-1)]
    _Hidden = [[Real(f"_h_{i}_{j}") for j in range(N[i])] for i in range(m-1)]
    Output = [Real(f"y_{i}") for i in range(10)]
    
    # Input constraint
    C_input = [Input[i] >= 0 for i in range(d)] + [Input[i] <= 1 for i in range(d)]

    # Forward constraint (linear part)
    C_linear = []
    for i in range(m):
        for j in range(N[i]):
            if i == 0:
                C_linear.append(_Hidden[i][j] == Sum([W[i][j][k] * Input[k] for k in range(d)] + [B[i][j]]))
            elif i == m-1:
                C_linear.append(Output[j] == Sum([W[i][j][k] * Hidden[i-1][k] for k in range(N[i-1])] + [B[i][j]]))
            else:
                C_linear.append(_Hidden[i][j] == Sum([W[i][j][k] * Hidden[i-1][k] for k in range(N[i-1])] + [B[i][j]]))
    
    # Boundary constraint
    if adversarial:
        C_boundary = [Output[target] == Output[boundary] - 0.01]
    else:
        C_boundary = [Output[target] == Output[boundary]]
    for k in range(10):
        if k == target or k == boundary:
            continue
        C_boundary.append(Output[target] >= Output[k])
    
    # Meaningful constraint
    r = args.r
    P = load_prototype(dataset, target)
    C_meaningful = [Input[i] <= P[i] + r for i in range(d)] + [Input[i] >= P[i] - r for i in range(d)]
    
    for AP in generate_APs(N):
        s = Then('simplify', 'normalize-bounds', 'solve-eqs', 'smt').solver()
        s.add(C_input)
        s.add(C_linear)
        s.add(C_boundary)
        s.add(C_meaningful)
        # Forward constraint (activation part)
        C_activation = []
        # Activation Condition
        C_ap = []
        for i in range(m-1):
            Act = AP[2*i]
            Ina = AP[2*i + 1]
            for j in range(N[i] // g):
                if j == Act:
                    C_activation += [Hidden[i][j*g + l]==_Hidden[i][j*g + l] * a for l in range(g)]
                    C_ap += [_Hidden[i][j*g + l] > 0 for l in range(g)]
                elif j == Ina:
                    C_activation += [Hidden[i][j*g + l]==0 for l in range(g)]
                    C_ap += [_Hidden[i][j*g + l] < 0 for l in range(g)]
                else:
                    C_activation += [Hidden[i][j*g + l]==_Hidden[i][j*g + l] for l in range(g)]
        s.add(C_activation)
        s.add(C_ap)
        #print(len(s.assertions()))
        check = s.check() 
        use_time = time() - start_time
        print(check, f'use time {use_time:.1f}s')
        if check == sat:   
            X = [s.model()[Input[i]] for i in range(d)]
            X = [float(x.as_fraction()) for x in X]
            X = np.array(X).reshape(28,28)
            #print(X)
            
            return X
        else:
            continue
    return None # No examples found in any AP
        


if __name__ == '__main__':
    args = get_args()
    dataset = args.dataset
    if args.mode == 'all':
        start = args.start
        for i in range(start, 10):
            j = (i+1)%10
            try:
                x = generate_example(dataset, i, j, args.adv)
                if x is not None:
                    plt.axis('off')
                    plt.xticks([])
                    plt.yticks([])
                    plt.imshow(x)
                    plt.savefig(f'generate_figs/{dataset}_{i}_{(i+1)%10}{"_adv" if args.adv else ""}', 
                            bbox_inches='tight', dpi=200)
                    plt.clf()
            except:
                continue
    elif args.mode == 'customized':
        x = generate_example(dataset, args.target, args.boundary)
        if x is not None:
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x)
            plt.savefig(f'generate_figs/{dataset}_{args.target}_{args.boundary}', 
                    bbox_inches='tight', dpi=200)
            plt.clf()
        x = generate_example(dataset, args.target, args.boundary, True)
        if x is not None:
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x)
            plt.savefig(f'generate_figs/{dataset}_{args.target}_{args.boundary}_adv', 
                    bbox_inches='tight', dpi=200)
            plt.clf()
    else:
        raise ValueError
