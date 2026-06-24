"""
viz of decreasing support for L96.
hard coded
"""
import numpy as np
import matplotlib.pyplot as plt

if __name__== '__main__':

    D = 8
    M = 1000
    J = 2

    colors = [
        'lightsteelblue', 'cornflowerblue', 'royalblue', 
        'mediumblue', 'darkblue'
    ]

    plt.figure(figsize=(7,5))
    x = np.array([0, 1, 2])
    for d in range(4, D+1):
        y = np.array([d*J**d, d*(16), d*4])
        plt.axhline(d*16, color='red', linestyle='--', alpha=0.4,
                    label=f'true coarse support sizes'*(d==4))
        plt.axhline(d*4, color='green', linestyle='--', alpha=0.4,
                    label=f'true fine support size'*(d==4))
        plt.plot(x,y,marker='o', color=colors[d-4], alpha=0.8,
                 label=f'D = {d}')

    plt.legend()
    plt.xticks(range(3))
    plt.xlabel('Iteration count')
    plt.ylabel('Support size')
    plt.title(r'Support size for Lorenz96, D=$4,\dots,8$, M=1000, $\lambda_{TT} = \lambda_{flat} = 10^{-2}$')
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    plt.show()