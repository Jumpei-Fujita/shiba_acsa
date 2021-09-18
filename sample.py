import numpy as np
import torch

def main():
    for i in range(10):
        print(np.random.randint(100))
        print(torch.tensor([i]).cuda())

if __name__ == '__main__':
    main()