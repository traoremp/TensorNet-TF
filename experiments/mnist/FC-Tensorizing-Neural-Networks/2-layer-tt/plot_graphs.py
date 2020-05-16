import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot results')
parser.add_argument('--tests', '-t', type=str, default = '100', help='test results to plot')
parser.add_argument('--over_step', '-os', type=int, default = '20', help='overview step')
args = parser.parse_args()


samples = str.split(args.tests, ',')
tests = '666,666'
os = args.over_step
samples = str.split(tests, ',')
samples = [int(i) for i in samples]

data = train_loss = np.load('t_loss_%d.npy' %(samples[0]))
x = np.arange(len(data))*os+1

f1 = plt.figure(1)
for i in samples:    
    train_loss = np.load('t_loss_%d.npy' %(i))
    valid_loss = np.load('v_loss_%d.npy' %(i))
    plt.plot(x, train_loss)
    plt.plot(x, valid_loss+1)
plt.legend()
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
  
f2 = plt.figure(2)  
for i in samples:
    train_acc = np.load('t_acc_%d.npy' %(i))
    valid_acc = np.load('v_acc_%d.npy' %(i))
    plt.plot(x, train_acc)
    plt.plot(x, valid_acc+1)
plt.legend()
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
