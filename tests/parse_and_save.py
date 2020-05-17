import matplotlib.pyplot as plt
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--file_name', '-fn', type=str, help='file_name')
args = parser.parse_args()

fl = args.file_name
filename = fl + '.txt'

with open(filename) as file:
#    file_contents = file.read()
    training_loss1 = []
    for ln in file:
        if ln.startswith("Step "):
            line = ln.split()
            training_loss1.append(float(line[4]))
with open(filename) as file:  
    train_loss = []
    train_percision = []
    for ln in file:
        if ln.startswith("Training Data Eval"):
            ln2 = next(file)
            line = ln2.split()
            train_loss.append(float(line[9]))
            train_percision.append(float(line[11]))
with open(filename) as file:         
    valid_loss = []
    valid_percision = []
    for ln in file:
        if ln.startswith("Validation Data Eval"):
            ln2 = next(file)
            line = ln2.split()
            valid_loss.append(float(line[9]))
            valid_percision.append(float(line[11]))

np.save('t_loss_100_%s' %(fl) , training_loss1)
np.save('t_loss_%s' %(fl) , train_loss)
np.save('v_loss_%s' %(fl) , valid_loss)
np.save('t_perc_%s' %(fl) , train_percision)
np.save('v_perc_%s' %(fl) , valid_percision)

f1 = plt.figure(1)
x = [(i+1)*1000 for i in range(len(train_loss))]
plt.plot(x,train_loss, label = 'train')
plt.plot(x,valid_loss, label= 'validation')
plt.grid(True)
plt.title(fl)
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('loss_%s.png' %(fl))
plt.show()


f1 = plt.figure(1)
plt.plot(x,train_percision, label = 'train')
plt.plot(x,valid_percision, label= 'validation')
plt.grid(True)
plt.title(fl)
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('acc_%s.png' %(fl))
plt.show()

