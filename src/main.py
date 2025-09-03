from data_loader import DataLoader
from rnn import RNN
import numpy as np

hidden_size = 100
seq_length = 25
learning_rate = 0.01

data_loader = DataLoader("data/shakespeare.txt")
dataset = data_loader.load_data()

rnn = RNN(dataset["vocab_size"], hidden_size, seq_length, learning_rate)

n, p = 0, 0
hprev = np.zeros((hidden_size, 1))
smooth_loss = -np.log(1.0 / dataset["vocab_size"]) * seq_length

while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p+seq_length+1 >= len(dataset["data"]) or n == 0: 
        hprev = np.zeros((hidden_size, 1)) # reset RNN memory
        p = 0 # go from start of data

    inputs = [dataset["char_to_ix"][ch] for ch in dataset["data"][p:p+seq_length]]
    targets = [dataset["char_to_ix"][ch] for ch in dataset["data"][p+1:p+seq_length+1]]

    # sample from the model now and then
    if n % 100 == 0:
        sample_ix = rnn.sample(hprev, inputs[0], 200)
        txt = ''.join(dataset["ix_to_char"][ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))

    xs, hs, ys, ps = rnn.forward(inputs, hprev)
    loss = rnn.compute_loss(ps, targets)

    grads = rnn.backward(xs, hs, ps, targets)
    rnn.update_params(grads)

    hprev = hs[len(inputs)-1]

    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0: 
        print('iter %d, loss: %f' % (n, smooth_loss))

    p += seq_length
    n += 1

