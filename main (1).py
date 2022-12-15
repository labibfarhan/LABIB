import numpy as np 
X = np.array(([10, 8], [6, 10], [9, 2]), dtype=float)
y = np.array(([920], [860], [890]), dtype=float)

X = X/np.amax(X,axis=0) # Normalize 
y = y/1000
X
y
def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid_grad(x):
    return x * (1 - x)

# Variable initialization
epoch=10000
eta =0.2
input_neurons = 2
hidden_neurons = 3
output_neurons = 1
wh=np.random.uniform(size=(input_neurons,hidden_neurons))	# 2x3 
bh=np.random.uniform(size=(1,hidden_neurons))	# 1x3 
wout=np.random.uniform(size=(hidden_neurons,output_neurons)) # 1x1 
bout=np.random.uniform(size=(1,output_neurons))
for i in range(epoch):
    h_ip=np.dot(X,wh) + bh 
    h_act = sigmoid(h_ip) 
    o_ip=np.dot(h_act,wout) + bout
    output = sigmoid(o_ip)
    
    Eo = y-output 
    outgrad = sigmoid_grad(output)
    d_output = Eo* outgrad
    
    Eh = d_output.dot(wout.T)
    hiddengrad = sigmoid_grad(h_act) 
    d_hidden = Eh * hiddengrad
    
    wout += h_act.T.dot(d_output) *eta 
    wh += X.T.dot(d_output) *eta

print("Normalized Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)