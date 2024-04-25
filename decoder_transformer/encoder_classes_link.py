import numpy as np 
from scipy.special import softmax
import math

def add_and_norm(array1, array2):
    if array1.ndim != array2.ndim:
        raise ValueError("Incompatible shapes!")

    result = array1 + array2

    if array1.ndim == 1:
        mean = result.mean()
        variance = result.var()
        result =(result - mean)/variance
    else:
        for r in range(array1.shape[0]):
            mean = result[r,:].mean()
            variance = result[r,:].var()
            result[r,:] =(result[r,:] - mean)/variance

    return result

class W_matrices:
    def __init__(self, X_size, d_k, d_v): #X_size == d_model ; d_k == d_q
        self.W_Q = np.random.uniform(low=-1, high=1, size=(X_size, d_k))
        self.W_K = np.random.uniform(low=-1, high=1, size=(X_size, d_k))
        self.W_V = np.random.uniform(low=-1, high=1, size=(X_size, d_v))

    def print_W_matrices(self):
        print('W_Q : \n', self.W_Q)
        print('W_K : \n', self.W_K)
        print('W_V : \n', self.W_V)

class One_Head_Attention:
    def __init__(self, X, d_k, d_v):
        self.d_model = len(X)
        self.W_mat = W_matrices(self.d_model, d_k, d_v)

        self.Q = np.matmul(X, self.W_mat.W_Q)
        self.K = np.matmul(X, self.W_mat.W_K)
        self.V = np.matmul(X, self.W_mat.W_V)

    def print_QKV(self):
        print('Q : \n', self.Q)
        print('K : \n', self.K)
        print('V : \n', self.V)

    def compute_1_head_attention(self):
        Attention_scores = np.matmul(self.Q, np.transpose(self.K)) 
        # print('Attention_scores before normalization : \n', Attention_scores)
        Attention_scores = Attention_scores / np.sqrt(self.d_model) 
        # print('Attention scores after Renormalization: \n ', Attention_scores)
        Softmax_Attention_Matrix = np.apply_along_axis(softmax, 1, Attention_scores)
        # print('result after softmax: \n', Softmax_Attention_Matrix)

        result = np.matmul(Softmax_Attention_Matrix, self.V)
        # print('softmax result multiplied by V: \n', result)

        return result

    def backpropagate(self):
        # do smth to update W_mat
        pass

class Multi_Head_Attention:
    def __init__(self, n_heads, X, d_k, d_v):
        self.d_model = len(X)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_concat = self.d_model*self.n_heads
        self.W_0 = np.random.uniform(-1, 1, size=(self.d_concat, self.d_model))
        self.heads = []
        self.heads_results = []
        i = 0
        while i < self.n_heads:
            self.heads.append(One_Head_Attention(X, d_k=d_k , d_v=d_v ))
            i += 1

    def print_W_0(self):
        print('W_0 : \n', self.W_0)

    def print_QKV_each_head(self):
        i = 0
        while i < self.n_heads:
            print(f'Head {i}: \n')
            self.heads[i].print_QKV()
            i += 1

    def print_W_matrices_each_head(self):
        i = 0
        while i < self.n_heads:
            print(f'Head {i}: \n')
            self.heads[i].W_mat.print_W_matrices()
            i += 1

    def compute(self):
        for head in self.heads:
            self.heads_results.append(head.compute_1_head_attention())

        multi_head_results = np.concatenate(self.heads_results, axis=1)

        V_updated = np.matmul(multi_head_results, self.W_0)
        return V_updated

    def back_propagate(self):
        # backpropagate W_0
        # call _backprop for each head
        pass

class FFN:
    def __init__(self, d_v, layer_sz, d_output):
        self.layer1_sz = layer_sz
        self.layer1_weights = np.random.uniform(low=-1, high=1, size=(d_v, layer_sz))
        self.layer2_weights = np.random.uniform(low=-1, high=1, size=(layer_sz, d_output))

    def compute(self, V_updated):
        result = np.matmul(V_updated, self.layer1_weights)
        result = np.matmul(result, self.layer2_weights)

        return result

    def backpropagate_ffn(self):
        pass

class Positional_Encoding:
    def __init__(self):
        pass

    def compute(self, X):
        self.PE_shape = X.shape
        self.PE = np.empty(self.PE_shape)
        if X.ndim == 2:
            self.d_model = self.PE_shape[1]
            for i in range(self.PE_shape[0]): 
                for j in range(int(self.PE_shape[1]/2)):
                    self.PE[i,2*j] = math.sin(i/(10000**(2*j/self.d_model)))
                    self.PE[i,2*j+1] = math.cos(i/(10000**(2*j/self.d_model)))
        else:
            self.d_model = len(X)
            for j in range(int(len(X)/2)):
                self.PE[2*j] = math.sin(1/(10000**(2*j/self.d_model)))
                self.PE[2*j+1] = math.cos(1/(10000**(2*j/self.d_model)))

        return X + self.PE

