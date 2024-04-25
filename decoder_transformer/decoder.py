import math 
import numpy as np

import encoder_classes_link as ENCODER


class One_Head_Masked_Attention:
    def __init__(self, d_model, d_k, d_v):
        self.d_model = d_model
        self.W_mat = ENCODER.W_matrices(self.d_model, d_k, d_v)

    def compute_QKV(self, X):
        self.Q = np.matmul(X, self.W_mat.W_Q)
        self.K = np.matmul(X, self.W_mat.W_K)
        self.V = np.matmul(X, self.W_mat.W_V)

    def print_QKV(self):
        print('Q : \n', self.Q)
        print('K : \n', self.K)
        print('V : \n', self.V)

    def compute_1_head_masked_attention(self):
        Attention_scores = np.matmul(self.Q, np.transpose(self.K)) 
        # print('Attention_scores before normalization : \n', Attention_scores)

        if Attention_scores.ndim > 1:
            M = np.zeros(Attention_scores.shape)
            for i in range(Attention_scores.shape[0]):
                for j in range(i+1, Attention_scores.shape[1]):
                    M[i,j] = -np.inf
        else:
            M = 0

        Attention_scores += M
        # print('Attention_scores after masking : \n', Attention_scores)
        Attention_scores = Attention_scores / np.sqrt(self.d_model) 
        # print('Attention scores after Renormalization: \n ', Attention_scores)

        if Attention_scores.ndim > 2:
            Softmax_Attention_Matrix = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, Attention_scores)
        else: 
            Softmax_Attention_Matrix = np.exp(Attention_scores) / np.sum(np.exp(Attention_scores))

        # print('result after softmax: \n', Softmax_Attention_Matrix)

        if Attention_scores.ndim > 1:
            if Softmax_Attention_Matrix.shape[1] != self.V.shape[0]:
                raise ValueError("Incompatible shapes!")

            result = np.matmul(Softmax_Attention_Matrix, self.V)
        else: 
            result = Softmax_Attention_Matrix * self.V
            # result = np.matmul(Softmax_Attention_Matrix, self.V)

        # print('softmax result multiplied by V: \n', result)

        return result

    def backpropagate(self):
        # do smth to update W_mat
        pass


class Multi_Head_Masked_Attention:
    def __init__(self, n_heads, d_model, d_k, d_v):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_concat = self.d_v*self.n_heads
        self.W_0 = np.random.uniform(-1, 1, size=(self.d_concat, self.d_v))
        self.heads = []
        i = 0
        while i < self.n_heads:
            self.heads.append(One_Head_Masked_Attention(d_model=d_model, d_k=d_k , d_v=d_v ))
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

    def compute(self, X):
        self.heads_results = []
        for head in self.heads:
            head.compute_QKV(X)
            self.heads_results.append(head.compute_1_head_masked_attention())

        if X.ndim > 1:
            multi_head_results = np.concatenate(self.heads_results, axis=1)
            V_updated = np.matmul(multi_head_results, self.W_0)
        else:
            multi_head_results = np.concatenate(self.heads_results, axis=0)
            print('Dimension of multihead_results:', multi_head_results.shape)
            print('Dimension of W_0:', self.W_0.shape)
            V_updated = np.matmul(multi_head_results, self.W_0)

        return V_updated

    def back_propagate(self):
        # backpropagate W_0
        # call _backprop for each head
        pass


class One_Head_Encoder_Decoder_Attention:
    def __init__(self, d_k):
        self.d_k = d_k

    def print_QKV(self):
        print('Q : \n', self.Q)
        print('K : \n', self.K)
        print('V : \n', self.V)

    def compute_1_head_attention(self, Q, K, V):
        self.Q = Q #from masked attention in decoder
        self.K = K #from encoder
        self.V = V #final result from encoder

        Attention_scores = np.matmul(self.Q, np.transpose(self.K)) 
        # print('Attention_scores before normalization : \n', Attention_scores)
        Attention_scores = Attention_scores / np.sqrt(self.d_k) 
        # print('Attention scores after Renormalization: \n ', Attention_scores)
        Softmax_Attention_Matrix = np.exp(Attention_scores - np.max(Attention_scores, axis=-1, keepdims=True))
        Softmax_Attention_Matrix /= np.sum(Softmax_Attention_Matrix, axis=-1, keepdims=True)

        # print('result after softmax: \n', Softmax_Attention_Matrix)

        if Softmax_Attention_Matrix.ndim > 1:
            if Softmax_Attention_Matrix.shape[1] != self.V.shape[0]:
                raise ValueError("Incompatible shapes!")

        result = np.matmul(Softmax_Attention_Matrix, self.V)

        return result

    # there is no learnable parameters in encoder-decoder attention


