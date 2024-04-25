import numpy as np 
import encoder_classes as ENCODER

sentence = "Today is sunday"

vocabulary = ['Today', 'is', 'sunday', 'saturday']

# Initial Embedding (one-hot encoding):

x_1 = [1,0,0,0] # Today 
x_2 = [0,1,0,0] # is 
x_3 = [0,0,1,0] # Sunday
x_4 = [0,0,0,1] # Saturday

X_vocab = [x_1, x_2, x_3, x_4]

init_embbeding = dict(zip(vocabulary, X_vocab))

X = np.stack([init_embbeding['Today'], init_embbeding['is'], init_embbeding['sunday']], axis=0)
# X = np.stack([x_1, x_2, x_3], axis=0) #equivalently

multi_head_attention = ENCODER.Multi_Head_Attention(2, X=X, d_k=4, d_v=4)

multi_head_attention.print_W_matrices_each_head()

multi_head_attention.print_QKV_each_head()

multi_head_attention.print_W_0()

V_updated_by_context = multi_head_attention.compute()

print(V_updated_by_context)
