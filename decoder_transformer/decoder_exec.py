import decoder as DECODER
import numpy as np

ENDCOLOR = '\033[0m'
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'

# encoder constructs the K and V vectors. Lets say we have 3
# five-dimensional vectors for each stack. 

K = np.random.uniform(low=-1, high=1, size=(3, 5))
V = np.random.uniform(low=-1, high=1, size=(3, 5))

vocabulary = {  'Hoje': np.array([1,0,0,0,0]),
                'é': np.array([0,1,0,0,0]),
                'domingo': np.array([0,0,1,0,0]),
                'sábado': np.array([0,0,0,1,0]),
                'EOS': np.array([0,0,0,0,1]),
                'START' : np.array([0.2,0.2,0.2,0.2,0.2]) 
                }

START = vocabulary['START']
EOS = vocabulary['EOS']
INPUT_TOKEN = START
OUTPUT_TOKENS = START
LAST_TOKEN = START
X = INPUT_TOKEN

PE = DECODER.ENCODER.Positional_Encoding()
#line below: d_model == d_v for residual connection to work
multi_head_masked_attention = DECODER.Multi_Head_Masked_Attention(n_heads=8, d_model=5, d_k=4, d_v=5) 
encoder_decoder_attention = DECODER.One_Head_Encoder_Decoder_Attention(d_k=4)
ffn = DECODER.ENCODER.FFN(d_v=5, layer_sz=8, d_output=5)

count = 0
while (not np.array_equal(LAST_TOKEN, EOS)) and (count < 10):
    X_PE = PE.compute(X)

    print(BLUE + 'shape of X:', X.shape, ENDCOLOR)
    print(BLUE+'X:\n', X, ENDCOLOR)
    print(BLUE+'X_PE:\n', X_PE, ENDCOLOR)

    output_masked_attention = multi_head_masked_attention.compute(X_PE)

    Q_star = DECODER.ENCODER.add_and_norm(output_masked_attention, X_PE) #residual_connection_1

    output_encoder_decoder_attention = encoder_decoder_attention.compute_1_head_attention(Q=Q_star, K=K, V=V)

    Rc2 = DECODER.ENCODER.add_and_norm(output_encoder_decoder_attention , Q_star)  # residual connection 2

    ffn_result = ffn.compute(Rc2)

    OUTPUT_TOKENS_before_softmax = DECODER.ENCODER.add_and_norm(ffn_result, Rc2) # -------> 3rd residual connection 3

    if OUTPUT_TOKENS_before_softmax.ndim == 1:
        # last softmax:
        OUTPUT_TOKENS = np.exp(OUTPUT_TOKENS_before_softmax) / np.sum(np.exp(OUTPUT_TOKENS_before_softmax))
        position_of_max = np.argmax(OUTPUT_TOKENS)
        OUTPUT_TOKENS = np.eye(OUTPUT_TOKENS.shape[0])[position_of_max]
        LAST_TOKEN = OUTPUT_TOKENS
    else:
        OUTPUT_TOKENS = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, OUTPUT_TOKENS_before_softmax)
        position_of_max = np.argmax(OUTPUT_TOKENS, axis=1)
        OUTPUT_TOKENS = np.eye(OUTPUT_TOKENS.shape[1])[position_of_max]
        LAST_TOKEN = OUTPUT_TOKENS[-1,:]

    X = np.vstack([X, LAST_TOKEN])

    print('shape of OUTPUT_TOKENS:', OUTPUT_TOKENS.shape)

    print(RED+'OUTPUT_TOKENS:\n', OUTPUT_TOKENS, ENDCOLOR)
    print(RED+'LAST_TOKEN:\n', LAST_TOKEN, ENDCOLOR)
    print(RED + '=====================================' + ENDCOLOR)

    count = count + 1

#identifying tokens in dictionary:
OUTPUT_SENTENCE = []
output_sentence_str = ''
for token_pos in range(len(X[:,0])):
    token = X[token_pos,:]
    for name, array in vocabulary.items():
        if np.array_equal(array, token):
            OUTPUT_SENTENCE.append(name)

for token_name in OUTPUT_SENTENCE:
    output_sentence_str += token_name + ' '

print(output_sentence_str)
