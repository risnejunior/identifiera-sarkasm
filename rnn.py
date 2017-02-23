# 1. Läsa in data från twitter, få datan på form x=tweettext, y={1,0}
# 2. Transformera x=text -> x=num2vec(text)
#    Se till att x är en reell vektor med storlek som bestäms av största ordet
#    i vårt dataset.
# 3. Konstruera nätverket: Vikter, biases, felfunktion

import tensorflow as tf

#Define network parameters
hidden_size = 28 #Number of hidden units. #Note: No intuition of how large this should be
out_size = 2 #Number of classes (2 since sarcastic and non-sarcastic)
n = 10 # n=(Dim. på största ordvektorn) #(10=placeholder)

x_data = tf.placeholder(tf.float32, [None, n]) # n=(Dim. på största ordvektorn)
y_data = tf.placeholder(tf.float32, [None, 2]) #y är på formen [1;0],[0;1]

def RNN(X, weights, biases):
    #add one layer and return the output of this layer given the input
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    
    

