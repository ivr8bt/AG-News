# AG-News

# Abstract:

I analyzed the AG news dataset using a variety of neural networks. This is a text dataset that consisted of over 120,000 news articles split up between world, sports, business and science/technology articles. I used vectorized vocabularies to analyze the text utilizing either one-hot encoding or word embedding. I utilized 6 simple RNNS, 1 GRU, 7 LSTMs and 1 CNN. I found that most models (as long as they were bidirectional) produced similar results of around 86% accuracy on the test set. The best model achieved an accuracy of 86.8% on the test set. Training accuracy is increased by increasing the number of nodes in the RNN or LSTM layer, but that increase causes the model to take far longer to train for only minimal gains in text accuracy.	

# Introduction: 

The experiments done here are categorizing whether news articles are World news, sports news, business news or science/tech news. I use 6 simple RNNs, 1 GRU, 7 LSTMs and a 1-dimensional CNN to categorize the text data. 

# Literature Review:

Recurrent neural networks were the standard for working with sequential data before memory-based models like LSTM (Kalita). RNNs incorporate feed-forward data and can deal with sequential information in a way that other algorithms cannot (Kalita). The hidden state of a RNN can be used to record sequences of data like words in a sentence, which would be impossible otherwise. RNNs are prone to vanishing or exploding gradients, which is an issue that can be solved with LSTMs (Kalita). LSTMs consist of 3 gates: input, forget and output. GRUs only consist of 2: update and reset (Srivatsavaya). Since LSTM has more gates, it has more parameters than GRU, but that also makes it more prone to overfitting (Srivatsavaya). GRUs also tend to be faster than LSTMs (Srivatsavaya). Both are better than using a traditional RNN.

# Methods:

The data used was the AG news dataset obtained using tensorflow-datasets package in python. This provided a test set of 7,000 articles and a train set of 114,000 articles with a validation set of 6,000 articles. The possible labels for the articles are world, sports, business and science/tech. The labels were then assigned as categorical variables. Each of the datasets are vectorized to ints using a function with a max sequence length of 150 and max tokens of 1,000. These vectorized datasets are then used for the following models. All models unless otherwise noted use word embeddings with an input dimension of 1000 and output dimension of 256. After the embedding each model uses a bidirectional layer utilizing either simpleRNN, GRU or LSTM with 32 nodes unless otherwise noted. Following this, there is then a dropout layer of 0.5 unless otherwise noted. Lastly, a final layer for every model is a dense layer of 4 nodes with a ‘softmax’ activation function for final classification. All models used ‘sparse categorical crossentropy’ for the loss function, were trained on accuracy and use ‘rmsprop’ as the optimizer. All models use 10 epochs, but early stopping is implemented when accuracy does not improve, so some models do not complete all 10 epochs. 15 different models were utilized and are described below:

* Model 1: This model is a simple RNN and uses a one-hot encoder with a depth of 1000 rather than word embeddings. This model stopped after 9 epochs.
* Model 2: This model is a simple RNN that utilizes word embeddings, all subsequent models use word embeddings with the design indicated earlier in the methods section. This model stopped after 7 epochs.
* Model 3: This model is the same as model 2, but does not use a bidirectional RNN layer. This model stopped after 4 epochs.
* Model 4: This model is the same as model2, but utilizes a GRU rather than a simple RNN. This model stopped after 9 epochs.
* Models 5: This model is the same as model 2, but utilizes a dropout of 0.2 rather than 0.5. This model stopped after 7 epochs.
* Models 6: This model is the same as model 2, but uses 128 nodes rather than 32 in the RNN layer. This model stopped after 8 epochs.
* Model 7: This model is the same as model 2, but uses a L2 regularization with a learning rate of 0.01. This model stopped after 8 epochs.
* Model 8: This model is the same as model 2, but uses a LSTM rather than a simple RNN.
* Model 9: This model is the same as model 8, but uses a LSTM layer of 128 nodes rather than 32.
* Model 10: This model is the same as model 8, but does not utilize a bidirectional layer. This model stopped after 4 epochs.
* Model 11: This model is the same as model 8, but uses a dropout of 0.2 rather than 0.5.
* Model 12: This model is the same as model 8, but utilizes L2 regularization with a learning rate of 0.01. This model stopped after 7 epochs.
* Model 13: This model is the same as model 12, but utilizes a learning rate of 0.1.
* Model 14: This model is the same as model 8, but uses a word embedding with an input of 10,000 rather than 1000.
* Model 15: This model is a 1D convolutional neural net utilizing 128 nodes and padding of 7 with a max pooling layer with a stride of 5. It uses a ‘relu’ activation function. Everything else is the same as what is indicated at the beginning of the methods section. This model stopped after 4 epochs. Accuracies and losses were recorded for the training, validation and testing sets for each model. The process time that it took to run the model was also reported.

# Results:

The whole dataset contains equal numbers of each article type. Preliminary data exploration lets us see the percentage of non-vocabulary words in each document of vocabularies of certain sizes. With a vocabulary size of 1000, the average percent of words that are not in the vocabulary is around 40%. In a vocab size of 10,000 that drops to around 10% and in a vocab size of 5000 it is around 15%. The most common words are what we would think they would be: ‘ ‘, [UNK], the, to, a, of, in, and, on, for, 39s, that. Most of the 15 models had very similar test accuracies between 84 and 86%. The notable exceptions were models 3 (RNN) and 10 (LSTM) that did not use bidirectionality. These models had a test accuracy of roughly 25% and only executed 4 epochs since they stalled out around that accuracy. Clearly the bidirectional RNN or LSTM layer is very important. Using a one-hot encoder as opposed to word embedding for the vectorized text also did not make much of a difference. Most models executed in between 1000 and 4000 seconds with exception of the ones that stopped very early. However, in model 9 (LSTM) each epoch took about 2 to 3 times as long as the others due to a greater number of nodes. Some models were much faster than others, but that is because they stopped sooner. However, the CNN had slightly lower training times per epoch and stopped after only 4 epochs yet was still able to perform similarly to the best models. The best performing model on the test set was model 11 with a classification accuracy of 86.8%. However, model 9 did have the highest training accuracy, which was continuing to increase even through epoch 10. If I had more time, I would have investigated this further, but the model took too long to train. The validation accuracy remained at around 87% though meaning that the model was likely just overtraining.	

# Conclusion:

Layers without bidirectionality performed very poorly. Vocabulary size was not terribly important in determining which category an article belongs to. Layers that incorporated more nodes in the LSTM layer performed slightly better, but took far longer to train. Depending on the goal and how long you have to train will determine what approach is best. I think an LSTM approach is best for a Chatbot since this will take a very long time to train anyway, and you want the RNN to perform as well as possible if it is communicating with customers.
	
# References:
Kalita, Debasish. 2024. “A Brief Overview of Recurrent Neural Networks (RNN)”.  Analytics Vidhya. Accessed May 12, 2024. https://www.analyticsvidhya.com/blog/2022/03/a-brief-overview-of-recurrent-neural-networks-rnn/
Srivatsavaya, Prudhviraju. 2024. “LSTM vs GRU”. Medium.Accessed May 12, 2024. https://medium.com/@prudhviraju.srivatsavaya/lstm-vs-gru-c1209b8ecb5a
