# Transformer_sentiment
Using the powerful transformer model to implement sentiment analysis upon IMDB movie reviews
![Transformer_sentiment](./images/pic_transf.png)

## Purpose of this project:
1. learn how to build a modulized transformer model from scratch.
2. learn how to adapt transformer modules to a real world use case - sentiment analysis by using custom python code.
3. Practice on some important NLP concepts: tokenization, embedding.
4. Get a deep understanding of how transformer model works by look into the self-attention mechanism

## The final model performance measurement:
Based on some quick and simple model training (roughly 200 epochs), the model achived AUC 0.94 and accuracy of 86%, indicating the outstanding prediction power of transformer model

## The project codes 1 (transformer modules):
Build standard code modules to implement transformer model efficiently. There are two modules, transformer and stepbystep, under the transformer_modules directory. The first one is a modulized implementation of transformer model, and the second one is for implementing various functions such as train, dataloader, predict etc.

p.s. All these modules are inspired by the book <deep learning with pytorch step-by-step>, which I also added some functions (early stopping, automatically print losses based on epoch interval etc.) to fulfill my needs

## The project codes 2 (the jupyter notebook):
In the main code file - the jupyter notebook (Transformer_Sentiment_dev), I developed custom code to adapt the transoformer model as a classification model to fulfill the goal of sentiment analysis

## Study on self-attention's alpha matrix
In the jupyter notebook, I was able to look into the attention scores derived from the self-attention machnism. It is interesting to see how the transformer was able to add more weight on those more relevant texts so eventually being able to deliver solid prediction results

## background: transformer model
To be added

## background: IMDB dataset
To be added



