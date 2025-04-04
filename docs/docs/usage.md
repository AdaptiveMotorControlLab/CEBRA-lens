# 📊 How to use?

The CEBRA-Lens package allows for analyzing the embeddings of a single model, but also offers the functionality of comparing embeddings and behavior through layers between models. For this purpose the code contains the `class MultiModel which has an attribute of the defined class metric, by running the class function compute of the MultiModel it will run the defined metric on all the models inside the activations dictionary which contains all activations per layer for models. This analysis can be done also on the activations of one model by only defining the metric class.
