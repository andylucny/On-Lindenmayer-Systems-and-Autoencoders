# On-Lindenmayer-Systems-and-Autoencoders

1. use parametric Lindenmayer system for generation of a dataset by dataset_generator.py

2. analyze the dataset via PCA by eigenimages.py and try to generate images

3. train autoencoder by autoencoder.py, check quality of the trained model and create a dataset of feature vectors

4. analyze the latet space of the autoencoder via PCA by eigencodes.py

5. play with the generator of images from feature vectors by generator.py

6. recognize paramaters of the dataset by encoder-regression.py or encoder-perceptron.py

7. train perceptron which generates feature vectors from the dataset parameters by decoder-perceptron.py and employ it for generator of images from the parameters by final-generator.py

see <a href="http://www.agentspace.org/andy/lucny-itat2020-presentation.pdf">presentation on ITAT 2020</a>
