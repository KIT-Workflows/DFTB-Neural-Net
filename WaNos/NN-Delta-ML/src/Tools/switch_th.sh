# Script to Switch Keras Backend To Theano
source deactivate 
cp ~/.keras/keras_th.json ~/.keras/keras.json
source activate theano
