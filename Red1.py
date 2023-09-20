#mnist carga la base de datos
#network es el conjunto de fun RNA

import mnist_loader
import network

#los mnist son 3 de tipo .zip y su funcion
#son los datos de entrenamiento de pesos y baias
#para mejorar el proceso
#proceso de entrenamiento
#datos de validación
#datos de prueba que evaluan el entrenamiento

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data) #extrae datos en forma de lista
p1=training_data[0][0] #estrae los datos de prueba
#trabajamos con las 784 entradas de pixeles etiquetadas
#correspondientes a cada imagen
#[(pixeles,digito),(pixeles,digito),...]

#llamamos la fun Network de la libreria para darle estructura
net = network.Network([784, 30, 10])
#no. de neuronas en la capa de entrada escondida y de salida


#usamos la función Stochastic Gradient Descent
#para entrenar a la red
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
#(datos de entrenamiento,epocas,tamaño minibatch,datos de prueba)