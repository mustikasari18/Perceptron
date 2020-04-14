import numpy as np


class Perc(object):

    def __init__(ident, datatinput, nilaithreshold=100, nilaipembelajaran=0.01):
        ident.nilaithreshold = nilaithreshold
        ident.nilaipembelajaran = nilaipembelajaran
        ident.berat = np.zeros(datatinput + 1)

    def predict(ident, input):
        hasilj = np.dot(input, ident.berat[1:]) + ident.berat[0]
        if hasilj > 0:
            aktivasi = 1
        else:
            aktivasi = 0
        return aktivasi

    def train(ident, inputdt, output):
        for _ in range(ident.nilaithreshold):
            for inputan, label in zip(inputdt, output):
                nilaiprediksi = ident.predict(inputan)
                ident.berat[1:] += ident.nilaipembelajaran * (label - nilaiprediksi) * input
                ident.berat[0] += ident.nilaipembelajaran * (label - nilaiprediksi)


inputdt = []
inputdt.append(np.array([1, 1]))
inputdt.append(np.array([1, 0]))
inputdt.append(np.array([0, 1]))
inputdt.append(np.array([0, 0]))

output = np.array([1, 0, 0, 0])

perceptron = Perceptron(2)
perceptron.train(inputdt, output)

inputan = np.array([1, 1])
perceptron.predict(inputan)
# => 1

inputan = np.array([0, 1])
perceptron.predict(inputan)
# => 0