import math
import random

SIGNAL = [0, 1]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoidDerivative(x):
    return x * (1 - x)

def printSeparator():
    print('='*30)

### Класс, описывающий нейросеть со скрытым слоем
class NeuralNetwork:

    def __init__(self, inp, hid, out, bias = True):
        '''
            inp - количество входных нейронов
            hid - количество нейронов в скрытом слое
            out - количество выходных нейронов
        '''
        self.bias = bias
        self.inp = inp
        self.hid = hid
        self.out = out
        self.wih = [[0 for _ in range(hid)] for _ in range(inp)]
        self.who = [[0 for _ in range(out)] for _ in range(hid)]
        self.hidden_output = [0 for _ in range(hid)]
        self.output = [0 for _ in range(out)]
        for i in range(inp):
            for j in range(hid):
                self.wih[i][j] =  random.random()
        for i in range(hid):
            for j in range(out):
                self.who[i][j] = random.random()

    def printNetwork(self):
        print('Скрытый слой: ')
        for i in range(self.inp):
            for j in range(self.hid):
                print('%.4f'%self.wih[i][j], end = ' ')
            print()
        print()
        print('Выходы скрытого слоя: ')
        for i in range(self.hid):
            print('%.4f'%self.hidden_output[i], end = ' ')
        print('\n')
        print('Выходной слой: ')
        for i in range(self.hid):
            for j in range(self.out):
                print('%.4f'%self.who[i][j], end = ' ')
            print()

        print()
        print('Выход сети: ')
        for i in range(self.out):
            print('%.4f'%self.output[i], end = ' ')
        print()

    def feedForward(self, signal):
        for i in range(self.inp):
            for j in range(self.hid):
                self.hidden_output[j] = self.hidden_output[j] + signal[i] * self.wih[i][j]
        self.hidden_output = [sigmoid(x) for x in self.hidden_output]

        for i in range(self.hid):
            for j in range(self.out):
                self.output[j] = self.output[j] + self.hidden_output[i] * self.who[i][j]
        self.output = [sigmoid(x) for x in self.output]
        

if __name__ == '__main__':
    random.seed()
    nn = NeuralNetwork(2, 2, 1)
    nn.printNetwork()
    nn.feedForward(SIGNAL)
    printSeparator()
    nn.printNetwork()
        
