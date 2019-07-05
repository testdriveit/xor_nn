import math
import random

SIGNAL = [[0, 0], [0, 1], [1, 0], [1, 1]]
TARGET = [[0], [1], [1], [0]]

TEST_SIGNAL = [1, 1]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoidDerivative(x):
    return x * (1 - x)

def printSeparator():
    print('='*30)

### Класс, описывающий нейросеть со скрытым слоем
class NeuralNetwork:

    LEARN_RATE = 0.2

    def __init__(self, inp, hid, out):
        '''
            inp - количество входных нейронов
            hid - количество нейронов в скрытом слое
            out - количество выходных нейронов
        '''
        
        self.inp = inp
        self.hid = hid
        self.out = out
        self.wih = [[0 for _ in range(hid)] for _ in range(inp + 1)]
        self.who = [[0 for _ in range(out)] for _ in range(hid + 1)]
        self.hidden_output = [0 for _ in range(hid)]
        self.output = [0 for _ in range(out)]
        self.erro = [0 for _ in range(out)]
        self.errh = [0 for _ in range(hid)]
        
        for i in range(inp + 1):
            for j in range(hid):
                self.wih[i][j] =  random.random()
        for i in range(hid + 1):
            for j in range(out):
                self.who[i][j] = random.random()

    def printNetwork(self):
        print('Скрытый слой: ')
        for i in range(self.inp + 1):
            for j in range(self.hid):
                print('%.4f'%self.wih[i][j], end = ' ')
            print()
        print()
        print('Выходы скрытого слоя: ')
        for i in range(self.hid):
            print('%.4f'%self.hidden_output[i], end = ' ')
        print('\n')
        print('Выходной слой: ')
        for i in range(self.hid + 1):
            for j in range(self.out):
                print('%.4f'%self.who[i][j], end = ' ')
            print()

        print()
        print('Выход сети: ')
        for i in range(self.out):
            print('%.4f'%self.output[i], end = ' ')
        print()

    def feedForward(self, signal):
        tmp_signal = signal + [1]
        
        for i in range(self.hid):
            tmp_sum = 0.0
            for j in range(self.inp + 1):
                tmp_sum += tmp_signal[j] * self.wih[j][i]
            self.hidden_output[i] = sigmoid(tmp_sum)

        tmp_hidden_output = self.hidden_output + [1]
        for i in range(self.out):
            tmp_sum = 0.0
            for j in range(self.hid + 1):
                tmp_sum += tmp_hidden_output[j] * self.who[j][i]
            self.output[i] = sigmoid(tmp_sum)

    def backPropagate(self, signal, target):
        for i in range(self.out):
            self.erro[i] = (target[i] - self.output[i])*sigmoidDerivative(self.output[i])

        for i in range(self.hid):
            self.errh[i] = 0.0
            for j in range(self.out):
                self.errh[i] += self.erro[j]*self.who[i][j]
            self.errh[i] *= sigmoidDerivative(self.hidden_output[i])

        for i in range(self.out):
            for j in range(self.hid):
                self.who[j][i] += (self.LEARN_RATE * self.erro[i]*self.hidden_output[j])
            self.who[self.hid][i] += (self.LEARN_RATE * self.erro[i])

        for i in range(self.hid):
            for j in range(self.inp):
                self.wih[j][i] += (self.LEARN_RATE * self.errh[i]*signal[j])
            self.wih[self.inp][i] += (self.LEARN_RATE * self.errh[i])

    def trainIteration(self, signal, target):
        self.feedForward(signal)
        self.backPropagate(signal, target)      
        
        

if __name__ == '__main__':
    random.seed()
    nn = NeuralNetwork(2, 5, 1)
    nn.printNetwork()
    printSeparator()
        
    for i in range(5000):
        for signal, target in zip(SIGNAL, TARGET):
            nn.trainIteration(signal, target)
    
    nn.printNetwork()
    printSeparator()

    for signal in SIGNAL:
        nn.feedForward(signal)
        print(signal, end = '')
        print(nn.output)
