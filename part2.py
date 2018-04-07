import numpy as np

bias = 0

train_file = open("./digitdata/optdigits-orig_train.txt")
test_file = open("./digitdata/optdigits-orig_test.txt")

weights = np.zeros((10, 1024))

def print_numpy(array):
    shape = array.shape
    for i in range(shape[0]):
        string = ""
        for j in range(shape[1]):
            string += str(round(array[i][j], 3)) + "\t"
            
        print(string)

def decide(x):
    argmax = -float('inf')
    decision = -1
    for c in range(10):
        product = np.inner(weights[c], x)
        if product > argmax:
            argmax = product
            decision = c
            
    return decision

def read_digit(f):
    string = f.read(1058)
    if len(string) < 1058:
        return None
    
    x = np.empty(1024)
    index = 0
    for i in range(len(string) - 2):
        if string[i] != "\n":
            x[index] = string[i]
            index += 1
    
    f.read(1)
    
    return (x, int(string[len(string) - 1]))

def train():
    num_epoch = 5
    training_curve = np.empty(num_epoch)
    for epoch in range(1, 1 + num_epoch):
        correct = 0.0
        total = 0.0
        while(True):
            eta = 1 / epoch
            
            tup = read_digit(train_file)
            if tup is None:
                break
            x = tup[0]
            digit_class = tup[1]
            
            decision = decide(x)
            print("c : " + str(digit_class))
            print("c': " + str(decision) + "\n")
            if digit_class != decision:
                weights[digit_class] = weights[digit_class] + eta * x
                weights[decision] = weights[decision] - eta * x
            else:
                correct += 1
            total += 1
            
        train_file.seek(0)
        
        training_curve[epoch - 1] = correct / total
        
    print(training_curve)

def test():
    confusion = np.zeros((10, 10))
    occurences = np.zeros(10)
    correct = 0.0
    total = 0.0
    while(True):
        tup = read_digit(test_file)
        if tup is None:
            break
        x = tup[0]
        digit_class = tup[1]
        
        decision = decide(x)
        
        occurences[digit_class] += 1
        confusion[digit_class][decision] += 1
        
        if decision == digit_class:
            correct += 1
        total += 1
        
    for x in range(10):
        confusion[x] = confusion[x] / occurences[x]
        
    accuracy = correct / total
        
    print_numpy(confusion)
    print(accuracy)
