import numpy as np
import matplotlib.pyplot as plt

smoothing = 0.1

train_file = open("./digitdata/optdigits-orig_train.txt")
test_file = open("./digitdata/optdigits-orig_test.txt")

prob_list = np.zeros((10, 32, 32))
prob_digit = np.zeros(10)

confusion = np.zeros((10, 10))
total_accuracy = 0.0

accuracy_array = np.empty(10)

def test_smoothing():
    global prob_list, prob_digit, confusion, total_accuracy, smoothing
    for i in range(1, 11):
        smoothing = i
        train()
        test_eval()
        accuracy_array[i-1] = total_accuracy
        train_file.seek(0)
        test_file.seek(0)
        prob_list = np.zeros((10, 32, 32))
        prob_digit = np.zeros(10)
        confusion = np.zeros((10,10))
        total_accuracy = 0.0
    
    print(accuracy_array)

def print_prob_list(x):
    for i in range(32):
        string = ""
        for j in range(32):
            string += str(prob_list[x][i][j]) + " "
        
        print(string)

def train():
    feature_occurences = np.zeros((10, 32, 32))
    digit_occurences = 0
    while(True):
        string = train_file.read(1058)
        if len(string) < 1058:
            break
        
        digit_class = int(string[len(string) - 1])
        
        prob_digit[digit_class] += 1
        
        digit_occurences += 1
        
        i = 0
        j = 0
        for x in range(len(string) - 2):
            if string[x] == '\n':
                j = 0
                i += 1
                continue
            
            feature_occurences[digit_class][i][j] += 1
            prob_list[digit_class][i][j] += int(string[x])
            j += 1
        
        train_file.read(1)
    
    for x in range(10):
        for i in range(32):
            for j in range(32):
                prob_list[x][i][j] = (prob_list[x][i][j] + smoothing) / (feature_occurences[x][i][j] + smoothing * 2)
                
        prob_digit[x] = prob_digit[x] / digit_occurences
    
def test_eval():
    global total_accuracy
    test_list = []
    
    accuracy = np.zeros(10)
    occurences = np.zeros(10)
    
    correct = 0.0
    total = 0.0
    
    best = np.zeros((10, 32, 32), dtype = int)
    best_prob = np.full(10, -np.inf)
    worst = np.zeros((10, 32, 32), dtype = int)
    worst_prob = np.full(10, np.inf)
    best_class = np.zeros(10)
    worst_class = np.zeros(10)
    
    while(True):
        string = test_file.read(1058)
        if len(string) < 1058:
            break
       
        test_digit = np.zeros((32,32), dtype = int)
        digit_class = int(string[len(string) - 1])
        occurences[digit_class] += 1
        
        i = 0
        j = 0
        for x in range(len(string) - 2):
            if string[x] == '\n':
                j = 0
                i += 1
                continue
            
            test_digit[i][j] = int(string[x])
            j += 1
        
        test_file.read(1)
        
        prob = np.zeros(10)
        max_prob = (-float("inf"), 0)
        for x in range(10):
            prob[x] += np.log(prob_digit[x])
            for i in range(32):
                for j in range(32):
                    if test_digit[i][j] == 1:
                        prob[x] += np.log(prob_list[x][i][j])
                    elif test_digit[i][j] == 0:
                        prob[x] += np.log(1 - prob_list[x][i][j])
                    
            if max_prob[0] < prob[x]:
                max_prob = (prob[x], x)
                
        if max_prob[1] == digit_class:
            accuracy[digit_class] += 1
            correct += 1
            
        total += 1
        
        if prob[digit_class] > best_prob[digit_class]:
            best_prob[digit_class] = prob[digit_class]
            best[digit_class] = np.copy(test_digit)
            best_class[digit_class] = max_prob[1]
        if prob[digit_class] < worst_prob[digit_class]:
            worst_prob[digit_class] = prob[digit_class]
            worst[digit_class] = np.copy(test_digit)
            worst_class[digit_class] = max_prob[1]
        
        
        confusion[digit_class][max_prob[1]] += 1
        
        test_list.append(max_prob[1])
        
    accuracy  = accuracy / occurences
    total_accuracy = correct / total
    for x in range(10):
        confusion[x] = confusion[x] / occurences[x]
    
    print("Classifications:")
    print(test_list)
    print("Accuracy per digit:")
    print(accuracy)
    print("Overall accuracy:")
    print(total_accuracy)
    print("Confusion Matrix:")
    print_numpy(confusion)
    print("Best: ")
    for x in range(10):
        print_digit(best[x])
        print(best_class[x])
    print("Worst:")
    for x in range(10):
        print_digit(worst[x])
        print(worst_class[x])
    
def print_numpy(array):
    shape = array.shape
    for i in range(shape[0]):
        string = ""
        for j in range(shape[1]):
            string += str(round(array[i][j], 3)) + "\t"
            
        print(string)
        
def print_digit(array):
    shape = array.shape
    for i in range(shape[0]):
        string = ""
        for j in range(shape[1]):
            string += str(array[i][j])
            
        print(string)
    
def odds(c1, c2):
    odds_array = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            odds_array[i][j] = prob_list[c1][i][j] / prob_list[c2][i][j]
            
    return odds_array

def heatmap(c1, c2):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.imshow(prob_list[c1], cmap="hot", interpolation="nearest")
    fig.colorbar(cax)
    
    fig = plt.figure(2)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.imshow(prob_list[c2], cmap="hot", interpolation="nearest")
    fig.colorbar(cax)
    
    fig = plt.figure(3)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.imshow(np.log(odds(c1, c2)), cmap="hot", interpolation="nearest")
    fig.colorbar(cax)
    
    plt.show()
    
def odds_map():
    heatmap(2, 8)
    heatmap(5, 9)
    heatmap(4, 7)
    heatmap(4, 8)
