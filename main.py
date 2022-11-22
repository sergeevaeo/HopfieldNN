import numpy as np
from neupy import algorithms

with open("AAPL.csv") as file:
    stock = [row.strip() for row in file]

# заменим каждый элемент массива

size = 210

riseWane = [0] * size

for i in range(size):
    if i == 0:
        riseWane[i] = 1
    else:
        if stock[i] > stock[i - 1]:
            riseWane[i] = 1
        else:
            riseWane[i] = 0

matrixOne = np.matrix([riseWane[:42]])
matrixTwo = np.matrix([riseWane[42:84]])
matrixThree = np.matrix([riseWane[84:126]])
matrixFour = np.matrix([riseWane[126:168]])
matrixFive = np.matrix([riseWane[168:210]])
print("Массив исходных акций")
print(stock)
print("Увеличение и уменьшение акций")
print(riseWane)
print("Исходные значения акций с шагом 42")
print(matrixOne)
print(matrixTwo)
print(matrixThree)
print(matrixFour)
print(matrixFive)

# тренировка
data = np.concatenate([matrixOne, matrixTwo, matrixThree, matrixFour, matrixFive], axis=0)
dhnet = algorithms.DiscreteHopfieldNetwork(mode='async')
dhnet.n_times = 1000
dhnet.train(data)


# изменение некоторых значений


def changing(matrix, length, value):
    changedRiseWane = [0] * length

    for j in range(length):
        if j % value == 0:
            if matrix[j] == 0:
                changedRiseWane[j] = 1
            else:
                changedRiseWane[j] = 0
        else:
            changedRiseWane[j] = matrix[j]

    return changedRiseWane


value = 5
matrixOneC = np.matrix([changing(riseWane, 210, value)[:42]])
matrixTwoC = np.matrix([changing(riseWane, 210, value)[42:84]])
matrixThreeC = np.matrix([changing(riseWane, 210, value)[84:126]])
matrixFourC = np.matrix([changing(riseWane, 210, value)[126:168]])
matrixFiveC = np.matrix([changing(riseWane, 210, value)[168:210]])

print("Меняем некоторые значения")
print(matrixOneC)
print(matrixTwoC)
print(matrixThreeC)
print(matrixFourC)
print(matrixFiveC)

# восстановление исходного
print("Восстановленные")


def recover(matrixC):
    result = dhnet.predict(matrixC)
    return result


result1 = recover(matrixOneC)
result2 = recover(matrixTwoC)
result3 = recover(matrixThreeC)
result4 = recover(matrixFourC)
result5 = recover(matrixFiveC)
print(result1)
print(result2)
print(result3)
print(result4)
print(result5)

total = 0


# сравнение результатов
def compare(matrix, result):
    count = 0
    if (matrix == result).all():
        print("True")
    else:
        for t in range(42):
            if matrix[:, t] != result[:, t]:
                count = count + 1
        print("False")
        print(count)
        print(matrix)
        print(result)


def gettingTotal(matrix, result):
    count = 0
    if (matrix != result).any():
        for t in range(42):
            if matrix[:, t] != result[:, t]:
                count = count + 1
    return count


print("Сравнение")
compare(matrixOne, result1)
compare(matrixTwo, result2)
compare(matrixThree, result3)
compare(matrixFour, result4)
compare(matrixFive, result5)

total = total + gettingTotal(matrixOne, result1)
total = total + gettingTotal(matrixTwo, result2)
total = total + gettingTotal(matrixThree, result3)
total = total + gettingTotal(matrixFour, result4)
total = total + gettingTotal(matrixFive, result5)

value = 210/value
value = value / 210 * 100
accuracy = 100 - total / 210 * 100

print('При искажении значений на', value, '%  значение точности равно', accuracy, '%')

