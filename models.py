import csv
import math
import random
'''
author: Harshavardhan Ramamurthy
netID: hxr7678
studentID: 1001767678
'''

class LinearRegression:
    def __init__(self, file):
        self.beta = []
        self.__DATA = []
        self.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
        self.labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        with open(file, 'rt') as f:
            reader = csv.DictReader(f, fieldnames=self.columns)
            for record in reader:
                self.__DATA.append(
                    ((float(record['sepal_len']), float(record['sepal_wid']), float(record['petal_len']),
                      float(record['petal_wid'])), (self.labels.index(record['class'])))
                )

    def transpose(self, mat):
        return [[mat[j][i] for j in range(len(mat))] for i in range(len(mat[0]))]

    def dot(self, mat1, mat2):
        result = [[0 for i in range(len(mat2[0]))] for j in range(len(mat1))]
        for i in range(len(mat1)):
            # iterate through columns of mat2
            for j in range(len(mat2[0])):
                # iterate through rows of mat2
                for k in range(len(mat2)):
                    result[i][j] += mat1[i][k] * mat2[k][j]
        return result

    def get_minor(self, m, i, j):
        return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]

    def get_determinant(self, m):
        # base case for 2x2 matrix
        if len(m) == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]

        determinant = 0
        for c in range(len(m)):
            determinant += ((-1) ** c) * m[0][c] * self.get_determinant(self.get_minor(m, 0, c))
        return determinant

    def inverse(self, m):
        determinant = self.get_determinant(m)
        # special case for 2x2 matrix:
        if len(m) == 2:
            return [[m[1][1] / determinant, -1 * m[0][1] / determinant],
                    [-1 * m[1][0] / determinant, m[0][0] / determinant]]

        # find matrix of co-factors
        cofactors = []
        for r in range(len(m)):
            cofactorRow = []
            for c in range(len(m)):
                minor = self.get_minor(m, r, c)
                cofactorRow.append(((-1) ** (r + c)) * self.get_determinant(minor))
            cofactors.append(cofactorRow)
        cofactors = self.transpose(cofactors)
        for r in range(len(cofactors)):
            for c in range(len(cofactors)):
                cofactors[r][c] = cofactors[r][c] / determinant
        return cofactors

    def fit(self, passes, split):
        __temp_beta = []
        slice_index = math.floor(len(self.__DATA) * split)
        for i in range(passes):
            random.shuffle(self.__DATA)
            train, test = self.__DATA[:slice_index], self.__DATA[slice_index:]
            A_train = [x[0] for x in train]
            Y_train = [[x[1]] for x in train]
            A_test = [x[0] for x in test]
            Y_test = [[x[1]] for x in test]

            AT = self.transpose(A_train)
            ATA_inv = self.inverse(self.dot(AT, A_train))
            ATY = self.dot(AT, Y_train)
            __temp_beta = self.dot(ATA_inv, ATY)
            __temp_beta = [i[0] for i in __temp_beta]
            self.beta = __temp_beta
            # print("Calculated co-efficients: {}".format(self.beta))
            print("Pass#: {} \t Train Records: {} \t Test Records: {}\nCalculated Beta vector: {}".format(i + 1,
                                                                                                          len(A_train),
                                                                                                          len(A_test),
                                                                                                          self.beta))
            self.predict(A_test, Y_test)

    def predict(self, A_test, Y_test):
        total = 0.0
        correct = 0
        error = 0.0
        for i in range(len(A_test)):
            for j in range(len(self.beta)):
                total = total + (A_test[i][j] * self.beta[j])
            result = round(total)
            if result == Y_test[i][0]:
                correct += 1
            print("\tPredicted: {}({:4.3f}), Actual: {}({:1.0f}), Error: {:4.3f}".format(self.labels[result], total,
                                                                                         self.labels[Y_test[i][0]],
                                                                                         Y_test[i][0],
                                                                                         total - Y_test[i][0]))
            error += (total - Y_test[i][0])
            total = 0.0
        print("Accuracy = {:6.3f} % \t Average Error: {:4.3f}\n\n".format(correct / len(Y_test) * 100,
                                                                          error / len(Y_test)))
