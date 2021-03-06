import sys

import numpy as np
import scipy

from scipy.sparse import diags, linalg, spdiags
from scipy.sparse.linalg import eigsh
from scipy import linalg as lg
from random import randint

np.set_printoptions(linewidth=100)
import matplotlib.pyplot as plt

# mpt.use('TkAgg')
# plt.ion()
np.set_printoptions(suppress=True)

a = 1.1
b = 0.8

# a = 1.3
# b = 0.9

epsilon = 0.0001


h = 0.2

grid_size = int(1 / h)


def exact_function(x, y):
    # return np.cos(2 * x - y ** 2) + y * x
    return np.sin(y * y) + x * x * y


def f(x, y):
    # return -a * (-4 * np.cos(2 * x - y ** 2)) - b * (2 * np.sin(2 * x - y ** 2) - 4 * y ** 2 * np.cos(2 * x - y ** 2))
    return -a * 2 * y - b * 2 * (np.cos(y * y) - 2 * y *y * np.sin(y * y))


def createMatrix(grid_size):
    # creating D
    iter_row = np.zeros(grid_size - 2)
    iter_row[0] = 2 * (b + a) / (h * h)
    iter_row[1] = -1 * a / (h * h)
    iter_column = np.zeros(grid_size - 2)
    iter_column[0] = 2 * (b + a) / (h * h)
    iter_column[1] = -1 * a / (h * h)
    D = lg.toeplitz(iter_column, iter_row)

    # Setting D on the main diagonal
    temp = []
    for i in range((grid_size - 2)):
        temp.append(D)
    matrix = lg.block_diag(*temp)

    # Setting -1*a on upper and lower diagonals
    index = grid_size - 2
    for i in range((grid_size - 2) ** 2 - (grid_size - 2)):
        matrix[index][i] = -b / (h * h)
        matrix[i][index] = -b / (h * h)
        index = index + 1

    return matrix


def createB(h, grid_size, u):
    b_size = (grid_size - 2) ** 2
    B = np.zeros(b_size)

    f_index1 = 1
    f_index2 = 1

    for i in range(0, b_size):
        # print (f_index1, f_index2)
        c_1 = -a / (h * h)
        c_2 = -b / (h * h)

        B[i] = f(f_index1 * h, f_index2 * h)


        # adding boundary conditions
        if (f_index1 - 1) == 0:
            B[i] += -c_1 * u[f_index1 - 1, f_index2]


        if (f_index1 + 1) == grid_size - 1:
            B[i] += -c_1 * u[f_index1 + 1, f_index2]



        if (f_index2 - 1) == 0:
            B[i] += -c_2 * u[f_index1, f_index2 - 1]


        if (f_index2 + 1) == grid_size - 1:
            B[i] += -c_2 * u[f_index1, f_index2 + 1]

        # calculating f indexes
        f_index1 = f_index1 + 1
        if f_index1 == grid_size - 1:
            f_index1 = 1
            f_index2 = f_index2 + 1
            if f_index2 == grid_size - 1:
                f_index2 = 1

    return B


x = np.linspace(0, 1, grid_size + 1)
u = np.zeros((grid_size + 1, grid_size + 1))


# u[0] = exact_function(0, x[:])
#
# u[grid_size, :] = exact_function(grid_size, x[:])
#
# u[:, 0] = exact_function(x[:], 0)
#
# u[:, grid_size] = exact_function(x[:], grid_size)

for k in range(grid_size+1):
    u[k, 0] = exact_function(k*h, 0)
    u[k, grid_size] = exact_function(k*h, (grid_size)*h)
    u[0, k] = exact_function(0, k*h)
    u[grid_size, k] = exact_function((grid_size)*h, k*h)




# creating sparse matrix

main_diag = [2 * (a + b) / (h * h) for _ in range((grid_size - 1) ** 2)]
second_diag = [-1 * a / (h * h) for _ in range((grid_size - 1) ** 2 - 1)]
offset_diag = [-1 * b / (h * h) for _ in range((grid_size - 1) ** 2 - (grid_size - 1))]

count = 0
for i in range(len(second_diag)):
    if count == (grid_size - 2):
        second_diag[i] = 0
        count = 0
        continue
    count = count + 1

offset = (grid_size - 1) ** 2 - len(offset_diag)
A = diags([offset_diag, second_diag, main_diag, second_diag, offset_diag], [-offset, -1, 0, 1, offset])

# calculating largest eigenvalue
eigenVector = np.ones((grid_size - 1) * (grid_size - 1))

for i in range((grid_size - 1) * (grid_size - 1)):
    eigenVector[i] = randint(0, 10)

lambda_old = eigenVector.dot(A.dot(eigenVector))
lambda_new = max(abs(eigenVector))

error = 1
count = 0
while error > epsilon:

    eigenVector = A.dot(eigenVector)

    lambda_new = max(abs(eigenVector))

    eigenVector = eigenVector / lambda_new

    error = np.linalg.norm(A.dot(eigenVector) - eigenVector*lambda_new)/lambda_new
    lambda_old = lambda_new
    if (count % 10 == 0) & (h < 0.02):
        sys.stdout.write(f"\r abs: {error}")
        sys.stdout.flush()
    count = count + 1
if (h < 0.02):
    print()

print("largest eigenvalue is", lambda_new)
eigenSum = lambda_new


# calculating smallest eigenvalue
new_diag = [lambda_new for _ in range((grid_size - 1) ** 2)]

reverseA = diags(new_diag) - A

eigenVector = np.ones((grid_size - 1) * (grid_size - 1))

for i in range((grid_size - 1) * (grid_size - 1)):
    eigenVector[i] = randint(0, 10)

error = 100
lambda_old = 0
lambda_new = max(abs(eigenVector))
count = 0
while error > epsilon:
    eigenVector = reverseA.dot(eigenVector)

    lambda_new = max(abs(eigenVector))

    eigenVector = eigenVector / lambda_new

    error = np.linalg.norm(reverseA.dot(eigenVector) - eigenVector*lambda_new)/lambda_new

    lambda_old = lambda_new

    if (count % 10 == 0) & (h < 0.02):
        sys.stdout.write(f"\r abs: {error}")
        sys.stdout.flush()
    count = count + 1

if (h < 0.02):
    print()

print("smallest eigenvalue is", eigenSum - lambda_new)


# eigvals = sorted(lg.eigvalsh(A.todense()))
# print("largest true eigenvalue is " + str(eigvals[-1]))
# print("smallest true eigenvalue is " + str(eigvals[0]))

# print("largest true eigenvalue is " + str(max(np.linalg.eigvals(A.todense()))))
# print("smallest true eigenvalue is " + str(min(np.linalg.eigvals(A.todense()))))

# norm = np.linalg.norm(A.toarray(), np.inf)
# print("largest true eigenvalue is " + str(linalg.eigsh(A, k=1, return_eigenvectors=False)[0]))
# print("smallest true eigenvalue is " + str(norm - linalg.eigsh(norm * diags(np.ones((grid_size - 1) ** 2), 0) - A, k=1, return_eigenvectors=False)[0]))



eigenSum = eigenSum + eigenSum - lambda_new

# solving the system
B = createB(h, grid_size + 1, u)
# print(B)

calculated_u = np.ones(((grid_size - 1) ** 2))
previous_calculated_u = np.ones(((grid_size - 1) ** 2))

w = 2 / eigenSum
tau = B - A.dot(calculated_u)
error = 1
count = 0


while np.linalg.norm(tau) > epsilon:
    temp = calculated_u
    calculated_u = previous_calculated_u + w * (B - A.dot(previous_calculated_u))
    previous_calculated_u = temp

    tau = B - A.dot(calculated_u)
    if (count % 10 == 0) & (h < 0.02):
        sys.stdout.write(f"\r abs: {np.linalg.norm(tau)}")
        sys.stdout.flush()
    count = count + 1


temp = scipy.sparse.csr_matrix(A)
realSolution = scipy.sparse.linalg.spsolve(temp, B)


print("solution is")
print(calculated_u)

print("real solution is")
print(realSolution)


index = -1
tempIndex = 0
for i in range(1, grid_size):
    index += 1
    for j in range(1, grid_size):
        u[i][j] = calculated_u[index + tempIndex]

        tempIndex += grid_size-1
    tempIndex = 0


# creating calculated graph
xy = np.linspace(0, 1, grid_size+1)
X, T = np.meshgrid(xy, xy)

X_exact, T_exact = np.meshgrid(xy, xy)



CalculatedGraph = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, T, u, 30, cmap='binary')
ax.plot_surface(X, T, u, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.invert_xaxis()
ax.set_title('Calculated Solution')




# Creating exact graph
xy = np.linspace(0, 1, grid_size+1)
X, T = np.meshgrid(xy, xy)


# X_exact, T_exact = np.meshgrid(xy, xy)
U_exact = exact_function(T, X)

print(np.max(np.abs(u[:][:] - U_exact[:][:])))


ExactGraph = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X_exact, T_exact, U_exact, 30, cmap='binary')
ax.plot_surface(X_exact, T_exact, U_exact, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.invert_xaxis()
ax.set_title('Exact Solution')

plt.show()
