import numpy as np
from matplotlib import pyplot as plt

# parameter setting
rate = 1
PROCESS_MODE = False
w = np.random.random(3) * 10
# w = np.array([8.834294, 1.84971803, 0.21815951])
x = np.array([[1,1,1,1], [0,0,1,1], [0,1,0,1]])
d = np.array([0,1,1,0])
y = np.array([0,0,0,0])                                                                 # change with files
print("w", w)

# learning process
errors = []
rollout = []
flag_do = True
while flag_do:
    # infer
    v = np.dot(w, x)
    for index, item in enumerate(v):
        if item > 0:
            y[index] = 1
        else:
            y[index] = 0

    # record
    e = d - y
    error = np.mean(np.abs(e))
    errors.append(error)
    rollout.append(w)

    # update
    w = w + rate * np.dot(x, e)
    flag_do = ((any(y != d) or (len(errors) < 7)) and not (len(errors) > 100))              # change as showing

print('e', errors)

fig1 = plt.figure(figsize=(10, 10))
plt.plot(list(range(len(errors))), errors)
plt.title("Error change graph ")
plt.xlabel('Number of iterations')
plt.ylabel("Error")
plt.show()

if not PROCESS_MODE:
    x2 = np.linspace(0, 1, 10)
    y2 = w[0] / -w[2] + (w[1] / -w[2]) * x2
    fig2 = plt.figure(figsize=(10, 10))
    plt.xlabel('x1')
    plt.ylabel("x2")
    plt.scatter([0, 1], [0, 1], marker='o', color='green', s=40, label='class 1')
    plt.scatter([0, 1], [1, 0], marker='x', color='red', s=40, label='class 0')  # change with files
    plt.plot(x2, y2, label='decision boundary')
    plt.legend(loc='best')
    plt.show()
else:
    for index in range(len(rollout)):
        w[0], w[1], w[2] = rollout[index]
        x2 = np.linspace(0, 1, 10)
        y2 = w[0] / -w[2] + (w[1] / -w[2]) * x2
        fig2 = plt.figure(figsize=(10, 10))
        plt.xlabel('x1')
        plt.ylabel("x2")
        plt.scatter([0, 1], [0, 1], marker='o', color='green', s=40, label='class 1')
        plt.scatter([0, 1], [1, 0], marker='x', color='red', s=40, label='class 0')           # change with files
        plt.plot(x2, y2, label='decision boundary')
        plt.legend(loc='best')
        plt.show()



fig3=plt.figure(figsize=(10,10))
rollout = np.asarray(rollout)
plt.plot(list(range(len(rollout))), rollout[:, 0], label='w0', color='tab:blue')
plt.plot(list(range(len(rollout))), rollout[:, 1], label='w1', color='tab:orange')
plt.plot(list(range(len(rollout))), rollout[:, 2], label='w2', color='tab:green')
plt.xlabel('iteration time')
plt.ylabel('weight')
plt.title('trajectory of weight')
plt.legend(loc='best')
plt.show()

print('We did it!')
