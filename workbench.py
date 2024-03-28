import matplotlib.pyplot as plt

x10, y10, x11, y11  = list(map(float, input().split(' '))
x20, y20, x21, y21  = list(map(float, input().split(' '))



plt.plot([x10,x11], [y10, y11])
plt.plot([x20,x21], [y20, y21])
plt.show()