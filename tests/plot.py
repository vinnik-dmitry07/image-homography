import matplotlib.pyplot as plt
shape = plt.imread('test.png').shape
benches = []
x = []
x_ticks = []
for i in range(1, 11):
    with open(f'test{i}x.txt', 'r') as f:
        benches.append(list(map(float, f.readlines())))
        x.append(shape[1] * i * shape[0] * i)
        if i < 3:
            x_ticks.append(f'{shape[1] * i * shape[0] * i}')
        else:
            x_ticks.append(f'{shape[1] * i} * {shape[0] * i}')
y = [sum(b) / len(b) for b in benches]

plt.plot(x, y)
plt.scatter(x, y)
plt.xticks(rotation=15)
plt.gca().set_xticks(x)
plt.gca().set_xticklabels(x_ticks)
plt.gca().set_yticks(y)
plt.gca().set_yticklabels([round(yi, 3) for yi in y])

plt.show()