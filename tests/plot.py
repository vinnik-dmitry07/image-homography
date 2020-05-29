import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


shape = plt.imread('test.png').shape

benches_omp = []
benches_mpi = []

x = []
x_ticks = []
for i in range(1, 11):
    x.append(shape[1] * i * shape[0] * i)
    x_ticks.append(f'{shape[1] * i} * {shape[0] * i}')
        
    with open(f'test{i}x_omp.txt', 'r') as f:
        benches_omp.append(list(map(float, f.readlines())))
        
    with open(f'test{i}x_mpi.txt', 'r') as f:
        benches_mpi.append(list(map(float, f.readlines())))

y_omp = [sum(b) / len(b) for b in benches_omp]
y_mpi = [sum(b) / len(b) for b in benches_mpi]

y_ticks_ = sorted(y_omp + y_mpi)
y_ticks = [y_ticks_[0]]
for t in y_ticks_[1:]:
    if t - y_ticks[-1] > 0.3:      
        y_ticks.append(t)

line_omp, = plt.plot(x, y_omp)
line_mpi, = plt.plot(x, y_mpi)

line_omp.set_label('OpenMP')
line_mpi.set_label('MPI')

plt.scatter(x, y_omp)
plt.scatter(x, y_mpi)

plt.xticks(rotation=15)

ax = plt.gca()
ax.set_xticks(x)
ax.set_yticks(y_ticks)
ax.set_xticklabels(x_ticks)
ax.set_yticklabels([round(yi, 3) for yi in y_ticks])
ax.set_xlabel('Image size')
ax.set_ylabel('Time (sec)')

plt.legend()

plt.gca().grid()

plt.show()