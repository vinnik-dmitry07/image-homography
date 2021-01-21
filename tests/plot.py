import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

Y_TICKS_MIN_DELTA = 3.0  # magic number

shape = plt.imread('test.png').shape

benches_no = []
benches_omp = []
benches_mpi = []

x = []
x_ticks = []
for i in range(1, 11):
    x.append(shape[1] * i * shape[0] * i)
    x_ticks.append(f'{shape[1] * i} * {shape[0] * i}')

    with open(f'test{i}x_no.txt', 'r') as f:
        benches_no.append(list(map(float, f.readlines())))

    with open(f'test{i}x_omp.txt', 'r') as f:
        benches_omp.append(list(map(float, f.readlines())))

    with open(f'test{i}x_mpi.txt', 'r') as f:
        benches_mpi.append(list(map(float, f.readlines())))

y_no = [sum(b) / len(b) for b in benches_no]
y_omp = [sum(b) / len(b) for b in benches_omp]
y_mpi = [sum(b) / len(b) for b in benches_mpi]

y_ticks_ = sorted(y_no + y_omp + y_mpi)
y_ticks = [y_ticks_[0]]
for t in y_ticks_[1:]:
    if t - y_ticks[-1] > Y_TICKS_MIN_DELTA:      
        y_ticks.append(t)

plt.plot(x, y_no, label='No')
plt.plot(x, y_omp, label='OpenMP')
plt.plot(x, y_mpi, label='MPI')

plt.scatter(x, y_no)
plt.scatter(x, y_omp)
plt.scatter(x, y_mpi)

plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

ax = plt.gca()
ax.set_xticks(x)
ax.set_yticks(y_ticks)
ax.set_xticklabels(x_ticks)
ax.set_yticklabels([round(yi, 3) for yi in y_ticks])
ax.set_xlabel('Image size')
ax.set_ylabel('Time (sec)')

plt.legend()

plt.gca().grid()
plt.gcf().set_size_inches(8, 8)
plt.tight_layout()
plt.show()
