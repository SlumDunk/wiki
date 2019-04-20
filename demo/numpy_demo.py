import numpy as np

array = np.array([[1, 2, 3], [2, 3, 4]])

print(array)
print("number of dim:", array.ndim)
print("shape:", array.shape)
print("size", array.size)

array2 = np.array([2, 23, 4], dtype=np.int)
print(array2.dtype)

array3 = np.zeros((3, 4))
print(array3)

array4 = np.ones((3, 4), dtype=float)
print(array4)

array5 = np.arange(10, 20, 2)
print(array5)

array6 = np.arange(12).reshape((3, 4))
print(array6)

array7 = np.linspace(1, 10, 20)
print(array7)

a = np.array([10, 20, 30, 40])
b = np.arange(4)

print(a, b)
c = a + b
print(c)

c = b ** 2
print(c)

c = 10 * np.sin(a)
print(c)

print(b)
print(b < 3)

a = np.array([[1, 1], [0, 1]])
b = np.arange(4).reshape((2, 2))
# 对应位置相乘
c = a * b
# 矩阵相乘
c_dot = np.dot(a, b)
c_dot_2 = a.dot(b)
print(c)
print(c_dot)
print(c_dot_2)

array8 = np.random.random((2, 4))
print(array8)
print(np.sum(array8))
print(np.min(array8))
print(np.max(array8))

print(np.sum(array8, axis=1))
print(np.min(array8, axis=0))
print(np.max(array8, axis=1))

array9 = np.arange(2, 14).reshape((3, 4))
print(np.argmin(array9))
print(np.argmax(array9))
print(np.mean(array9))
print(np.average(array9))
print(np.median(array9))
print(np.cumsum(array9))
print(np.diff(array9))
print(np.nonzero(array9))
# 逐行排序
print(np.sort(array9))
print(np.transpose(array9))
print(array9.T.dot(array9))

print(np.clip(array9, 5, 9))

# 索引
array10 = np.arange(3, 15)
print(array10)
print(array10[3])
array10.reshape((3, 4))
print(array10[2])
print(array10[1][1])
print(array10[1, 1])
print(array10[2, :])
print(array10[:, 1])
print(array10[1, 1:3])

for row in array10:
    print(row)

for col in array10.T:
    print(col)

print(array10.flatten())
for item in array10.flat:
    print(item)

# 合并
a = np.array([1, 1, 1])
b = np.array([2, 2, 2])
c = np.vstack((a, b))  # vertical stack
print(a.shape, c.shape)

d = np.hstack((a, b))
print(d)
print(d.shape)

print(a[:, np.newaxis])
print(a[np.newaxis, :].shape())

a = np.array([1, 1, 1])[:, np.newaxis]
b = np.array([2, 2, 2])[np.newaxis, :]

c = np.concatenate((a, b, b, a), axis=0)
print(c)

c = np.concatenate((a, b, b, a), axis=1)
print(c)

# array分割

a = np.arange(12).reshape((3, 4))
print(a)
# 按列分成两块 等份分割
print(np.split(a, 2, axis=1))
print(np.split(a, 3, axis=0))
# 支持不等份分割
print(np.array_split(a, 3, axis=1))
print(np.vsplit(a, 3))
print(np.hsplit(a, 2))

# array copy
a = np.arange(4)
print(a)
# 指向同块内存
b = a
a[0] = 11
print(b is a)
print(b)

b[1:3] = [22, 33]
print(a)

b = a.copy()  # deep copy
a[3] = 44
print(a)
print(b)
