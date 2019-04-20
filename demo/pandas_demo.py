import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = pd.Series([1, 3, 5, np.nan, 44, 1])
print(s)

dates = pd.date_range('20190412', periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
print(df)

df1 = pd.DataFrame(np.arange(12).reshape(3, 4))
print(df1)

df2 = pd.DataFrame({
    'A': 1,
    'B': pd.Timestamp('20190412'),
    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
    'D': np.array([3] * 4, dtype='int32'),
    'E': pd.Categorical(['test', 'train', 'test', 'train']),
    'F': 'foo'
})
print(df2)
print(df2.dtypes)

print(df2.index)
print(df2.columns)
print(df2.values)
print(df2.describe())
print(df2.T)

print(df2.sort_index(axis=1, ascending=False))
print(df2.sort_values(by='E'))

# 选择数据
df3 = pd.DataFrame(np.arange(24).reshape(6, 4), index=dates, columns=['A', 'B', 'C', 'D'])
print(df3['A'])
print(df3.A)
print(df3[0:3])
print(df3['20190402':'20190405'])
# select by label
print(df3.loc['20190412'])
print(df3.loc[:, ['A', 'B']])

print(df3.loc['20190412', ['A', 'B']])

# select by position, iloc
print(df3.iloc[3, 1])
print(df3.iloc[3:5, 1:3])

# mixed selection: ix
print(df3.ix[:3, ['A', 'C']])

# boolean indexing
print(df3[df3.A > 8])

# 设置值
df3[2, 2] = 1111
print(df3)
df3.loc['20190412', 'B'] = 2222
print(df3)
df3[df3.A > 4] = 0
df3.A[df3.A > 4] = 0
print(df3)

df3['F'] = np.nan
df3['E'] = pd.Series([1, 2, 3, 4, 5, 6], index=dates)
print(df3)

# 处理丢失数据
df4 = pd.DataFrame(np.arange(24).reshape(6, 4), index=dates, columns=['A', 'B', 'C', 'D'])
df4.iloc[0, 1] = np.nan
df4.iloc[1, 2] = np.nan
print(df4.dropna(axis=0, how='any'))
print(np.any(df.isnull()) == True)
print(df4.fillna(value=0))

# 导入导出
df4.to_pickle('helloworld.pickle')

# 数据合并 concatenating
df1 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.ones((3, 4)) * 2, columns=['a', 'b', 'c', 'd'])
df3 = pd.DataFrame(np.ones((3, 4)) * 3, columns=['a', 'b', 'c', 'd'])
print(df1)
print(df2)
print(df3)

res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
print(res)

# join ['inner','outer']
df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])
print(df1)
print(df2)

#      a    b    c    d    e
# 1  0.0  0.0  0.0  0.0  NaN
# 2  0.0  0.0  0.0  0.0  NaN
# 3  0.0  0.0  0.0  0.0  NaN
# 2  NaN  1.0  1.0  1.0  1.0
# 3  NaN  1.0  1.0  1.0  1.0
# 4  NaN  1.0  1.0  1.0  1.0
res = pd.concat([df1, df2], join='outer')
print(res)

#      b    c    d
# 0  0.0  0.0  0.0
# 1  0.0  0.0  0.0
# 2  0.0  0.0  0.0
# 3  1.0  1.0  1.0
# 4  1.0  1.0  1.0
# 5  1.0  1.0  1.0
res = pd.concat([df1, df2], join='inner', ignore_index=True)
print(res)

# join axes
#      a    b    c    d    b    c    d    e
# 1  0.0  0.0  0.0  0.0  NaN  NaN  NaN  NaN
# 2  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
# 3  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
res = pd.concat([df1, df2], axis=1, join_axes=[df1.index])
print(res)

# append
#      a    b    c    d    e
# 0  0.0  0.0  0.0  0.0  NaN
# 1  0.0  0.0  0.0  0.0  NaN
# 2  0.0  0.0  0.0  0.0  NaN
# 3  NaN  1.0  1.0  1.0  1.0
# 4  NaN  1.0  1.0  1.0  1.0
# 5  NaN  1.0  1.0  1.0  1.0
res = df1.append(df2, ignore_index=True)
print(res)

#      a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 3  1.0  2.0  3.0  4.0
s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
res = df1.append(s1, ignore_index=True)
print(res)

# merge
left = pd.DataFrame({
    'key': ['k0', 'k1', 'k2', 'k3'],
    'A': ['a0', 'a1', 'a2', 'a3'],
    'B': ['b0', 'b1', 'b2', 'b3']
})

right = pd.DataFrame({
    'key': ['k0', 'k1', 'k2', 'k3'],
    'C': ['c0', 'c1', 'c2', 'c3'],
    'D': ['d0', 'd1', 'd2', 'd3']
})

print(left)
print(right)
res = pd.merge(left, right, on='key')
print(res)

# consider two keys
left = pd.DataFrame({
    'key1': ['k0', 'k0', 'k1', 'k2'],
    'key2': ['k0', 'k1', 'k0', 'k1'],
    'A': ['a0', 'a1', 'a2', 'a3'],
    'B': ['b0', 'b1', 'b2', 'b3']
})

right = pd.DataFrame({
    'key1': ['k0', 'k1', 'k1', 'k2'],
    'key2': ['k0', 'k0', 'k0', 'k0'],
    'C': ['c0', 'c1', 'c2', 'c3'],
    'D': ['d0', 'd1', 'd2', 'd3']
})

print(left)
print(right)
# how=['left','right','outer','inner']
res = pd.merge(left, right, on=['key1', 'key2'], how='inner')
print(res)

# indicator
df1 = pd.DataFrame({
    'col1': [0, 1],
    'col_left': ['a', 'b']
})

df2 = pd.DataFrame({
    'col1': [1, 2, 2],
    'col_right': [2, 2, 2]
})

print(df1)
print(df2)
res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
res = pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_name')
print(res)

df1 = pd.DataFrame({
    'a': ['a0', 'a1', 'a2'],
    'b': ['b0', 'b1', 'b2'],
}, index=['k0', 'k1', 'k2'])

df2 = pd.DataFrame({
    'c': ['c1', 'c2', 'c3'],
    'd': ['d1', 'd2', 'd3']
}, index=['k0', 'k2', 'k3'])

print(df1)
print(df2)
res = pd.merge(df1, df2, left_index=True, right_index=True, how='outer')
print(res)

boys = pd.DataFrame({'k': ['k0', 'k1', 'k2'], 'age': [1, 2, 3]})

girls = pd.DataFrame({'k': ['k0', 'k0', 'k3'], 'age': [4, 5, 6]})
print(boys)
print(girls)
res = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='inner')
print(res)

# plot
data = pd.Series(np.random.randn(1000), index=np.arange(1000))
data = data.cumsum()
data.plot()
plt.show()

data = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000), columns=list("ABCD"))
data = data.cumsum()
print(data.head(5))
data.plot()
plt.show()

# plot methods: 'bar', 'scatter', 'hist', 'box', 'area', 'hexbin', 'pie'
ax = data.plot.scatter(x='A', y='B', color='blue', label='class1')
data.plot.scatter(x='A', y='C', color='orange', label='class2', ax=ax)
plt.show()
