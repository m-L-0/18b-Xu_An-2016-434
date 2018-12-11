# Tensorflow Schoolwork—— Iris Classification using KNN

### 河北师范大学软件学院  2016级 机器学习方向  徐安  2016011434

#### Nov 21 2018

## 一、基本思路

1. 对所有样本进行标准化处理

2. 指定参数K，确定距离度量方式

3. 对于每个测试样本，根据度量方式找到其k个近邻，根据等权投票规则对测试集进行类别划分
## 二、导入数据与数据集预处理

使用sklearn对iris数据进行导入：

```python
# 导入iris数据集
iris = load_iris()  # 加载数据集

features = iris.data # 获取属性数据(各种特征值)
labels = iris.target # 获取类别数据（花的分类）
target_names = iris.target_names # 获取数据的分类名称（花的分类名称）
```

按照8：2的比例随机划分训练集和测试集:

```python
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.2, random_state=1,shuffle=True)
```

对数据进行标准化预处理：

```python
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)
```



## 三、生成距离矩阵

生成测试集到训练集每个点之间的距离，并储存为一个矩阵。

```python
# 生成距离矩阵
dis_array = np.array([])
for x_test_index,x_test in enumerate(X_test):
    for x_train_index,x_train in enumerate(X_train):
        distance = tf.sqrt(tf.reduce_sum(tf.square(x_test-x_train))) # 计算测试集数据与每个训练集数据的欧式距离
        distance = tf.to_float(distance)
        dis_array = np.append(dis_array, distance)

dis_array = dis_array.reshape(30,120) # 改变距离矩阵形状，变成30行120列
```



## 四、执行KNN算法

根据上一步得到的距离矩阵，计算测试集到每个训练集的欧式距离，将距离从小到大依次排序，选择距离最近的前N个，统计N个中数量最多的那个类别作为测试集的类别。

```python
# 将距离矩阵从大到小进行排序,经过KNN决策后输出最终结果
dis_tensor = tf.constant(dis_array)

n = 29
samples_num =  len(X_test)

predict_label = np.array([])
for i in range(samples_num):
    top_k = tf.nn.top_k(-dis_tensor[i],n) # 选择n个近邻
    class_0_counter = 0
    class_1_counter = 0
    class_2_counter = 0
    for j in range(n):
        if y_train[top_k[1][j]] == 0:
            class_0_counter += 1
        elif y_train[top_k[1][j]] == 1:
            class_1_counter += 1
        else:
            class_2_counter += 1
        dict = {'class_0': class_0_counter, 'class_1': class_1_counter,'class_2':class_2_counter} # 初始化字典
        dict = sorted(dict.items(),key=lambda item:item[1]) # 将计数列表进行排序后，输出在k个近邻中相邻最多的那个类别
    if dict[2][0] == 'class_0':
        predict_label = np.append(predict_label,0)
    elif dict[2][0] == 'class_1':
        predict_label = np.append(predict_label,1)
    else:
        predict_label = np.append(predict_label,2)
```



## 五、模型评价

使用ACC对分类结果进行评价

```python
acc = accuracy_score(predict_label,y_test)
print(acc)
```



## 六、参数选择

使用暴力搜索法从0到训练集数目依次调整N，选择最优N进行预测

```python
dis_tensor = tf.constant(dis_array)
samples_num =  len(X_test)

acc_list = np.array([])
for n in range(120):
    predict_label = np.array([])
    for i in range(samples_num):
        top_k = tf.nn.top_k(-dis_tensor[i],n) # 选择n个近邻
        class_0_counter = 0
        class_1_counter = 0
        class_2_counter = 0
        for j in range(n):
            if y_train[top_k[1][j]] == 0:
                class_0_counter += 1
            elif y_train[top_k[1][j]] == 1:
                class_1_counter += 1
            else:
                class_2_counter += 1
            dict = {'class_0': class_0_counter, 'class_1': class_1_counter,'class_2':class_2_counter} # 初始化字典
            dict = sorted(dict.items(),key=lambda item:item[1])
        if dict[2][0] == 'class_0':
            predict_label = np.append(predict_label,0)
        elif dict[2][0] == 'class_1':
            predict_label = np.append(predict_label,1)
        else:
            predict_label = np.append(predict_label,2)
    acc = accuracy_score(predict_label,y_test)
    acc_list = np.append(acc_list,acc)
```

使用Matplotlib对搜索到的准确度进行绘制：

![img_1](https://github.com/m-L-0/18b-Xu_An-2016-434/Tensorflow_Schoolwork/img_1.png)

通过图像可知，当n=29时，分类性能最佳。