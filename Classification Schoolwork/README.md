# Classification Schoolwork——高光谱分析

### 河北师范大学软件学院  2016级 机器学习方向  徐安  2016011434
#### Dec 3 2018

## 一、数据集读入

```python
# 读入数据
dataset = np.array([])
file_num = [2,3,5,6,8,10,11,12,14]
for i in file_num:
    dataFile = 'dataset/data'+str(i)+'_train.mat'
    data = scio.loadmat(dataFile)
    data = np.array(data['data'+str(i)+'_train'])
    label = np.full((len(data)),i)
    data = np.c_[data,label]
    dataset = np.append(dataset,data)

# 将一维数据转化为行列形式
dataset = dataset.reshape(-1,201)

# 将Numpy格式的数据数据转化为Dateframe格式
dataset = pd.DataFrame(dataset)
dataset.rename(columns={ dataset.columns[200]: "y" }, inplace=True) # rename不是传递对象
```

使用scipy自带包对.mat格式的数据集进行读入。读入之后，按照文件名分布添加类别标签，并将每个文件的数据集进行合并，成为一个整体的Dataframe。
## 二、相关度分析&特征提取

在相关度分析的过程中，使用每个类别特征对其他特征的相关度系数来描述每个特征的相关性，并使用heatmap来绘制其相关度矩阵。

部分分类的相关度分析heatmap如下：

![corr_1](https://github.com/m-L-0/18b-Xu_An-2016-434/Classification Schoolwork/img/corr_1.png)

![corr_2](https://github.com/m-L-0/18b-Xu_An-2016-434/Classification Schoolwork/img/corr_2.png)

![corr_3](https://github.com/m-L-0/18b-Xu_An-2016-434/Classification Schoolwork/img/corr_3.png)

经分析相关度的heatmap可知，几乎所有的特征都与相关特征呈强烈的正相关或负相关性，所以在特征选择阶段不考虑剔除特征。剔除特征会有丢失信息的情况发生，经试验表明，剔除特征会导致准确率下降。

## 三、数据集预处理

### （一）数据集标签划分

```python
dataset_label = dataset['y']
dataset = dataset.drop(['y'], axis=1)
```

读入数据后，由于在读取时添加了标签信息，为了方便对数据进行更好的预处理，需要将数据集合标签进行分离。

### （二）数据降维

经统计，本数据集共有**9个类别**，特征数（列数）为**200维**，共有**6924个**数据项。
由于样本的特征维数较多，我们考虑到模型的时间代价和模型的复杂程度，需要对其进行降维操作。

**常见的降维操作如下：**

#### 1. 主成分分析（Principal components analysis, PCA）

 在多元统计分析中，**主成分分析**是一种分析、简化数据集的技术。主成分分析经常用于减少数据集的维数，同时保持数据集中的对方差贡献最大的特征。这是通过保留低阶主成分，忽略高阶主成分做到的。这样低阶成分往往能够保留住数据的最重要方面。但是，这也不是一定的，要视具体应用而定。由于主成分分析依赖所给数据，所以数据的准确性对分析结果影响很大。

**在sklearn中使用主成分分析：**

```python
# PCA降维
from sklearn.decomposition import PCA
spca = PCA()
dataset = spca.fit_transform(dataset)
```

#### 2. 局部线性嵌入（Locally Linear Embedding, LLE）

局部线性嵌入（Locally Linear Embedding, LLE）是**无监督非线性降维**算法，是流型学习的一种。

LLE和Isomap一样试图在降维过程中保持高维空间中的流形结构。Isomap把任意两个样本点之间的测地距离作为流形结构的特征，而LLE认为局部关系刻画了流形结构。LLE认为，在高维中间中的任意一个样本点和它的邻居样本点近似位于一个超平面上，所以该样本点可以通过其邻居样本点的线性组合重构出来。

**在sklearn中使用局部线性嵌入：**

```python
# LLE降维
from sklearn.manifold import LocallyLinearEmbedding
LLE = LocallyLinearEmbedding()
dataset = LLE.fit_transform(dataset)
```

#### 3. 线性降维分析（Linear Discriminant Analysis, LDA）

Linear Discriminant Analysis(也有叫做Fisher Linear Discriminant)是一种**有监督的（supervised）**线性降维算法。与PCA保持数据信息不同，LDA是为了使得降维后的数据点尽可能地容易被区分。

假设原始数据表示为X，（m*n矩阵，m是维度，n是sample的数量）

既然是线性的，那么就是希望找到映射向量a， 使得 a‘X后的数据点能够保持以下两种性质：

1. **同类的数据点尽可能的接近（within class）**

2. **不同类的数据点尽可能的分开（between class）**

**在sklearn中使用局部线性嵌入：**

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA = LinearDiscriminantAnalysis()
dataset = LDA.fit_transform(dataset,dataset_label)
```

**在实际任务中，使用不同降维方法数据集进行处理，得到的结果如下：**


| 降维方法 | 验证集准确率 |
|:-------:|:--------:|
| 主成分分析（PCA） | 0.9025 |
| 局部线性嵌入（LLE） | 0.6223 |
| 线性判别分析（LDA） | 0.9069 |

*注：以上验证集准确率为使用LightGBM在未调参时得到的正确率，训练集与验证集划分比例为8：2*

经分析，使用LDA准确度较高，所以使用LDA对数据进行处理。

## 四、模型选择

本次分类任务为多类别分类任务，由于数据规模相对较小，所以尝试了不同模型对数据集进行分类。

经测试，模型对验证集的准确度分别如下：

| 使用分类器            | 验证集准确率       |
| :-----------------: | :----------------: |
| 核SVM (NuSVC)(nu=0.2) | 0.2823 |
| 随机森林（Random Forest） | 0.8600 |
| 梯度提升树（GradientBoosting） | 0.9025 |
| AdaBoost | 0.2729 |
| 多层感知机（MLP） | 0.4173 |
| xGBoost | 0.8745 |
| Catboost | N/A |
| LightGBM | 0.9069 |

*注：以上模型均使用默认参数对验证集进行预测*



## 五、参数优化

使用grid search对模型进行优化，调整参数以达到最优性能

```python
param_test1 = {'n_estimators':list(range(10,300,100))}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(
),param_grid = param_test1,scoring='accuracy',cv=2)
gsearch1.fit(dataset,dataset_label)
print(gsearch1.grid_scores_)      
print(gsearch1.best_params_)  
print(gsearch1.best_score_)

param_test2 = {'min_leaf_split':list(range(10,300,10))}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(
),param_grid = param_test2,scoring='accuracy',cv=2)
gsearch2.fit(dataset,dataset_label)
print(gsearch1.grid_scores_)      
print(gsearch1.best_params_)  
print(gsearch1.best_score_)
```



##六、结果预测

将模型进行训练后，对测试集进行结果预测：

```python
test = scio.loadmat('./dataset/data_test_final.mat')
test = np.array(test['data_test_final'])

# PCA降维
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA = LinearDiscriminantAnalysis()
test = LDA.fit_transform(test)

test_pre = clf.predict(test)
```

