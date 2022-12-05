epoch 5, test 0.8586

# label representation stability
**研究问题**：多次训练的label representation正负两个label representation的距离是否足够稳定？。
| 距离种类      | 正负label 距离 mean std     |
| ----------- | ----------- | 
| euclidean      | 0.268(0.043)      |
| cosine    | 0.974(0.019)       |
| dot   | 0.592(0.054)       | 

#
**研究问题**：当相同数据进行两次training后，能否通过添加label representation regularizer的方法保证下游frozen classifier能够复用？

LR score: 0.8586,0.5929

| 次      | exp settings      | LR valid  |  LR (test) classifier | gap mean std| note|
| ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- |
| 第一次      | 原sage (epoch 3)      | 0.85   | 0.85 | -|
|     |        
| 第二次   | 原sage (epoch 5)       | 0.85 | 不稳定 |  0.1987(0.1693)
|    | label sage cosine (5<epoch<100)      | 1 |0.85 | 0.0000(0.0000)|coef 0.5
|    | label sage euclidean  (5<epoch<100)     | 1 |0.85| 0.0385(0.1462)|coef 0.5
