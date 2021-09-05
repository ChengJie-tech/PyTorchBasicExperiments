# PyTorch Basic Experiments
--------
### 1: PyTorch Basic operations
pytorch框架与autograd入门，简单前向神经网络
#### 套路
1. 定义输入输出
2. 定义模型
3. 定义loss function
4. 定义optimizer
5. 定义训练过程：获得模型预测，做loss，清理梯度，反向传播，更新参数
### 2: Simple Word Embedding Experiment
Word2vec：Skip-Gram模型，基于分布式假设

假任务：用中心词的词向量预测附近词的词向量（这样我们就知道这个词是什么）

真任务：用上面训练出的参数获得词向量

目标函数：一个对数概率和，sigma(log(p(w_t+j | wt)))这样，其中概率密度由一个softmax给出：p(o | c)=exp(u_o^T·v_c)/sigma(exp(u_w^T·v_c))，这个点积越大概率就越大

缺陷：比如词向量空间维度有50w，那每次进来一个词向量都要和这50w维做一次点积！太惊人了。

eg. 有50000个词，input embedding:50000 * 100（即用一个100维的vec表征一个word）, output embedding:50000 * 100, 模型参数就是这些embeddings，是要学习的；在我们优化目标函数（假目标）的过程中，真参数这些embeddings就被学习出来了。

改进：负例采样（negative sampling）一个正样本，V-1个负样本（采样得到，子空间），优化的是正样本的点积和负样本的负点积的联合和。
