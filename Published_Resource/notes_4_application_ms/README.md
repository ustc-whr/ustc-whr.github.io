# Note

以下答案均为个人作答，很可能不正确，欢迎和我讨论

### 统计算法期中复习：

如何求数值积分：F(x)，其中F为N(0,1)的累积分布函数；同时让数值积分更稳定
$$
用Monte Carlo 方法，同时运用对偶变量法，控制变量法，重要抽样法，分层抽样法等进行方差缩减
\\
当然数值积分中的Gauss积分也可以使用
$$


### 数学分析期中复习：

𝑓(𝑥)可微，∀$a\leq𝑥_1 < 𝑥_2\leq b$ , $𝑓(𝑥_1 ) ≤ 𝑓(𝑥_2)$，且𝑓’(𝑥)仅在有限个点上为 0。求证：f(x)在[a,b]上严格递增。
$$
用积分证明
$$
$ f(x)=\left\{\begin{array}{cc}
x^2 \sin \frac{1}{x^2}, & x \neq 0 \\
0 & , x=0
\end{array}  \right.$在0处的连续，可微性
$$
显然连续，可导
$$
$ f(x)=\left\{\begin{array}{cc}
x \sin \frac{1}{x^2}, & x \neq 0 \\
0 & , x=0
\end{array}  \right.$​在0处的连续，可微性
$$
连续但不可导
$$
判断二元函数  $f(x, y)=\frac{x^2 y}{\sqrt{x^{4}+y^{2}}}$在  (0,0)  处是否连续、可导以及可微。
$$
连续性显然，可导是因为\frac{f(x,0)-f(0,0)}{x-0}=0,
不可微是因为:\frac{f(x,y)-f(0,0)}{\sqrt{x^2+y^2}}不趋向于0
$$
求极限  $\lim _{x \rightarrow a} \frac{sin{x}-sin{a}}{x-a}  $。
$$
洛必达或者泰勒展开
$$
$ \iiint_{x^{2}+y^{2}+z^{2} \leq 1} \frac{1/4}{x^{2}+y^{2}+z^{2}} d x d y d z  $
$$
换元  \left\{\begin{array}{c}x=\rho \sin \varphi \cos \theta \\ y=\rho \sin \varphi \sin \theta, \\ z=\rho \cos \varphi\end{array}\right.  I=1/4*\int_{0}^{1}\int_{0}^{\pi}{2\pi\rho^2 sin{\varphi}/\rho^2d\varphi}=2 \cdot 2 \pi \cdot 1/4= \pi
$$
求 $\int_{0}^{\infty} x^{2}\left(1+x^{2024}\right) e^{-x} d x  $。
$$
记对于x^k e^{-x}积分为F(k),则有F(k)=kF(k-1)
$$
求 $\int_{-k}^{k} x^{6}\left(1+x^{2024}\right) (e^x-e^{-x}) d x  $。
$$
奇函数，积分为0
$$


### 线性代数期中复习：

5.

$\left(\begin{array}{cc}
A & B \\
B^{T} & D
\end{array}\right)$正定, 求证:$\left|\begin{array}{cc}
A & B \\
B^{T} & D
\end{array}\right|\leq|A||D|$​
$$
\left|\begin{array}{cc}
A & B \\
B^{T} & D
\end{array}\right|=det(A)*det(D-B^TA^{-1}B)
\\
显然，A,D正定；于\tilde{A}=B^TA^{-1}B正定；于是D与\tilde{A}可以同时正交对角化
\\
于是det(D)>=det(D-\tilde{A});证毕
\\Lemma:
对于两个对称阵A，而言：若有AB=BA,或者A，B有一个是正定矩阵，则二者可以同时正交对角化
$$
$A=\beta\beta^t+\mu I$，分析A的特征值及其重数
$$
\beta\beta^t的非零特征值为\norm{\beta}^2,进而A的特征值为n-1重\mu,1重\mu+\norm{\beta}^2
$$
3维实对称矩阵的一个二重非零特征根为 $\lambda $, 对应两个特征向量为  $(0,-1,-1),(1,0,-1) $, 求另一特征根对应的特征向量
$$
根据对阵实对称阵的性质，可以知道第三个特征向量和不同特征值的特征向量是正交的，于是answer=(1,-1,1)
$$
A为实矩阵且 $ A^{k}=\mathbb{0} $,k为整数, 求A的特征值
$$
化零多项式x^2=0，于是A的特征值只有0
$$
$A^{k}=\mathbb{0}$ , $I_{n}+A  $是否可逆？
$$
显然可逆，其逆为I_n+A+A^2+...+A^{k-1}
$$
问：$ABx=0$,$BAy=0$是否有一定有解？期中$x\in R^n ;y\in R^m ;A\in R^{n*m};B\in R^{m*n} $,m>n
$$
前者不一定，后者一定有解
$$

### 概统期中复习：

一个均匀的骰子，投掷许多次 

(1) 6 次就集齐{1,…,6}的概率?     (2) 7 次就集齐{1,…,6}的概率?    (3) 8 次就集齐{1,…,6}的概率?    (4) 平均需几次能够集齐{1,…,6}?
$$
tips:X_i 定义为在出现i-1个数字之后，首次出现第i个新的数字所需要的次数，X_iiid几何分布Geo(1-(i-1)/6)
\\
1.
P_1=P(X_i=1,i=1,..,6)=1*\frac{5}{6}*\frac{4}{6}*\frac{3}{6}*\frac{2}{6}*\frac{1}{6}
\\
2.P_2=P(5个i,1个j,X_i=1,X_j=2)=\sum{\frac{i}{6}}*P_1
\\
3.P_3=P(4个i,2个j,X_i=1,X_j=2;或者5个i,1个j,X_i=1,X_j=3)=P_1*\sum_{i,j}{\frac{ij}{36}}+P_1*\sum_{i}{(\frac{i}{6})^2}
\\P_1*()
4.
E[\sum_{i=1}^{6}X_i]=\sum_{i=1}^{6}\frac{6}{i}
$$
在某城市，有两种颜色的汽车：红色和蓝色。红色汽车占所有汽车的 70%，蓝色汽车占 30%。老王的视力不是很好，他在观察到的情况下有 90% 的概率正确地辨别出红色汽车，但也有 20% 的概率会错误地将蓝色汽车看成红色。现在，老王说他看到了一辆红色的汽车。求在这种情况下，这辆汽车实际上是红色的概率。

$$
贝叶斯公式
$$
给出样本  $\left(x_{1}, x_{2}, x_{3} \ldots, x_{n}\right)$  的均值，方差无偏估计量
$$
\bar{X},S^2=\frac{\sum{(x_i-\bar{x})^2}}{n-1}
$$
$X,Y$~$N(0,1),Cov(X,Y)=\rho$求$aX+bY$和$bX-aY$的相关系数
$$
lemma:cov(Ax,Bx)=AVar(x)B^t
$$
一天的消耗量X为连续型随机变量，pdf为f(x)，初始剩余量为$\theta$；求一天后消耗完的概率；求一天后剩余量的期望；求消耗完需要的天数的期望
$$
1.P(X>=\theta)
\\
2.E[\theta-x]
\\
3.E[N|\sum_{i=1}^{N}X_i>=\theta]
$$


### 微分方程期中复习：

求解二阶常微分方程 $ \frac{\partial^{2} y}{\partial x^{2}}=e^{x} $   
$$
y=e^{x}+c_{1} x+c_{2}
$$

$$

$$
