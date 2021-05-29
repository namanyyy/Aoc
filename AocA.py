import numpy as np
import pandas as pd
from efficient_apriori import apriori
#等宽度区间离散
def Do_labels(num,list1,str1):
    for i in range(12):
        if num<=list1[0]+i*list1[1]:
            return str1+str(i)     #对数据进行编码
def DiscreteData(list1,str1):
    data_max=max(list1)
    data_min=min(list1)
    interval_length=(data_max-data_min)/10   #等宽离散区间
    cur=[]
    cur.append(data_min)
    cur.append(interval_length)
    cur1=[]
    for i in list1:
        cur1.append(Do_labels(i,cur,str1))
    return cur1
####################################################################################

#加载数据，按属性区间离散化
f=np.loadtxt("D:\文件\pyproject\数据挖掘/prodata.txt",delimiter="    ",dtype=float)
data,labels=np.split(f,(6,),axis=1)
new_data=[]
#对每个属性进行区间离散
data_range=[x[0] for x in data]
data_max=[x[1] for x in data]
data_mean=[x[2] for x in data]
data_skew=[x[3] for x in data]
data_var=[x[4] for x in data]
data_median=[x[5] for x in data]
new_data.append(DiscreteData(data_range,"r"))   #传入不同的字符区分不同的属性值
new_data.append(DiscreteData(data_max,"m"))
new_data.append(DiscreteData(data_mean,"n"))
new_data.append(DiscreteData(data_skew,"s"))
new_data.append(DiscreteData(data_var,"v"))
new_data.append(DiscreteData(data_median,"e"))
####################################################################

#显示区间离散化结果
graph={}
feature_name=["range","max","mean","skew","var","median"]
j=0
for i in range(6):
    graph[feature_name[i]]=new_data[j]
    j+=1
print(pd.DataFrame(graph))   #显示编码后的结果。

#######################################################################
new_data1=[]
for i in range(600):
    cur=[x[i] for x in new_data]
    new_data1.append(cur)
#关联分析
itemsets,rules = apriori(new_data1,min_support =0.5,min_confidence = 0.5)
print('频繁项集:',itemsets)
print('关联规则条数:',len(rules))
print("关联规则",rules)


