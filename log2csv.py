import re
import numpy as np
import pandas as pd

fpath=r"path/to/log.txt"
csvpath=r"path/to/log.csv"
f=open(fpath, "r",encoding='utf-8')
t=f.read()
f.close()

pmap95 = re.compile('AP.*0.50:0.95.* all.*= .*')   #定义正则表达式
pmap5 = re.compile('AP.*0.50 .* all.*= .*')
ploss = re.compile('total_loss: .*\n.*save')
smap95 = pmap95.findall(t)        #进行匹配，找到所有满足条件的
smap5 = pmap5.findall(t)
sloss = ploss.findall(t)
vmap95=[float(s.split(' ')[-1])for s in smap95]
vmap5=[float(s.split(' ')[-1])for s in smap5]
vloss=[float(s.split(' ')[1][:-1])for s in sloss]

print(vmap95,vmap5,vloss)
df=pd.DataFrame({'epoch':[i for i in range(len(vmap5))],'mAP_0.5':vmap5,'mAP_0.5:0.95':vmap95,'total_loss':vloss})
df.to_csv(csvpath,index=False,sep=',')