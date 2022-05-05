import json
import os
import math
files=os.listdir('labelme')
'''
在同级目录下把labelme.rar解压，里面是数据
'''
W=[]
H=[]
S=[]
R=[]
Count=[]
# W=width, H=height, S=sqrt(width*height), R=width/height
Label=[0,0,0,0,0,0,0,0,0,0]
'''
all lables:
papule open_comedo nevus closed_comedo other melasma atrophic_scar nodule pustule hypertrophic_scar
'''
for skin in files:
	with open("labelme/"+skin) as f:
		print(skin)
		data=json.load(f)
		shapes=data["shapes"]
		Count.append(len(shapes))
		try:
			# (shapes[0]["points"])
			# the code above is a list of label points
			# the shapes[0]["points"][i] indicates a point of the label (in type list)
			# the [0] is x, the [1] is y           
			for j in range(0,len(shapes)): # in every area
				X=[]
				Y=[]
				label=shapes[j]['label']
				if(label=='papule'):
					Label[0]+=1
				elif(label=='open_comedo'):
					Label[1]+=1
				elif(label=='nevus'):
					Label[2]+=1
				elif(label=='closed_comedo'):
					Label[3]+=1
				elif(label=='other'):
					Label[4]+=1
				elif(label=='melasma'):
					Label[5]+=1
				elif(label=='atrophic_scar'):
					Label[6]+=1
				elif(label=='nodule'):
					Label[7]+=1
				elif(label=='pustule'):
					Label[8]+=1
				elif(label=='hypertrophic_scar'):
					Label[9]+=1
				
				for i in shapes[j]["points"]: # in every point
					X.append(i[0])
					Y.append(i[1])
				W.append(max(X)-min(X))
				H.append(max(Y)-min(Y))
				S.append(math.sqrt((max(X)-min(X))*(max(Y)-min(Y))))
				width=max(X)-min(X)
				height=max(Y)-min(Y)
				R.append(float(width*1.0/height))
		except:
			print("error")


print(Label)
# print(R)
import matplotlib.pyplot as plt
plt.figure(figsize=(100,600))
from matplotlib.font_manager import *
# set font
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.style.use('ggplot')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# change the label name for actual use
plt.xlabel("类别",fontproperties=font)
plt.ylabel("频数",fontproperties=font)
plt.title("类别分布",fontproperties=font)
# change the x range and y range for actual use
plt.xlim(0,100)
plt.ylim(0,1000)
import numpy as np
bins=np.arange(0,10,1)
# plot width height size ratio
# plt.hist(sorted(W),bins=bins)
all_labels=['丘疹', '开口粉刺', '痣', '闭口粉刺',
 '其他', '色斑', '萎缩疤', '结节', '脓包', '肥厚疤']
 # plot class
bb=plt.bar([0,1,2,3,4,5,6,7,8,9], Label, width=0.8, bottom=None, align='center')
k=0
for b in bb:
	h=b.get_height()
	plt.text(b.get_x()+b.get_width()/2,h+200,'%s'%str(all_labels[k]),ha='center',va='bottom',fontproperties=font)
	plt.text(b.get_x()+b.get_width()/2,h,'%d'%int(h),ha='center',va='bottom',fontproperties=font)
	k+=1
plt.show()
