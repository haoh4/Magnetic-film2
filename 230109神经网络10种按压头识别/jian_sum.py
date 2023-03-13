#将所有的数据 减去好后 分成 2mm 4mm 和 2+4mm 汇总保存到一起

import math
import numpy as np
# 将文件中的36个文件全部进行了delta 计算 得到delta值 或者 angle值



# save the delta value
def text_save2(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        if i in range(len(data)-1):
            s = s.replace("'",'')+','   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
        if i == len(data)-1:
            file.write(',')
        file.write('\n')
    
    file.write('\n')
    file.close()
    print("保存文件成功") 

B_all = []


B_all_square = []
B_all_triangle = []
B_all_star = []
B_all_eight = []



#圆形数据
B_all_circle = []
B_all_circle2mm = []
B_all_circle4mm = []

#圆形按压所有数据
for v in range (13,14): #表示有 从 1-25 25个txt
    b = v+100 #2mm数据
    c = v+200 #4mm数据
    # 读取
    dataset1 = []
    dataset2 = []
    dataset3 = []
    B_2mm = []
    B_4mm = []
    filepath1 = ('230109神经网络数据/八边形/%d.txt' %v)
    filepath2 = ('230109神经网络数据/八边形/%d.txt' %b)
    filepath3 = ('230109神经网络数据/八边形/%d.txt' %c)
    lines1 = open(filepath1, 'r')
    lines2 = open(filepath2, 'r')
    lines3 = open(filepath3, 'r')
    data1 = []
    data2 = []
    data3 = []
    for line in lines1: #没有按压 原始数据
        data1.append(list(line.strip('\n').split(',')))
    for i in range(0,len(data1)):
        for j in range(0,len(data1[0])):
            data1[i][j] = float(data1[i][j])
    dataset1.append(data1)

    for line in lines2: #按压2mm的按压数据
        data2.append(list(line.strip('\n').split(',')))
    for i in range(0,len(data2)):
        for j in range(0,len(data2[0])):
            data2[i][j] = float(data2[i][j])
    dataset2.append(data2)

    for line in lines3: #按压4mm的按压数据
        data3.append(list(line.strip('\n').split(',')))
    for i in range(0,len(data3)):
        for j in range(0,len(data3[0])):
            data3[i][j] = float(data3[i][j])
    dataset3.append(data3)

    print(dataset2)

    #计算 2mm
    for i in range(len(dataset1[0])):
        for j in range(len(dataset1[0][0])):
        
            B_2mm.append(round(dataset2[0][i][j] - dataset1[0][i][j],2)) # x值

        B_all_circle2mm.append(B_2mm)  
        B_2mm = []

    #计算 4mm
    for i in range(len(dataset1[0])):
        for j in range(len(dataset1[0][0])):
        
            B_4mm.append(round(dataset3[0][i][j] - dataset1[0][i][j],2)) # x值

        B_all_circle4mm.append(B_4mm)  
        B_4mm = []

B_all_circle.extend(B_all_circle2mm)
B_all_circle.extend(B_all_circle4mm) #将2mm 4mm的数据进行汇总

# 写入s
v = '第13格2mm汇总'
filewrite = open(("230109神经网络数据/八边形/%s.txt" %v),'w')
text_save2(("230109神经网络数据/八边形/%s.txt" %v), B_all_circle2mm) #写入delta 差值 

v = '第13格4mm汇总'
filewrite = open(("230109神经网络数据/八边形/%s.txt" %v),'w')
text_save2(("230109神经网络数据/八边形/%s.txt" %v), B_all_circle4mm) #写入delta 差值 

v = '第13格汇总'
filewrite = open(("230109神经网络数据/八边形/%s.txt" %v),'w')
text_save2(("230109神经网络数据/八边形/%s.txt" %v), B_all_circle) #写入delta 差值 



#圆形中空数据
B_all_circle_kong = []
B_all_circle2mm_kong = []
B_all_circle4mm_kong = []

#圆形中空按压所有数据
for v in range (13,14): #表示有 从 1-25 25个txt
    b = v+100 #2mm数据
    c = v+200 #4mm数据
    # 读取
    dataset1 = []
    dataset2 = []
    dataset3 = []
    B_2mm = []
    B_4mm = []
    filepath1 = ('230109神经网络数据/八边形(中空)/%d.txt' %v)
    filepath2 = ('230109神经网络数据/八边形(中空)/%d.txt' %b)
    filepath3 = ('230109神经网络数据/八边形(中空)/%d.txt' %c)
    lines1 = open(filepath1, 'r')
    lines2 = open(filepath2, 'r')
    lines3 = open(filepath3, 'r')
    data1 = []
    data2 = []
    data3 = []
    for line in lines1: #没有按压 原始数据
        data1.append(list(line.strip('\n').split(',')))
    for i in range(0,len(data1)):
        for j in range(0,len(data1[0])):
            data1[i][j] = float(data1[i][j])
    dataset1.append(data1)

    for line in lines2: #按压2mm的按压数据
        data2.append(list(line.strip('\n').split(',')))
    for i in range(0,len(data2)):
        for j in range(0,len(data2[0])):
            data2[i][j] = float(data2[i][j])
    dataset2.append(data2)

    for line in lines3: #按压4mm的按压数据
        data3.append(list(line.strip('\n').split(',')))
    for i in range(0,len(data3)):
        for j in range(0,len(data3[0])):
            data3[i][j] = float(data3[i][j])
    dataset3.append(data3)

    print(dataset2)

    #计算 2mm
    for i in range(len(dataset1[0])):
        for j in range(len(dataset1[0][0])):
        
            B_2mm.append(round(dataset2[0][i][j] - dataset1[0][i][j],2)) # x值

        B_all_circle2mm_kong.append(B_2mm)  
        B_2mm = []

    #计算 4mm
    for i in range(len(dataset1[0])):
        for j in range(len(dataset1[0][0])):
        
            B_4mm.append(round(dataset3[0][i][j] - dataset1[0][i][j],2)) # x值

        B_all_circle4mm_kong.append(B_4mm)  
        B_4mm = []

B_all_circle_kong.extend(B_all_circle2mm_kong)
B_all_circle_kong.extend(B_all_circle4mm_kong) #将2mm 4mm的数据进行汇总

# 写入s
v = '第13格2mm汇总'
filewrite = open(("230109神经网络数据/八边形(中空)/%s.txt" %v),'w')
text_save2(("230109神经网络数据/八边形(中空)/%s.txt" %v), B_all_circle2mm_kong) #写入delta 差值 

v = '第13格4mm汇总'
filewrite = open(("230109神经网络数据/八边形(中空)/%s.txt" %v),'w')
text_save2(("230109神经网络数据/八边形(中空)/%s.txt" %v), B_all_circle4mm_kong) #写入delta 差值 

v = '第13格汇总'
filewrite = open(("230109神经网络数据/八边形(中空)/%s.txt" %v),'w')
text_save2(("230109神经网络数据/八边形(中空)/%s.txt" %v), B_all_circle_kong) #写入delta 差值 



'''
#三角形按压所有数据
for v in range (1,25):
    b = v+100
    # 读取
    dataset1 = []
    dataset2 = []
    B = []

    filepath1 = ('221211 神经网络按压头识别gt/三角形/4mm/%d.txt' %v)
    filepath2 = ('221211 神经网络按压头识别gt/三角形/4mm/%d.txt' %b)
    lines1 = open(filepath1, 'r')
    lines2 = open(filepath2, 'r')
    data1 = []
    data2 = []
    for line in lines1:
        data1.append(list(line.strip('\n').split(',')))
    for i in range(0,len(data1)):
        for j in range(0,len(data1[0])):
            data1[i][j] = float(data1[i][j])
    dataset1.append(data1)

    for line in lines2:
        data2.append(list(line.strip('\n').split(',')))
    for i in range(0,len(data2)):
        for j in range(0,len(data2[0])):
            data2[i][j] = float(data2[i][j])
    dataset2.append(data2)
    #print(dataset1)
    #print(dataset2)

    print(dataset2)

    #计算
    for i in range(len(dataset1[0])):
        for j in range(len(dataset1[0][0])):
        
            B.append(round(dataset1[0][i][j] - dataset2[0][i][j],0)) # x值

        B_all_triangle.append(B)  
        B = []

    # 写入
v = '汇总'
filewrite = open(("221211 神经网络按压头识别gt/三角形/绝对值/%s.txt" %v),'w')
text_save2(("221211 神经网络按压头识别gt/三角形/绝对值/%s.txt" %v), B_all_triangle) #写入delta 差值 



#正方形按压所有数据
for v in range (1,25):
    b = v+100
    # 读取
    dataset1 = []
    dataset2 = []
    B = []

    filepath1 = ('221211 神经网络按压头识别gt/正方形/4mm/%d.txt' %v)
    filepath2 = ('221211 神经网络按压头识别gt/正方形/4mm/%d.txt' %b)
    lines1 = open(filepath1, 'r')
    lines2 = open(filepath2, 'r')
    data1 = []
    data2 = []
    for line in lines1:
        data1.append(list(line.strip('\n').split(',')))
    for i in range(0,len(data1)):
        for j in range(0,len(data1[0])):
            data1[i][j] = float(data1[i][j])
    dataset1.append(data1)

    for line in lines2:
        data2.append(list(line.strip('\n').split(',')))
    for i in range(0,len(data2)):
        for j in range(0,len(data2[0])):
            data2[i][j] = float(data2[i][j])
    dataset2.append(data2)
    #print(dataset1)
    #print(dataset2)

    print(dataset2)

    #计算
    for i in range(len(dataset1[0])):
        for j in range(len(dataset1[0][0])):
        
            B.append(round(dataset1[0][i][j] - dataset2[0][i][j],0)) # x值

        B_all_square.append(B)  
        B = []

    # 写入
v = '汇总'
filewrite = open(("221211 神经网络按压头识别gt/正方形/绝对值/%s.txt" %v),'w')
text_save2(("221211 神经网络按压头识别gt/正方形/绝对值/%s.txt" %v), B_all_square) #写入delta 差值 

B_all.extend(B_all_circle)
B_all.extend(B_all_square)
B_all.extend(B_all_triangle)

    # 写入
v = '汇总'
filewrite = open(("221211 神经网络按压头识别gt/%s.txt" %v),'w')
text_save2(("221211 神经网络按压头识别gt/%s.txt" %v), B_all) #写入delta 差值 
'''