#汇总 生成输入X数据
#手动保存 
#顺序 正方形（1-2500） 圆形（2500-5000） 三角形 （5000-7500）八边形 （7500-10000）五角星（10000-12500）

a = [1,0,0,0,0] #正方形
b = [0,1,0,0,0] #圆形
c = [0,0,1,0,0] #三角形
d = [0,0,0,1,0] #八边形
e = [0,0,0,0,1] #五角星

v = "Y_5种按压头"
matrix = [a,b,c,d,e]
for number in range(len(matrix)):
    file = open(("%s.txt" %v),'a')
    for i in range(2500):
        s = str(matrix[number]).replace('[','').replace(']','')
        file.write(s)
        file.write(',')
        file.write('\n')
    print("保存文件成功") 

file.close()