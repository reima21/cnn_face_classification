import random
import sys

file=sys.argv[1]

a=[]
 #ファイル読み込み
for line in open(file,'r'):
        a.append(line.strip("\n"))

#シャッフル
random.shuffle(a)
#出力
for str2 in a:
        print (str2)
