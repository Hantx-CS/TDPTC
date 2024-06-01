import os


if __name__ == '__main__':
    name1 = "re.log"
    name2 = "l2.log"
    f1 = open(name1, 'w')
    f2 = open(name2, 'w')
    for i in range(10, 300):
        sourceFilename = "result_" + str(i) + ".log"
        ft = open(sourceFilename, 'r')
        temp = []
        for line in ft:
            temp.append(line)
        ft.close()
        f1.write(temp[-1])
        f2.write(temp[-2])
    for i in range(10):
        sourceFilename = "result_0" + str(i) + ".log"
        ft = open(sourceFilename, 'r')
        temp = []
        for line in ft:
            temp.append(line)
        ft.close()
        f1.write(temp[-1])
        f2.write(temp[-2])
    f1.close()
    f2.close()







