import numpy as np


if __name__ == '__main__':
    filename1 = "re.dat"
    filename2 = "l2.dat"
    fileout1 = open(filename1, 'w')
    fileout2 = open(filename2, 'w')
    log1 = "re.log"
    log2 = 'l2.log'
    result = []
    current_path = "."
    filein1 = open(log1, 'r')
    filein2 = open(log2, 'r')

    counter = 0
    length = 0
    values = np.array([])
    for line in filein1.readlines():
        counter += 1
        line = np.array(eval(line))
        length = len(line)
        values = np.append(values, line)
    values = values.reshape(counter, length)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    print("Mean: {}".format(list(mean)), file=fileout1)
    print("Std: {}".format(list(std)), file=fileout1)

    counter = 0
    length = 0
    values = np.array([])
    for line in filein2.readlines():
        counter += 1
        line = np.array(eval(line))
        length = len(line)
        values = np.append(values, line)
    values = values.reshape(counter, length)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    print("Mean: {}".format(list(mean)), file=fileout2)
    print("Std: {}".format(list(std)), file=fileout2)

    fileout1.close()
    fileout2.close()
    filein1.close()
    filein2.close()

    # Split Model
    # print("Mean: ", list(mean), file=fp)
    # print("Var: ", list(std), file=fp)
    # counter = 0
    # elist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # for e in elist:
    #     print("RR Epsilon = {}, Percent Mean = {}".format(e, mean[counter]),
    #           file=fp)
    #     counter += 1
    # counter = 0
    # for e in elist:
    #     print("RR Epsilon = {}, Percent Var = {}".format(e, std[counter]),
    #           file=fp)
    #     counter += 1
