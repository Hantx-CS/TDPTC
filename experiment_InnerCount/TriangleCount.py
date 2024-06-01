import datetime
import random
import sys
import numpy as np
import networkx as nx
import time
import math
import itertools

ERROR_FILE = "error.log"
LOG_FILE = "run.log"
RESULT_FILE = "result.log"
CANDIDATE_LENGTH = []


class Timer:
    def __init__(self, name="This action"):
        self.name = name
        self.begin = 0
        self.end = 0

    def setup(self, name=None):
        if name is not None:
            self.name = name
        self.begin = time.time()

    def finish(self, name=None):
        if name is not None:
            self.name = name
        self.end = time.time()

    def printNow(self, name=None):
        if name is not None:
            self.name = name
        costTime = time.time() - self.begin
        print("{} costs time is {} sec, {} min, {} hour".
              format(self.name, costTime, costTime / 60, costTime / 3600))

    def printLog(self, name=None):
        if name is not None:
            self.name = name
        costTime = self.end - self.begin
        print("{} costs time is {} sec, {} min, {} hour".
              format(self.name, costTime, costTime / 60, costTime / 3600))

    def run(self, name=None):
        self.finish(name)
        self.printNow(name)


def printLine(source):
    for i in range(0, source):
        print(i, source[i])


def printLine2(source):
    for i in range(0, len(source)):
        print(i, len(source[i]), source[i])


def printLine2Count(source):
    for i in range(0, len(source)):
        print(i, len(source[i]))


def printLine3(source):
    for i in range(0, len(source)):
        for j in range(0, len(source[i])):
            print((i, j), len(source[i][j]), source[i][j])


def printLine3Count(source):
    for i in range(0, len(source)):
        for j in range(0, len(source[i])):
            print((i, j), len(source[i][j]))


def listSet2DFlatList(list1):
    result = []
    for subSet in list1:
        result.extend(list(subSet))
    return result


def listSet2DList(list1):
    result = []
    for subSet in list1:
        result.append(list(subSet))
    return result


def listSet2DFlatSet(list1):
    result = set()
    for subSet in list1:
        result.update(subSet)
    return result


def isShareSameEdge(twoStar1, twoStar2):
    edge1, edge2 = twoStar1
    edge3, edge4 = twoStar2
    if set(edge1) == set(edge3):
        return tuple(reversed(edge2)), tuple(reversed(edge4)), edge1, edge3
    elif set(edge1) == set(edge4):
        return tuple(reversed(edge2)), tuple(reversed(edge3)), edge1, edge4
    elif set(edge2) == set(edge3):
        return tuple(reversed(edge1)), tuple(reversed(edge4)), edge2, edge3
    elif set(edge2) == set(edge4):
        return tuple(reversed(edge1)), tuple(reversed(edge3)), edge2, edge4
    return None


def isTrueTriangle(edge1, edge2, edge3):
    nodes = set()
    nodes = nodes.union(edge1)
    nodes = nodes.union(edge2)
    nodes = nodes.union(edge3)
    if len(nodes) == 3:
        return True
    if len(nodes) < 3:
        print("ERROR of isTrueTriangle: {}".format(len(nodes)), file=ERROR_FILE)
    return False


def discreteLaplace(epsilon, delta):
    alpha = math.exp(-epsilon / delta)
    r = random.random()
    if r < (1 - alpha) / (1 + alpha):
        return 0
    else:
        r = random.random()
        res = np.random.geometric(1 - alpha)
        if r < 0.5:
            return res
        else:
            return -res


def EM(epsilon, delta, group):
    weights = []
    p1 = math.exp(epsilon * 1 / (2 * delta))
    p2 = math.exp(epsilon * (-1) / (2 * delta))
    for i in range(0, len(group)):
        star = group[i]
        if star[0][0] == star[1][0]:
            weights.append(p1)
        else:
            weights.append(p2)
    return random.choices(group, weights=weights)[0]


def RR(epsilon, flag):
    p = math.exp(epsilon)
    p = p / (1 + p)
    r = random.random()
    if r > p:
        flag = not flag
    return flag


def splitGroups(twoStars, groupSize=None, groupNumber=None):
    number = len(twoStars)
    remainder = 10
    groups = []
    if groupNumber is None:
        remainder = number % groupSize
        groupNumber = int(number / groupSize)
    elif groupSize is None:
        remainder = number % groupNumber
        groupSize = math.floor(number / groupNumber)
    for i in range(groupNumber):
        if i < remainder:
            groups.append(twoStars[i * (groupSize + 1): (i + 1) * (groupSize + 1)])
        else:
            groups.append(twoStars[i * groupSize + remainder: (i + 1) * groupSize + remainder])
    return groups


class PrivacyBudget:
    def __init__(self):
        self.dLap = 10
        self.EM = 10
        self.RR = 10
        self.e1 = 10
        self.e2 = 10
        self.e3 = 10

    def setDLap(self, epsilon):
        self.dLap = epsilon

    def setEM(self, epsilon):
        self.EM = epsilon

    def setRR(self, epsilon):
        self.RR = epsilon

    def setALL(self, e1, e2, e3):
        self.setDLap(e1)
        self.setEM(e2)
        self.setRR(e3)
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3


class Sensitivity:
    def __init__(self):
        self.dLap = 1
        self.EM = 2

    def setDLap(self, sensitivity):
        self.dLap = sensitivity

    def setEM(self, sensitivity):
        self.EM = sensitivity

    def setALL(self, dLap, em):
        self.setDLap(dLap)
        self.setEM(em)


class GlobalController:
    def __init__(self, filename):
        self.G = nx.read_edgelist(filename)
        self.communities = []
        self.boundary_edge = []  # boundary_edge
        self.boundary_node = []
        self.subG = []
        self.triangle1, self.triangle2, self.triangle3 = dict(), dict(), dict()
        self.triangle = dict()
        self.triangleSum1, self.triangleSum2, self.triangleSum3 = 0, 0, 0

        self.Partition(2)
        # self.RandomPartition(2, 0.5)
        print("{}: Len of boundary edge is {},{},{},{}".format(
            datetime.datetime.now(), len(self.boundary_edge[0][0]), len(self.boundary_edge[0][1]),
            len(self.boundary_edge[1][0]), len(self.boundary_edge[1][1])))
        self.triangleCount()
        # print(max(nx.degree(self.G)))

    def Partition(self, maxNum=3):
        louvain = nx.community.louvain_communities(self.G)
        for i in range(0, len(louvain)):
            self.communities.append(list(louvain[i]))
            # print(i, len(self.communities[i]), self.communities[i])

        while len(self.communities) > maxNum:
            self.communities.sort(key=len, reverse=True)
            temp = self.communities[maxNum]
            del self.communities[maxNum]
            self.communities[maxNum - 1].extend(temp)
        random.shuffle(self.communities)
        # printLine2(self.communities)

        for i in range(0, len(self.communities)):
            self.subG.append(nx.subgraph(self.G, self.communities[i]))

        for i in range(0, len(self.communities)):
            self.boundary_edge.append([])
            self.boundary_node.append([])
            for j in range(0, len(self.communities)):
                if i == j:
                    self.boundary_edge[i].append(set())
                    self.boundary_node[i].append(set())
                    continue
                self.boundary_edge[i].append(
                    set(nx.edge_boundary(self.G, nx.nodes(self.subG[i]), nx.nodes(self.subG[j]))))
                self.boundary_node[i].append(
                    set(nx.node_boundary(self.G, nx.nodes(self.subG[i]), nx.nodes(self.subG[j]))))
        # printLine3Count(self.boundary)
        self.subG.append(nx.Graph(self.boundary_edge[0][1]))

    def RandomPartition(self, maxNum=3, fraction=0.5):
        print("fraction = {}".format(fraction))
        nodes1, nodes2 = [], []
        for node in nx.nodes(self.G):
            r = random.random()
            if r < fraction:
                nodes1.append(node)
            else:
                nodes2 .append(node)
        self.communities.append(nodes1)
        self.communities.append(nodes2)
        self.subG.append(nx.subgraph(self.G, nodes1))
        self.subG.append(nx.subgraph(self.G, nodes2))
        for i in range(0, len(self.communities)):
            self.boundary_edge.append([])
            self.boundary_node.append([])
            for j in range(0, len(self.communities)):
                if i == j:
                    self.boundary_edge[i].append(set())
                    self.boundary_node[i].append(set())
                    continue
                self.boundary_edge[i].append(
                    set(nx.edge_boundary(self.G, nx.nodes(self.subG[i]), nx.nodes(self.subG[j]))))
                self.boundary_node[i].append(
                    set(nx.node_boundary(self.G, nx.nodes(self.subG[i]), nx.nodes(self.subG[j]))))
        self.subG.append(nx.Graph(self.boundary_edge[0][1]))

    def triangleCount1(self, i):
        G = self.subG[i]
        result = sum(nx.triangles(G).values()) / 3
        self.triangle1[i] = result
        return result

    def triangleCount2(self, i, j):
        G = nx.subgraph(self.G, self.boundary_node[i][j])
        count = sum(nx.triangles(G).values()) / 3
        G = nx.Graph(list(nx.edges(G)) + list(self.boundary_edge[i][j]))
        result = sum(nx.triangles(G).values()) / 3 - count
        self.triangle2[i][j] = result
        # G = nx.subgraph(self.G, self.communities[i] + self.communities[j])
        # result = sum(nx.triangles(G).values()) / 3 - self.triangle1[i] - self.triangle1[j]
        # self.triangle21[i][j] = result
        return result

    def triangleCount3(self, i, j, k):
        G = nx.subgraph(self.G, self.communities[i] + self.communities[j] + self.communities[k])
        result = (sum(nx.triangles(G).values()) / 3
                  - self.triangle1[i] - self.triangle1[j] - self.triangle1[k]
                  - self.triangle2[i][j] - self.triangle2[i][k] - self.triangle2[j][k]
                  - self.triangle2[j][i] - self.triangle2[k][i] - self.triangle2[k][j])
        self.triangle3[i][j][k] = result
        return result

    def triangleCount(self):
        for i in range(len(self.communities)):
            self.triangleSum1 += self.triangleCount1(i)

        for i in range(len(self.communities)):
            self.triangle2[i] = dict()
            for j in range(len(self.communities)):
                self.triangleSum2 += self.triangleCount2(i, j)

        for i in range(len(self.communities)):
            self.triangle3[i] = dict()
            for j in range(i + 1, len(self.communities)):
                self.triangle3[i][j] = dict()
                for k in range(j + 1, len(self.communities)):
                    self.triangleSum3 += self.triangleCount3(i, j, k)

        print("All triangle of G is {}, one party is {}, two parties is {}, three parties is {}".format(
            sum(nx.triangles(self.G).values()) / 3, self.triangleSum1, self.triangleSum2, self.triangleSum3
        ), file=LOG_FILE)


class Aggregator:
    def __init__(self):
        self.triangle1 = dict()
        self.triangle2 = dict()
        self.triangle3 = dict()

    def TriangleCount1(self, number, count):
        if number in self.triangle1.keys():
            print("Error in TriangleCount1, Party Id = {}, Count = {}".format(number, count))
            print(datetime.datetime.now(),
                  "Error in TriangleCount1, Party Id = {}, Count = {}".format(number, count), file=ERROR_FILE)
        self.triangle1[number] = count

    def TriangleCount2(self, number, count):
        if number in self.triangle2.keys():
            print("Error in TriangleCount2, Party Id = {}, Count = {}".format(number, count))
            print(datetime.datetime.now(),
                  "Error in TriangleCount2, Party Id = {}, Count = {}".format(number, count), file=ERROR_FILE)
        self.triangle2[number] = count

    def TriangleCount3(self, number, count):
        if number in self.triangle3.keys():
            print("Error in TriangleCount3, Party Id = {}, Count = {}".format(number, count))
            print(datetime.datetime.now(),
                  "Error in TriangleCount3, Party Id = {}, Count = {}".format(number, count), file=ERROR_FILE)
        self.triangle3[number] = count

    def printResult(self):
        t1, t2, t3 = sum(self.triangle1.values()), sum(self.triangle2.values()), sum(self.triangle3.values())
        print("Triangle Count of 1 is {}, 2 is {}, 3 is {}, all is {}".format(t1, t2, t3, t1 + t2 + t3))


class Party:
    edgeId = 0
    errorCounter = 0
    trueCounter = 0

    def __init__(self, partyId, partyNumber, globalG, G, outE, privacyBudget, sensitivity,
                 groupNumber, dMax, groupFraction):
        self.groupFraction = groupFraction
        self.globalG = globalG
        self.groupNumber2, self.fraction3 = groupNumber[0], groupNumber[1]
        self.id = partyId
        self.number = partyNumber
        self.G = G
        self.outE = outE
        self.outEList = listSet2DList(outE)
        self.outEFlat = listSet2DFlatList(outE)
        self.pb = privacyBudget
        self.ss = sensitivity
        self.dMax = dMax
        self.twoStars2, self.twoStars3 = [], []
        self.twoStars2Flat, self.twoStars3Flat = set(), set()
        self.sendRecord3 = set()
        self.receivedMessages2, self.receivedMessages3 = [], []
        self.noise2 = 0
        self.distinct2 = 0
        self.noise3 = dict()
        self.chosenEdge3, self.sendEdge3, self.receivedEdge3 = [], [], []
        for i in range(self.number):
            self.chosenEdge3.append([])
            self.sendEdge3.append([])

    def getParty(self, edge):
        for i in range(len(self.outE)):
            if edge in self.outE[i]:
                return i
        return None

    def triangleCount1(self):
        counter = sum(nx.triangles(self.G).values()) / 3
        return counter + discreteLaplace(self.pb.e2, self.dMax[0] - 2)

    def findTwoStars2(self):
        print("Find{}".format(datetime.datetime.now()))
        for i in range(len(self.outE)):
            self.twoStars2.append([])
            elist = self.outE[i]
            remain_elist = elist.copy()
            # print("Elist type is {}, {}".format(type(elist), type(remain_elist)))
            for e1 in elist:
                remain_elist.remove(e1)
                # print(list(nx.neighbors(self.globalG, e1[0])))
                for n2 in list(nx.neighbors(self.globalG, e1[0])):
                    e2 = (e1[0], n2)
                    if e2 in remain_elist:
                        self.twoStars2[i].append((e1, e2))
                # for e2 in remain_elist:
                #     if e1[0] == e2[0]:
                #         self.twoStars2[i].append((e1, e2))
        self.twoStars2Flat = listSet2DFlatSet(self.twoStars2)
        starNum = []
        for twoStar in self.twoStars2:
            starNum.append(len(twoStar))
        print("{}: Two stars for type2 num is: {}, all is {}".format(
            datetime.datetime.now(), starNum, len(self.twoStars2Flat)))
        print("{}: Two stars for type2 num is: {}, all is {}".format(
            datetime.datetime.now(), starNum, len(self.twoStars2Flat)), file=LOG_FILE)
        # print(self.twoStars2Flat)
        # print(len(self.twoStars2Flat))
        # for i in range(len(self.twoStars2)):
        #     print("i: ", i, len(self.twoStars2[i]))

    def findFakeTwoStars2(self, number):
        # counter = 0
        # result = []
        # while counter < 1:
        #     e1, e2 = random.sample(self.outEFlat, 2)
        #     if self.getParty(e1) is None or self.getParty(e2) is None or self.getParty(e1) != self.getParty(e2):
        #         continue
        #     if (e1, e2) in self.twoStars2Flat or (e2, e1) in self.twoStars2Flat:
        #         continue
        #     while counter < number:
        #         result.append((e1, e2))
        #         counter += 1
        # return result
        counter = 0
        result = []
        while counter < number:
            e1, e2 = random.sample(self.outEFlat, 2)
            if self.getParty(e1) != self.getParty(e2):
                continue
            if (e1, e2) in self.twoStars2Flat or (e2, e1) in self.twoStars2Flat:
                continue
            result.append((e1, e2))
            counter += 1
        return result

    # InterCount
    # def query2(self):
    #     result = []
    #     for i in range(len(self.outE)):
    #         result.append([])
    #     self.findTwoStars2()
    #     candidatesT = list(self.twoStars2Flat)
    #     self.noise2 = discreteLaplace(self.pb.e3 / 2, self.dMax[1] - 1)
    #     # number = len(self.twoStars2Flat) + self.noise2
    #     # random.shuffle(self.twoStars2Flat)
    #     candidatesF = self.findFakeTwoStars2(len(candidatesT))
    #     candidates = []
    #     candidateSize = len(candidatesT) + self.noise2
    #     indexListT = np.random.choice(range(len(candidatesT)), candidateSize)
    #     indexListF = np.random.choice(range(len(candidatesF)), candidateSize)
    #     for i in range(candidateSize):
    #         candidates.append(candidatesT[indexListT[i]])
    #         candidates.append(candidatesF[indexListF[i]])
    #     np.random.shuffle(candidates)
    #     for cand in candidates:
    #         if self.getParty(cand[0]) != self.getParty(cand[1]):
    #             print("Error on query2")
    #         pId = self.getParty(cand[0])
    #         result[pId].append(cand)
    #     return result

    # InterCount true 2-stars
    # def query2(self):
    #     result = []
    #     for i in range(len(self.outE)):
    #         result.append([])
    #     self.findTwoStars2()
    #     candidatesT = list(self.twoStars2Flat)
    #     self.noise2 = discreteLaplace(self.pb.e3 / 2, self.dMax[1] - 1)
    #     # number = len(self.twoStars2Flat) + self.noise2
    #     # random.shuffle(self.twoStars2Flat)
    #     candidatesF = self.findFakeTwoStars2(len(candidatesT))
    #     candidates = list(self.twoStars2Flat)
    #     candidateSize = len(candidatesT) + self.noise2
    #     indexListT = np.random.choice(range(len(candidatesT)), candidateSize)
    #     indexListF = np.random.choice(range(len(candidatesF)), candidateSize)
    #     for i in range(candidateSize):
    #         # candidates.append(candidatesT[indexListT[i]])
    #         candidates.append(candidatesF[indexListF[i]])
    #     np.random.shuffle(candidates)
    #     for cand in candidates:
    #         if self.getParty(cand[0]) != self.getParty(cand[1]):
    #             print("Error on query2")
    #         pId = self.getParty(cand[0])
    #         result[pId].append(cand)
    #     return result

    # InterCount final
    def query2(self):
        result = []
        for i in range(len(self.outE)):
            result.append([])
        self.findTwoStars2()
        candidatesT = list(self.twoStars2Flat)
        self.noise2 = discreteLaplace(self.pb.e3 / 2, self.dMax[1] - 1)
        # number = len(self.twoStars2Flat) + self.noise2
        # random.shuffle(self.twoStars2Flat)
        candidatesF = self.findFakeTwoStars2(len(candidatesT))
        candidates = []
        candidateSize = len(candidatesT) + self.noise2
        indexListT = np.random.choice(range(len(candidatesT)), candidateSize)
        indexListF = np.random.choice(range(len(candidatesF)), candidateSize)
        for i in range(candidateSize):
            candidates.append(candidatesT[indexListT[i]])
        # candidates = list(set(candidates))
        # self.distinct2 = len(candidates)
        for i in range(candidateSize):
            candidates.append(candidatesF[indexListF[i]])
        # candidates = list(set(candidates))
        np.random.shuffle(candidates)
        print("Candidates: {}".format(len(candidates)))
        for cand in candidates:
            if self.getParty(cand[0]) != self.getParty(cand[1]):
                print("Error on query2")
            pId = self.getParty(cand[0])
            result[pId].append(cand)
        return result

    # # InterCountComm
    # def query2(self):
    #     result = []
    #     for i in range(len(self.outE)):
    #         result.append([])
    #     self.findTwoStars2()
    #     candidatesT = list(self.twoStars2Flat)
    #     self.noise2 = discreteLaplace(self.pb.e3 / 4, self.dMax[1] - 1)
    #     print("Two Star noise is {}".format(self.noise2), file=LOG_FILE)
    #     # number = len(self.twoStars2Flat) + self.noise2
    #     # random.shuffle(self.twoStars2Flat)
    #     candidatesF = self.findFakeTwoStars2(len(candidatesT))
    #     candidates = []
    #     candidateSize = len(candidatesT) + self.noise2
    #     indexListT = np.random.choice(range(len(candidatesT)), candidateSize)
    #     indexListF = np.random.choice(range(len(candidatesF)), candidateSize)
    #     for i in range(candidateSize):
    #         candidates.append(candidatesT[indexListT[i]])
    #         candidates.append(candidatesF[indexListF[i]])
    #     # candidates = list(set(candidates))
    #     np.random.shuffle(candidates)
    #     # groups = splitGroups(candidates, groupNumber=max(min(1000, candidateSize), math.floor(np.sqrt(candidateSize))))
    #     CANDIDATE_LENGTH.append(len(candidates))
    #     if self.groupFraction is not None:
    #         self.groupNumber2 = int(len(candidates) * self.groupFraction)
    #         print("GroupFraction = {}, self.groupNumber = {}".format(
    #             self.groupFraction, self.groupNumber2), file=LOG_FILE)
    #     if self.groupNumber2 > len(candidates):
    #         print("groupNumber2 = {}, len(candidates) = {}".format(self.groupNumber2, len(candidates)),
    #               file=LOG_FILE)
    #         for cand in candidates:
    #             if self.getParty(cand[0]) != self.getParty(cand[1]):
    #                 print("Error on query2")
    #             pId = self.getParty(cand[0])
    #             result[pId].append(cand)
    #         return result
    #     groups = splitGroups(candidates, groupNumber=self.groupNumber2)
    #     print("GroupNum = {}".format(len(groups)), file=LOG_FILE)
    #     for group in groups:
    #         cand = EM(self.pb.e3 / 2, self.ss.EM, group)
    #         if self.getParty(cand[0]) != self.getParty(cand[1]):
    #             print("Error on query2")
    #         pId = self.getParty(cand[0])
    #         result[pId].append(cand)
    #     return result

    # def query2(self):
    #     result = []
    #     for i in range(len(self.outE)):
    #         result.append([])
    #     self.findTwoStars2()
    #     candidates = list(self.twoStars2Flat)
    #     self.noise2 = discreteLaplace(self.pb.e3 / 2, self.dMax - 1)
    #     # number = len(self.twoStars2Flat) + self.noise2
    #     # random.shuffle(self.twoStars2Flat)
    #     if self.noise2 < 0:
    #         for i in range(abs(self.noise2)):
    #             delId = random.randint(0, len(candidates) - 1)
    #             del candidates[delId]
    #     elif self.noise2 > 0:
    #         fakeList = self.findFakeTwoStars2(self.noise2)
    #         candidates.extend(fakeList)
    #     groups = splitGroups(candidates, self.groupNumber2)
    #     print(len(candidates), len(sum(groups, [])))
    #     print(len(groups))
    #     for i in range(len(groups)):
    #         group = groups[i]
    #         fakeList = self.findFakeTwoStars2(len(group))
    #         groups[i].extend(fakeList)
    #         # random.shuffle(groups[i])
    #     # random.shuffle(groups)
    #     for i in range(len(groups)):
    #         group = groups[i]
    #         res = EM(self.pb.EM, self.ss.EM, group)
    #         if self.getParty(res[0]) != self.getParty(res[1]):
    #             print("Error on query2")
    #         pId = self.getParty(res[0])
    #         result[pId].append(res)
    #     return result

    def response2(self, twoStars):
        result = []
        twoStars.sort()
        oldTwoStar = None
        oldFlag = None
        for twoStar in twoStars:
            if twoStar == oldTwoStar:
                flag = oldFlag
            else:
                flag = (twoStar[0][1], twoStar[1][1]) in nx.edges(self.G)
                flag = RR(self.pb.e3, flag)
                oldTwoStar = twoStar
                oldFlag = flag
            result.append((twoStar, flag))
        return result

    def received2(self, messages):
        self.receivedMessages2.extend(messages)

    def triangleCount2(self):
        result, counter = 0, 0
        p = math.exp(self.pb.e3)
        p = p / (1 + p)
        for message in self.receivedMessages2:
            if message[0][0][0] == message[0][1][0]:
                counter += 1
                if message[1]:
                    result += 1
        result = (p - 1) / (2 * p - 1) * counter + result / (2 * p - 1)
        candidateSize = len(self.twoStars2Flat)
        result = result * candidateSize / counter

        # result = result * self.groupSize2 * len(self.receivedMessages2) / counter
        # if self.noise2 < 0:
        #     result = result * (1 - self.noise2 / (len(self.twoStars2Flat) + self.noise2))
        return result

    # def findTwoStars3(self):
    #     # To modify
    #     for i in range(len(self.outE)):
    #         self.twoStars3.append([])
    #     for i in range(len(self.outE)):
    #         elist1 = self.outE[i]
    #         for j in range(i + 1, len(self.outE)):
    #             elist2 = self.outE[j]
    #             for e1 in elist1:
    #                 for e2 in elist2:
    #                     if e1[0] == e2[0]:
    #                         self.twoStars3[i].append((e1, e2))
    #                         self.twoStars3[j].append((e2, e1))
    #                         self.twoStars3Flat.add((e1, e2))
    #     starNum = []
    #     for twoStar in self.twoStars3:
    #         starNum.append(len(twoStar))
    #     print("(Tips: the sum of the former is twice of the latter)Two stars for type3 num is: {}, all is {}".format(
    #         starNum, len(self.twoStars3Flat)), file=LOG_FILE)
    #
    # def findFakeTwoStars3(self, number):
    #     counter = 0
    #     result = []
    #     while counter < number:
    #         e1, e2 = random.sample(self.outEFlat, 2)
    #         if self.getParty(e1) == self.getParty(e2):
    #             continue
    #         if (e1, e2) in self.twoStars3Flat or (e2, e1) in self.twoStars3Flat:
    #             continue
    #         result.append((e1, e2))
    #         counter += 1
    #     return result
    #
    # def query3(self):
    #     for i in range(self.number):
    #         if i == self.id:
    #             continue
    #         elif i < self.id:
    #             for chosenEdgeInfo in self.chosenEdge3[i]:
    #                 chosenEdge, edgeId, targetParty = chosenEdgeInfo
    #                 edgeList = []
    #                 for edge in self.outE[targetParty]:
    #                     if edge[0] == chosenEdge[0]:
    #                         edgeList.append(edge)
    #                 if len(edgeList) > 0:
    #                     sendEdge = random.sample(edgeList, 1)[0]
    #                     Party.trueCounter += 1
    #
    #                 else:
    #                     sendEdge = (-self.id, -self.id)
    #                     Party.errorCounter += 1
    #                 self.sendEdge3[targetParty].append((sendEdge, edgeId, self.id))
    #                 sendFakeEdge = random.sample(self.outEList[targetParty], 1)[0]
    #                 self.sendEdge3[targetParty].append((sendFakeEdge, edgeId + 1, self.id))
    #         else:
    #             self.noise3[i] = discreteLaplace(self.pb.dLap, self.ss.dLap)
    #             trueNumber = int(self.fraction3 * len(self.outE[i]))
    #             number = trueNumber + self.noise3[i]
    #             for j in range(number):
    #                 otherParty = list(np.arange(self.number))
    #                 otherParty.remove(self.id)
    #                 otherParty.remove(i)
    #                 Party.edgeId += 1
    #                 chosenEdge = random.sample(self.outEList[i], 1)[0]
    #                 targetParty = random.sample(otherParty, 1)[0]
    #                 self.chosenEdge3[i].append((chosenEdge, Party.edgeId, targetParty))
    #                 edgeList = []
    #                 for edge in self.outE[targetParty]:
    #                     if edge[0] == chosenEdge[0]:
    #                         edgeList.append(edge)
    #                 if len(edgeList) > 0:
    #                     sendEdge = random.sample(edgeList, 1)[0]
    #                     Party.trueCounter += 1
    #                 else:
    #                     sendEdge = (-self.id, -self.id)
    #                     Party.errorCounter += 1
    #                 self.sendEdge3[targetParty].append((sendEdge, Party.edgeId, self.id))
    #                 Party.edgeId += 1
    #                 sendFakeEdge = random.sample(self.outEList[targetParty], 1)[0]
    #                 self.sendEdge3[targetParty].append((sendFakeEdge, Party.edgeId, self.id))
    #     return self.chosenEdge3, self.sendEdge3
    #
    # def response3(self):
    #     result = []
    #     for i in range(self.number):
    #         result.append([])
    #     self.receivedEdge3.sort(key=lambda x: x[1])
    #     for i in range(0, len(self.receivedEdge3), 2):
    #         edge1, edgeId1, partyId1 = self.receivedEdge3[i]
    #         edge2, edgeId2, partyId2 = self.receivedEdge3[i + 1]
    #         if edgeId1 != edgeId2:
    #             print(datetime.datetime.now())
    #             print("Error in response3, edgeInfo1 = {}, edgeInfo2 = {}, current Party is {}".format(
    #                 self.receivedEdge3[i], self.receivedEdge3[i + 1], self.id), file=LOG_FILE)
    #         flag = edge1[1] == edge2[1]
    #         flag = RR(self.pb.RR, flag)
    #         result[partyId1].append((flag, edgeId1))
    #         result[partyId2].append((flag, edgeId2))
    #     return result
    #
    # def received3(self, messages, chosenEdges, receivedEdges):
    #     self.receivedMessages3.extend(messages)
    #     self.chosenEdge3.extend(chosenEdges)
    #     self.receivedEdge3.extend(receivedEdges)
    #
    # def triangleCount3(self):
    #     print("Party {}: Triangle Count begin: {}".format(self.id, datetime.datetime.now()))
    #     result, counter = 0, 0
    #     p = math.exp(self.pb.RR)
    #     p = p / (1 + p)
    #     for message in self.receivedMessages3:
    #         if message[1] % 2 == 1:
    #             counter += 1
    #             if message[0]:
    #                 result += 1
    #     result = (p - 1) / (2 * p - 1) * counter + result / (2 * p - 1)
    #     result = result / self.fraction3
    #     print("Triangle Count end: {}".format(datetime.datetime.now()))
    #     return result


class Algorithm:
    def __init__(self, globalController, privacyBudget, sensitivity, groupNumber, groupFraction=None):
        self.gc = globalController
        self.pb = privacyBudget
        self.aggregator = Aggregator()
        self.parties = []
        self.degreeMax = []
        self.getDegree()
        print("dMax = {}".format(self.degreeMax), file=LOG_FILE)
        for i in range(len(self.gc.communities)):
            party = Party(i, len(self.gc.communities), gc.G, nx.subgraph(self.gc.G, self.gc.communities[i]),
                          self.gc.boundary_edge[i], privacyBudget, sensitivity, groupNumber, self.degreeMax,
                          groupFraction)
            self.parties.append(party)
        self.partyNumber = len(self.parties)

    def setPrivacyBudget(self, e1, e2, e3):
        for p in self.parties:
            p.pb.dLap = e1
            p.pb.EM = e2
            p.pb.RR = e3
            p.pb.e1 = e1
            p.pb.e2 = e2
            p.pb.e3 = e3

    def getDegree(self):
        d11 = max(list(self.gc.subG[0].degree()), key=lambda item: item[1])[1] + discreteLaplace(self.pb.e1, 1)
        d12 = max(list(self.gc.subG[1].degree()), key=lambda item: item[1])[1] + discreteLaplace(self.pb.e1, 1)
        d1 = max(d11, d12)
        d2 = max(list(self.gc.subG[2].degree()), key=lambda item: item[1])[1] + discreteLaplace(self.pb.e1 / 2, 1)
        self.degreeMax = [d1, d2]

    def triangleCount1(self):
        for i in range(self.partyNumber):
            party = self.parties[i]
            self.aggregator.TriangleCount1(i, party.triangleCount1())
        triangle1 = sum(self.aggregator.triangle1.values())
        print("Estimate of Triangle1 is {}, True is {}, Percent is {}".format(
            triangle1, self.gc.triangleSum1, triangle1 / self.gc.triangleSum1), file=LOG_FILE)
        return triangle1, self.gc.triangleSum1, triangle1 / self.gc.triangleSum1

    def triangleCount2(self):
        for i in range(self.partyNumber):
            print(datetime.datetime.now(), "Party: ", i, " begin")
            party = self.parties[i]
            queries = party.query2()
            for j in range(self.partyNumber):
                print(datetime.datetime.now(), "Party: ", j, " answer")
                if i == j:
                    continue
                party.received2(self.parties[j].response2(queries[j]))
            self.aggregator.TriangleCount2(i, party.triangleCount2())
        triangle2 = sum(self.aggregator.triangle2.values())
        print("Aggregator: ", self.aggregator.triangle2, file=LOG_FILE)
        print("True: ", self.gc.triangle2, file=LOG_FILE)
        print("Estimate of Triangle2 is {}, True is {}, Percent is {}".format(
            triangle2, self.gc.triangleSum2, triangle2 / self.gc.triangleSum2), file=LOG_FILE)
        return triangle2, self.gc.triangleSum2, triangle2 / self.gc.triangleSum2

    def triangleCount3(self):
        for i in range(self.partyNumber):
            print(datetime.datetime.now(), "Triangle3: Party: ", i, " begin")
            party = self.parties[i]
            chosenEdges, sendEdges = party.query3()
            for j in range(self.partyNumber):
                print(datetime.datetime.now(), "    Party: ", j, " receive")
                if i == j:
                    continue
                self.parties[j].received3([], chosenEdges[j], sendEdges[j])
        for i in range(self.partyNumber):
            print(datetime.datetime.now(), "Triangle3: Party: ", i, " response")
            party = self.parties[i]
            responses = party.response3()
            for j in range(self.partyNumber):
                print(datetime.datetime.now(), "    Party: ", j, " receive")
                if i == j:
                    continue
                self.parties[j].received3(responses[j], [], [])
        for i in range(self.partyNumber):
            party = self.parties[i]
            self.aggregator.TriangleCount3(i, party.triangleCount3())
        triangle3 = sum(self.aggregator.triangle3.values())
        print("Aggregator: ", self.aggregator.triangle3, file=LOG_FILE)
        print("True: ", self.gc.triangle3, file=LOG_FILE)
        print("Estimate of Triangle3 is {}, True is {}, Percent is {}".format(
            triangle3, self.gc.triangleSum3, triangle3 / self.gc.triangleSum3))
        return triangle3, self.gc.triangleSum3, triangle3 / self.gc.triangleSum3

    def triangleCount3New(self):
        pass

    def run(self):
        # self.triangleCount1()
        # self.parties[0].findTwoStars2()
        t1 = self.triangleCount1()
        # t2 = self.triangleCount2()
        # estimatedTriangle = t1[0] + t2[0]
        # trueTriangle = t1[1] + t2[1]
        estimatedTriangle = t1[0]
        trueTriangle = t1[1] + self.gc.triangleSum2
        squaredError = pow(estimatedTriangle - trueTriangle, 2)
        relativeError = abs(estimatedTriangle - trueTriangle) / max(trueTriangle, 0.001 * self.gc.G.number_of_nodes())
        return estimatedTriangle, trueTriangle, squaredError, relativeError


def init_logfile():
    global ERROR_FILE
    global LOG_FILE
    global RESULT_FILE
    ERROR_FILE = open(ERROR_FILE, 'a', buffering=1)
    LOG_FILE = open(LOG_FILE, 'a', buffering=1)
    RESULT_FILE = open(RESULT_FILE, 'a', buffering=1)


if __name__ == '__main__':
    t = Timer()
    t.setup()
    # sourceFile = "Cit-HepTh.txt"
    # sourceFile = "Wiki-Vote.txt"
    sourceFile = "Email-Enron.txt"
    args = sys.argv
    if len(args) > 2:
        sourceFile = args[2] + ".txt"
    if len(args) > 1:
        ERROR_FILE = "error_" + args[1] + ".log"
        LOG_FILE = "run_" + args[1] + ".log"
        RESULT_FILE = "result_" + args[1] + ".log"
    init_logfile()
    #### IMDB ####
    # avg_t1, avg_t2, rounds = 0, 0, 100
    # sf = "outEdge_n80000_itr"
    # for i in range(rounds):
    #     a = rounds % 10
    #     sourceFile = sf + str(a) + ".txt"
    #     print(sourceFile, file=LOG_FILE)
    #     gc = GlobalController(sourceFile)
    #     avg_t1 = avg_t1 + gc.triangleSum1
    #     avg_t2 = avg_t2 + gc.triangleSum2
    # avg_t1, avg_t2 = avg_t1 / rounds, avg_t2 / rounds
    # print("Average of Triangle1 is {}, Triangle2 is {}".format(avg_t1, avg_t2))
    # print("Average of Triangle1 is {}, Triangle2 is {}".format(avg_t1, avg_t2), file=RESULT_FILE)
    # sys.exit(-1)

    #### Others ####
    # avg_t1, avg_t2, rounds = 0, 0, 100
    # for i in range(rounds):
    #     print(sourceFile, file=LOG_FILE)
    #     gc = GlobalController(sourceFile)
    #     avg_t1 = avg_t1 + gc.triangleSum1
    #     avg_t2 = avg_t2 + gc.triangleSum2
    # avg_t1, avg_t2 = avg_t1 / rounds, avg_t2 / rounds
    # print("Average of Triangle1 is {}, Triangle2 is {}".format(avg_t1, avg_t2))
    # print("Average of Triangle1 is {}, Triangle2 is {}".format(avg_t1, avg_t2), file=RESULT_FILE)
    # sys.exit(-1)

    gc = GlobalController(sourceFile)
    pb = PrivacyBudget()
    ss = Sensitivity()

    estimate2, true2, l2, re = [], [], [], []

    # alg = Algorithm(gc, pb, ss, [10, 10])
    # resList = alg.run()
    # print(resList)
    # sys.exit(-1)

    # Version0: test one round
    # pb.setALL(1, 1, 1)
    # alg = Algorithm(gc, pb, ss, [10, 2])
    # res = alg.run()
    # estimate2.append(res[0])
    # true2.append(res[1])
    # percent.append(res[2])

    # Version1: change RR
    # for e in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    # for e in [0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4]:
    # for e in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    #     print("Epsilon = ", e)
    #     print("Epsilon = ", e, file=LOG_FILE)
    #     pb.setALL(1, e, 0.4)
    #     alg = Algorithm(gc, pb, ss, [10, 10])
    #     resList = alg.run()
    #     estimate2.append(resList[0])
    #     true2.append(resList[1])
    #     percent.append(resList[2])

    # Version2: change groupSize
    # pb.setALL(1, 1, 1)
    # for gs in [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]:
    #     print("groupSize = ", gs)
    #     print("groupSize = ", gs, file=LOG_FILE)
    #     alg = Algorithm(gc, pb, ss, [gs, gs])
    #     res = alg.run()
    #     estimate2.append(res[0])
    #     true2.append(res[1])
    #     percent.append(res[2])

    # Version3:
    # print("TDPTC Origin without Opt")
    # print("{}: sourceFile is {}".format(datetime.datetime.now(), sourceFile), file=LOG_FILE)
    for e in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]:
    # for e in [1, 2]:
        print("{}: Epsilon = {}".format(datetime.datetime.now(), e))
        print("{}: Epsilon = {}".format(datetime.datetime.now(), e), file=LOG_FILE)
        # pb.setALL(0.2 * e, 0.1 * e, 0.7 * e)
        print("Inner Triangle Count")
        pb.setALL(0.2 * e, 0.8 * e, 0 * e)

        alg = Algorithm(gc, pb, ss, [10000, 10000])
        resList = alg.run()
        estimate2.append(resList[0])
        true2.append(resList[1])
        l2.append(resList[2])
        re.append(resList[3])
        print("Estimate is {}, True is {}, l2 = {}, re is {}".format(
            resList[0], resList[1], resList[2], resList[3]))

    # # Version4: change groupNumber to test EM for reducing communication cost(Fraction)
    # # print("TDPTC Origin without Opt")
    # print("{}: sourceFile is {}".format(datetime.datetime.now(), sourceFile), file=LOG_FILE)
    # e = 2
    # if len(args) > 3:
    #     e = args[3]
    # print("{}: Epsilon = {}".format(datetime.datetime.now(), e))
    # print("{}: Epsilon = {}".format(datetime.datetime.now(), e), file=LOG_FILE)
    # # for e in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]:
    # #     print("{}: Epsilon = {}".format(datetime.datetime.now(), e))
    # #     print("{}: Epsilon = {}".format(datetime.datetime.now(), e), file=LOG_FILE)
    # # for e in [1, 2]:
    # for i in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]:
    #     gf = math.pow(2, i)
    #     pb.setALL(0.2 * e, 0.1 * e, 0.7 * e)
    #     # print("Inner Triangle Count")
    #     # pb.setALL(0.2 * e, 0.8 * e, 0 * e)
    #
    #     alg = Algorithm(gc, pb, ss, [10000, 10000], groupFraction=gf)
    #     resList = alg.run()
    #     estimate2.append(resList[0])
    #     true2.append(resList[1])
    #     l2.append(resList[2])
    #     re.append(resList[3])
    #     print("Estimate is {}, True is {}, l2 = {}, re is {}".format(
    #         resList[0], resList[1], resList[2], resList[3]))
    #     # print("Candidate length is {}".format(CANDIDATE_LENGTH), file=ERROR_FILE)
    # # print("Average is ", file=ERROR_FILE)
    # # print(sum(CANDIDATE_LENGTH) / len(CANDIDATE_LENGTH), file=ERROR_FILE)

    # Version5: change groupNumber to test EM for reducing communication cost
    # print("TDPTC Origin without Opt")
    # for gn in [1000, 5000, 10000, 50000, 100000, 500000]:
    #     e = 2
    #     print("Epsilon = ", e)
    #     print("Epsilon = ", e, file=LOG_FILE)
    #     print("Group Number = {}".format(gn))
    #     print("Group Number = {}".format(gn), file=LOG_FILE)
    #
    #     pb.setALL(0.2 * e, 0.1 * e, 0.7 * e)
    #
    #     alg = Algorithm(gc, pb, ss, [gn, gn])
    #     resList = alg.run()
    #     estimate2.append(resList[0])
    #     true2.append(resList[1])
    #     l2.append(resList[2])
    #     re.append(resList[3])
    #     print("Estimate is {}, True is {}, l2 = {}, re is {}".format(
    #         resList[0], resList[1], resList[2], resList[3]))

    t.printNow()
    print(Party.errorCounter)
    print(Party.trueCounter)
    print(estimate2)
    print(true2)
    print(l2)
    print(re)
    print(estimate2, file=RESULT_FILE)
    print(true2, file=RESULT_FILE)
    print(l2, file=RESULT_FILE)
    print(re, file=RESULT_FILE)
