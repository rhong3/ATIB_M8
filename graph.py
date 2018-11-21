import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2


Pr = 6
RDr = 3
r = 4
sz = 200
Srepro = 1
RDrepro = 1
Prepro = 1


def init_placement(sz, num, x, Pr, RDr):
    mp = np.full((sz+1, sz+1, 3), 1)
    loc = []
    for i in range(num):
        rand = np.random.randint(sz, size=2)
        if x == 2:
            mp[rand[0], rand[1], x] = 255
            mp[rand[0], rand[1], 0] = 0
            mp[rand[0], rand[1], 1] = 0
        elif x == 1:
            mp[np.maximum((rand[0]-RDr), 0):np.minimum((rand[0] + RDr), sz),
            np.maximum((rand[1]-RDr), 0):np.minimum((rand[1] + RDr), sz), x] = 255

            mp[np.maximum((rand[0] - RDr), 0):np.minimum((rand[0] + RDr), sz),
            np.maximum((rand[1] - RDr), 0):np.minimum((rand[1] + RDr), sz), 0] = 0

            mp[np.maximum((rand[0] - RDr), 0):np.minimum((rand[0] + RDr), sz),
            np.maximum((rand[1] - RDr), 0):np.minimum((rand[1] + RDr), sz), 2] = 0
        elif x == 0:
            mp[np.maximum((rand[0] - Pr), 0):np.minimum((rand[0] + Pr), sz),
            np.maximum((rand[1] - Pr), 0):np.minimum((rand[1] + Pr), sz), x] = 255

            mp[np.maximum((rand[0] - Pr), 0):np.minimum((rand[0] + Pr), sz),
            np.maximum((rand[1] - Pr), 0):np.minimum((rand[1] + Pr), sz), 1] = 1

            mp[np.maximum((rand[0] - Pr), 0):np.minimum((rand[0] + Pr), sz),
            np.maximum((rand[1] - Pr), 0):np.minimum((rand[1] + Pr), sz), 2] = 0
        loc.append(rand)
    return mp, loc


def update(sloc, rdloc, ploc, oldmp, sz, repros, reprord, reprop, r):
    mp = np.full((sz+1, sz+1, 3), 1)
    nsloc = []
    nploc = []
    nrdloc = []
    for a in sloc:
        if oldmp[a[0], a[1], 2] > 10 or oldmp[a[0], a[1], 1] > 10:
            mp[a[0], a[1], 2] = 255
            mp[a[0], a[1], 1] = 0
            mp[a[0], a[1], 0] = 0
            nsloc.append(a)
            for i in range(repros):
                rand = np.random.randint(low=-r, high=r, size=2)
                x = a[0]+rand[0]
                y = a[1]+rand[1]
                offspring = np.asarray([x,y])
                offspring = np.clip(offspring, 0, 200)
                nsloc.append(offspring)
    for c in ploc:
        nploc.append(c)
        mp[c[0], c[1], 2] = 0
        mp[c[0], c[1], 1] = 1
        mp[c[0], c[1], 0] = 255
        for j in range(reprop):
            rand = np.random.randint(low=-r, high=r, size=2)
            x = c[0] + rand[0]
            y = c[1] + rand[1]
            offspring = np.asarray([x, y])
            offspring = np.clip(offspring, 0, 200)
            if mp[offspring[0], offspring[1], :].tolist() == [1, 1, 1]:
                nploc.append(offspring)
    for b in rdloc:
        nrdloc.append(b)
        mp[b[0], b[1], 2] = 0
        mp[b[0], b[1], 1] = 255
        mp[b[0], b[1], 0] = 0
        for j in range(reprord):
            rand = np.random.randint(low=-r, high=r, size=2)
            x = b[0] + rand[0]
            y = b[1] + rand[1]
            offspring = np.asarray([x, y])
            offspring = np.clip(offspring, 0, 200)
            if mp[offspring[0], offspring[1], :].tolist() == [1, 1, 1]:
                nrdloc.append(offspring)
    return mp, nsloc, nploc, nrdloc


def new_graph(loc, sz, Pr, RDr, x):
    mp = np.full((sz+1, sz+1, 3), 1)
    for i in loc:
        if x == 2:
            mp[i[0], i[1], x] = 255
            mp[i[0], i[1], 0] = 0
            mp[i[0], i[1], 1] = 0
        elif x == 1:
            mp[np.maximum((i[0]-RDr), 0):np.minimum((i[0] + RDr), sz),
            np.maximum((i[1]-RDr), 0):np.minimum((i[1] + RDr), sz), x] = 255

            mp[np.maximum((i[0] - RDr), 0):np.minimum((i[0] + RDr), sz),
            np.maximum((i[1] - RDr), 0):np.minimum((i[1] + RDr), sz), 0] = 0

            mp[np.maximum((i[0] - RDr), 0):np.minimum((i[0] + RDr), sz),
            np.maximum((i[1] - RDr), 0):np.minimum((i[1] + RDr), sz), 2] = 0
        elif x == 0:
            mp[np.maximum((i[0] - Pr), 0):np.minimum((i[0] + Pr), sz),
            np.maximum((i[1] - Pr), 0):np.minimum((i[1] + Pr), sz), x] = 255

            mp[np.maximum((i[0] - Pr), 0):np.minimum((i[0] + Pr), sz),
            np.maximum((i[1] - Pr), 0):np.minimum((i[1] + Pr), sz), 1] = 1

            mp[np.maximum((i[0] - Pr), 0):np.minimum((i[0] + Pr), sz),
            np.maximum((i[1] - Pr), 0):np.minimum((i[1] + Pr), sz), 2] = 0
    return mp


def main(round, init_num, sz, Pr, RDr, Srepro, RDrepro, Prepro, r):
    S, Sloc = init_placement(sz, init_num, 2, Pr, RDr)
    RD, RDloc = init_placement(sz, init_num, 1, Pr, RDr)
    P, Ploc = init_placement(sz, init_num, 0, Pr, RDr)
    result = np.multiply((S+RD-1), P)
    result_view = result.repeat(5, axis=0).repeat(5, axis=1)
    cv2.imwrite('round-1.png', result_view)
    updatedmap, nsloc, nploc, nrdloc = update(Sloc, RDloc, Ploc, result, sz, Srepro, RDrepro, Prepro, r)
    updatedmap_view = updatedmap.repeat(5, axis=0).repeat(5, axis=1)
    cv2.imwrite('result-1.png', updatedmap_view)
    for i in range(round-1):
        S = new_graph(nsloc, sz, Pr, RDr, 2)
        RD = new_graph(nrdloc, sz, Pr, RDr, 1)
        P = new_graph(nploc, sz, Pr, RDr, 0)
        new = np.multiply((S+RD-1), P)
        new_view = new.repeat(5, axis=0).repeat(5, axis=1)
        cv2.imwrite('round-{}.png'.format(str(i+2)), new_view)
        updatedmap, nsloc, nploc, nrdloc = update(nsloc, nrdloc, nploc, new, sz, Srepro, RDrepro, Prepro, r)
        updatedmap_view = updatedmap.repeat(5, axis=0).repeat(5, axis=1)
        cv2.imwrite('result-{}.png'.format(str(i+2)), updatedmap_view)


main(30, 10, sz, Pr, RDr, Srepro, RDrepro, Prepro, r)