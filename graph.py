"""
Created on 11/20/2018

@author: RH
"""
# This simulator is created by using the ideas of Extended Data Figure 3.
import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2


Pr = 6  # Antibiotics radius of species P.
RDr = 3  # Removing antibiotics radius of species RD
r = 4  # Reproducing radius of all 3 species
sz = 200  # Size of simulation graph
Srepro = 1  # Number of offsprings produced by species S each round
RDrepro = 1  # Number of offsprings produced by species RD each round
Prepro = 1  # Number of offsprings produced by species P each round


# Initialize the graph with 'num' of species randomly placed
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


# Update the graph (simulation happens: killing, protection, reproducing)
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


# Generate new graphs for next round based on the result of last round
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


# Main method. Save process and result graph of each round and eventually generate movies of simulation.
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

    height, width, layers = updatedmap_view.shape
    result_video = cv2.VideoWriter('result_video.mov', -1, 1, (width, height))
    rimg = cv2.imread('result-1.png')
    result_video.write(rimg)

    height, width, layers = result_view.shape
    process_video = cv2.VideoWriter('process_video.mov', -1, 1, (width, height))
    pimg = cv2.imread('round-1.png')
    process_video.write(pimg)

    for i in range(round-1):
        S = new_graph(nsloc, sz, Pr, RDr, 2)
        RD = new_graph(nrdloc, sz, Pr, RDr, 1)
        P = new_graph(nploc, sz, Pr, RDr, 0)
        new = np.multiply((S+RD-1), P)
        new_view = new.repeat(5, axis=0).repeat(5, axis=1)
        cv2.imwrite('round-{}.png'.format(str(i+2)), new_view)
        pimg = cv2.imread('round-{}.png'.format(str(i+2)))
        process_video.write(pimg)

        updatedmap, nsloc, nploc, nrdloc = update(nsloc, nrdloc, nploc, new, sz, Srepro, RDrepro, Prepro, r)
        updatedmap_view = updatedmap.repeat(5, axis=0).repeat(5, axis=1)
        cv2.imwrite('result-{}.png'.format(str(i+2)), updatedmap_view)
        rimg = cv2.imread('result-{}.png'.format(str(i+2)))
        result_video.write(rimg)
    cv2.destroyAllWindows()
    process_video.release()
    result_video.release()


if __name__ == "__main__":
    # Run
    main(20, 15, sz, Pr, RDr, Srepro, RDrepro, Prepro, r)
