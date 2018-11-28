"""
Created on 11/20/2018

@author: RH
"""
# This simulator is created by using the ideas of Extended Data Figure 3.
import numpy as np
import cv2


sz = 200  # Size of simulation graph
APr = 6  # Antibiotics radius
BPr = 6  # Antibiotics radius
CPr = 6  # Antibiotics radius
ARDr = 3  # Degrading radius
BRDr = 3  # Degrading radius
CRDr = 3  # Degrading radius
Ar = 4  # Reproducing radius
Br = 4  # Reproducing radius
Cr = 4  # Reproducing radius
Arepro = 1  # Number of offsprings produced
Brepro = 1  # Number of offsprings produced
Crepro = 1  # Number of offsprings produced


# Initialize the graph with 'num' of bacteria randomly placed
def init_placement(sz, num, xt):
    Smp = np.full((sz+1, sz+1, 3), 1)
    Pmp = np.full((sz+1, sz+1, 3), 1)
    RDmp = np.full((sz+1, sz+1, 3), 1)
    loc = []
    if xt == 'A':
        cl = [0, 1, 255]
        Pr = APr
        RDr = ARDr
    elif xt == 'B':
        cl = [1, 255, 0]
        Pr = BPr
        RDr = BRDr
    elif xt == 'C':
        cl = [255, 0, 1]
        Pr = CPr
        RDr = CRDr
    else:
        print('Error!')
        cl = [0,0,0]
        Pr = 0
        RDr = 0
    for i in range(num):
        rand = np.random.randint(sz, size=2)
        Smp[rand[0], rand[1], 0] = cl[0]
        Smp[rand[0], rand[1], 1] = cl[1]
        Smp[rand[0], rand[1], 2] = cl[2]

        RDmp[np.maximum((rand[0] - RDr), 0):np.minimum((rand[0] + RDr), sz+1),
        np.maximum((rand[1] - RDr), 0):np.minimum((rand[1] + RDr), sz+1), 0] = cl[0]
        RDmp[np.maximum((rand[0] - RDr), 0):np.minimum((rand[0] + RDr), sz+1),
        np.maximum((rand[1] - RDr), 0):np.minimum((rand[1] + RDr), sz+1), 1] = cl[1]
        RDmp[np.maximum((rand[0] - RDr), 0):np.minimum((rand[0] + RDr), sz+1),
        np.maximum((rand[1] - RDr), 0):np.minimum((rand[1] + RDr), sz+1), 2] = cl[2]

        Pmp[np.maximum((rand[0] - Pr), 0):np.minimum((rand[0] + Pr), sz+1),
        np.maximum((rand[1] - Pr), 0):np.minimum((rand[1] + Pr), sz+1), 0] = cl[0]
        Pmp[np.maximum((rand[0] - Pr), 0):np.minimum((rand[0] + Pr), sz+1),
        np.maximum((rand[1] - Pr), 0):np.minimum((rand[1] + Pr), sz+1), 1] = cl[1]
        Pmp[np.maximum((rand[0] - Pr), 0):np.minimum((rand[0] + Pr), sz+1),
        np.maximum((rand[1] - Pr), 0):np.minimum((rand[1] + Pr), sz+1), 2] = cl[2]

        loc.append(rand)

    mp = [Smp, RDmp, Pmp]
    return mp, loc


def fuse(ASmp, BSmp, CSmp):
    mpp = np.multiply(np.multiply(ASmp, BSmp), CSmp)
    mpp_view = mpp.repeat(5, axis=0).repeat(5, axis=1)
    return mpp_view


# Simulation happens: killing, protection, reproducing
def sim(Smp, RDmp, Pmp):
    result = np.clip(np.multiply(np.clip((Smp+RDmp), a_max = 255, a_min = 0), Pmp), a_max = 255, a_min = 0)
    result_view = result.repeat(5, axis=0).repeat(5, axis=1)
    return result, result_view


# Update the map for next round
def update(xt, loc, resultmp):
    Smp = np.full((sz + 1, sz + 1, 3), 1)
    Pmp = np.full((sz + 1, sz + 1, 3), 1)
    RDmp = np.full((sz + 1, sz + 1, 3), 1)
    newloc = []
    if xt == 'A':
        cl = [0, 1, 255]
        r = Ar
        repro = Arepro
        Pr = APr
        RDr = ARDr
    elif xt == 'B':
        cl = [1, 255, 0]
        r = Br
        repro = Brepro
        Pr = BPr
        RDr = BRDr
    elif xt == 'C':
        cl = [255, 0, 1]
        r = Cr
        repro = Crepro
        Pr = CPr
        RDr = CRDr
    else:
        print('Error')
        cl = [0,0,0]
        r = 0
        repro = 0
        Pr = 0
        RDr = 0
    for a in loc:
        if xt == 'A':
            det = resultmp[a[0], a[1], 0] == 255 or resultmp[a[0], a[1], 1] == 2
        elif xt == 'B':
            det = resultmp[a[0], a[1], 2] == 255 or resultmp[a[0], a[1], 0] == 2
        elif xt == 'C':
            det = resultmp[a[0], a[1], 1] == 255 or resultmp[a[0], a[1], 2] == 2
        else:
            print('Error!')
            det = False
        if det:
            Smp[a[0], a[1], 0] = cl[0]
            Smp[a[0], a[1], 1] = cl[1]
            Smp[a[0], a[1], 2] = cl[2]

            RDmp[np.maximum((a[0] - RDr), 0):np.minimum((a[0] + RDr), sz+1),
            np.maximum((a[1] - RDr), 0):np.minimum((a[1] + RDr), sz+1), 0] = cl[0]
            RDmp[np.maximum((a[0] - RDr), 0):np.minimum((a[0] + RDr), sz+1),
            np.maximum((a[1] - RDr), 0):np.minimum((a[1] + RDr), sz+1), 1] = cl[1]
            RDmp[np.maximum((a[0] - RDr), 0):np.minimum((a[0] + RDr), sz+1),
            np.maximum((a[1] - RDr), 0):np.minimum((a[1] + RDr), sz+1), 2] = cl[2]

            Pmp[np.maximum((a[0] - Pr), 0):np.minimum((a[0] + Pr), sz+1),
            np.maximum((a[1] - Pr), 0):np.minimum((a[1] + Pr), sz+1), 0] = cl[0]
            Pmp[np.maximum((a[0] - Pr), 0):np.minimum((a[0] + Pr), sz+1),
            np.maximum((a[1] - Pr), 0):np.minimum((a[1] + Pr), sz+1), 1] = cl[1]
            Pmp[np.maximum((a[0] - Pr), 0):np.minimum((a[0] + Pr), sz+1),
            np.maximum((a[1] - Pr), 0):np.minimum((a[1] + Pr), sz+1), 2] = cl[2]

            newloc.append(a)
            for i in range(repro):
                rand = np.random.randint(low=-r, high=r, size=2)
                x = a[0]+rand[0]
                x = np.clip(x, 0, 200)
                y = a[1]+rand[1]
                y = np.clip(y, 0, 200)
                offspring = np.asarray([x,y])
                offspring = np.clip(offspring, 0, 200)
                if xt == 'A':
                    rdet = resultmp[x, y, 2] == 2
                elif xt == 'B':
                    rdet = resultmp[x, y, 1] == 2
                elif xt == 'C':
                    rdet = resultmp[x, y, 0] == 2
                else:
                    print('Error~')
                    rdet = False

                if rdet:
                    Smp[x, y, 0] = cl[0]
                    Smp[x, y, 1] = cl[1]
                    Smp[x, y, 2] = cl[2]

                    RDmp[np.maximum((x - RDr), 0):np.minimum((x + RDr), sz+1),
                    np.maximum((y - RDr), 0):np.minimum((y + RDr), sz+1), 0] = cl[0]
                    RDmp[np.maximum((x - RDr), 0):np.minimum((x + RDr), sz+1),
                    np.maximum((y - RDr), 0):np.minimum((y + RDr), sz+1), 1] = cl[1]
                    RDmp[np.maximum((x - RDr), 0):np.minimum((x + RDr), sz+1),
                    np.maximum((y - RDr), 0):np.minimum((y + RDr), sz+1), 2] = cl[2]

                    Pmp[np.maximum((x - Pr), 0):np.minimum((x + Pr), sz+1),
                    np.maximum((y - Pr), 0):np.minimum((y+ Pr), sz+1), 0] = cl[0]
                    Pmp[np.maximum((x - Pr), 0):np.minimum((x + Pr), sz+1),
                    np.maximum((y - Pr), 0):np.minimum((y + Pr), sz+1), 1] = cl[1]
                    Pmp[np.maximum((x - Pr), 0):np.minimum((x + Pr), sz+1),
                    np.maximum((y - Pr), 0):np.minimum((y + Pr), sz+1), 2] = cl[2]

                    newloc.append(offspring)
    mp = [Smp, RDmp, Pmp]
    return mp, newloc


# MAIN METHOD
def main(round, sz, num):
    mpA, locA = init_placement(sz, num, 'A')
    mpB, locB = init_placement(sz, num, 'B')
    mpC, locC = init_placement(sz, num, 'C')
    result = fuse(mpA[0], mpB[0], mpC[0])
    cv2.imwrite('Triangle/1.png', result)

    height, width, layers = result.shape
    print(height, width)
    result_video = cv2.VideoWriter('Triangle/Result_video.mov', -1, 10, (height, width))
    A_video = cv2.VideoWriter('Triangle/A_video.mov', -1, 10, (height, width))
    B_video = cv2.VideoWriter('Triangle/B_video.mov', -1, 10, (height, width))
    C_video = cv2.VideoWriter('Triangle/C_video.mov', -1, 10, (height, width))
    rimg = cv2.imread('Triangle/1.png')
    result_video.write(rimg)

    for i in range(round):
        Asim, Asim_view = sim(mpA[0], mpC[1], mpB[2])
        Bsim, Bsim_view = sim(mpB[0], mpA[1], mpC[2])
        Csim, Csim_view = sim(mpC[0], mpB[1], mpA[2])
        cv2.imwrite('Triangle/A-{}.png'.format(str(i + 1)), Asim_view)
        cv2.imwrite('Triangle/B-{}.png'.format(str(i + 1)), Bsim_view)
        cv2.imwrite('Triangle/C-{}.png'.format(str(i + 1)), Csim_view)
        Aimg = cv2.imread('Triangle/A-{}.png'.format(str(i+1)))
        Bimg = cv2.imread('Triangle/B-{}.png'.format(str(i+1)))
        Cimg = cv2.imread('Triangle/C-{}.png'.format(str(i+1)))
        A_video.write(Aimg)
        B_video.write(Bimg)
        C_video.write(Cimg)

        mpA, locA = update('A', locA, Asim)
        mpB, locB = update('B', locB, Bsim)
        mpC, locC = update('C', locC, Csim)
        result = fuse(mpA[0], mpB[0], mpC[0])
        cv2.imwrite('Triangle/{}.png'.format(str(i+2)), result)
        rimg = cv2.imread('Triangle/{}.png'.format(str(i+2)))
        result_video.write(rimg)

    cv2.destroyAllWindows()
    A_video.release()
    B_video.release()
    C_video.release()
    result_video.release()


if __name__ == "__main__":
    # Run
    main(500, sz, 100)
