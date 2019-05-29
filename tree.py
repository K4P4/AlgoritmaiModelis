import math
import add

sizeX, sizeY, sizeZ = 10, 10, 10
n = 3
k = 0
trunk_ratio = 0.29
theta = -0.08
color = [200, 248, 253]
radius = 0.08
details = 6
iterations = 3

def draw_tree(order, theta, sz, posn, heading, color):
    trunk_ratio = 0.29
    heading+=theta

    trunk = sz * trunk_ratio
    delta_X = trunk * math.cos(heading)
    delta_Y = trunk * math.sin(heading)
    delta_z = trunk * math.sin(heading)

    (i, j, k) = posn
    newpos1 = [i + delta_X, j + delta_Y, k + delta_z]
    newpos2 = [i + delta_X, j - delta_Y, k + delta_z]
    newpos3 = [i - delta_X, j + delta_Y, k + delta_z]
    newpos4 = [i - delta_X, j - delta_Y, k + delta_z]

    newpos5 = [i + delta_Y, j + delta_X, k + delta_z]
    newpos6 = [i + delta_Y, j - delta_X, k + delta_z]
    newpos7 = [i - delta_Y, j + delta_X, k + delta_z]
    newpos8 = [i - delta_Y, j - delta_X, k + delta_z]

    add.cylinder2(posn, newpos1, radius, details, color)
    add.cylinder2(posn, newpos2, radius, details, color)
    add.cylinder2(posn, newpos3, radius, details, color)
    add.cylinder2(posn, newpos4, radius, details, color)
    add.cylinder2(posn, newpos5, radius, details, color)
    add.cylinder2(posn, newpos6, radius, details, color)
    add.cylinder2(posn, newpos7, radius, details, color)
    add.cylinder2(posn, newpos8, radius, details, color)

    if order > 0:
        color[0] -= 1
        newsz = sz * (1 - trunk_ratio)
        draw_tree(order-1, theta, newsz, newpos1, heading, color)
        draw_tree(order-1, theta, newsz, newpos2, heading, color)
        draw_tree(order-1, theta, newsz, newpos3, heading, color)
        draw_tree(order-1, theta, newsz, newpos4, heading, color)

        draw_tree(order-1, theta, newsz, newpos5, heading, color)
        draw_tree(order-1, theta, newsz, newpos6, heading, color)
        draw_tree(order-1, theta, newsz, newpos7, heading, color)
        draw_tree(order-1, theta, newsz, newpos8, heading, color)

if n == 6: k = 1.32
if n == 5: k = 1
if n == 4: k = 0.55
if n == 3: k = 0
if n == 2: k = -0.221
if n == 1: k = 4.85



draw_tree(iterations, theta, sizeZ * 0.6,
          [sizeX/2, sizeY/2, sizeZ-4-k], -math.pi/n, color)


theta = -theta
color = [200, 248, 253]
draw_tree(iterations, theta, sizeZ * 0.6,
          [sizeX/2, sizeY/2, sizeZ/10+0.73-4+k], math.pi/n, color)

add.off('Karolio.off')
