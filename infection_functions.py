from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
import numpy as np
import webcolors
import os

# Functions:
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def color_name(center_colours):
    colornames=[]
    for i in range(0,len(center_colours)):
        colornames.append(closest_colour((center_colours[i][0],center_colours[i][1],center_colours[i][2])))
    return colornames

def print_image(imagefile,image,center_colours,labels,k,output_dir):
    center_colours = np.uint8(center_colours)
    res = center_colours[labels.flatten()]
    res2 = res.reshape((image.shape))
    plt.imshow(res2)
    plt.title(os.path.splitext(os.path.basename(imagefile))[0])
    print('Imagefile: %s ' % os.path.basename(imagefile))
    plt.savefig('%s/%s_%i_kmeans.pdf' % (output_dir,os.path.splitext(os.path.basename(imagefile))[0],k))
    plt.clf()

def get_threshold_values(control,K,output_dir):
    ## Getting the healthy and infected thresholds
    plt.rcParams['figure.figsize'] = [14, 7]
    image = get_image(control)
    Z = image.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,labels,center_colours=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    print_image(control,image,center_colours,labels,K,output_dir)

    # counts2={}
    perc={}
    counts=[]
    perc2=[]
    for i in range(0,K):
        counts.append(list(labels.flatten()).count(i))
        # counts2[i]=list(labels.flatten()).count(i)
        perc[i]=round((list(labels.flatten()).count(i)/len(labels))*100,2)
        perc2.append(round((list(labels.flatten()).count(i)/len(labels))*100,2))
    colorlabels=color_name(center_colours)

    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colours[i]/255 for i in range(0,len(counts))]

    colorlabels1=colorlabels
    # Get rid of the background and get the values for colour green (healthy) and khaki (infected)
    while 'black' in colorlabels:
        blackindex=colorlabels.index('black')
        counts.pop(blackindex)
        perc2.pop(blackindex)
        ordered_colors.pop(blackindex)
        colorlabels.pop(blackindex)
        colornames=[]
        for colour in center_colours:
            colornames.append(closest_colour((colour[0],colour[1],colour[2])))
        colornames.pop(blackindex)
        center_colours2=list(center_colours)
        center_colours2.pop(blackindex)

    # Update percentage values after removing background
    newpercs=[round((i/sum(perc2))*100,2) for i in perc2]
    greater_perc=max(perc2)
    greater_perc=perc2.index(greater_perc)

    control_values=[center_colours[greater_perc][0],center_colours[greater_perc][1],center_colours[greater_perc][2]]
    control_pie = [counts,colorlabels,ordered_colors]
    center_colours = center_colours2

    basenamecontrol=os.path.splitext(os.path.basename(control))[0]
    plt.subplot(121)
    plt.pie(control_pie[0],labels=control_pie[1],colors=control_pie[2])
    plt.title(basenamecontrol)
    plt.savefig('%s/Control_%s_pie_k%d.png' % (output_dir,basenamecontrol,K))
    plt.clf()
    return control_values
