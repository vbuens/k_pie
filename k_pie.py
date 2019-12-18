import os
import sys
import argparse
import utils
import cv2
import glob
import webcolors
import numpy as np
import matplotlib.pyplot as plt
from infection_functions import get_image,color_name,print_image, get_threshold_values, closest_colour, get_threshold_value


def main(args):
    output_dir = args.outputdir
    input_dir = args.inputdir
    format = args.format
    healthy = args.negativecontrol
    infected = args.positivecontrol
    p_rgb = args.positiveRGBvalues
    n_rgb = args.negativeRGBvalues
    K = args.k #number of colours

    # Writing output file
    output_file = str(output_dir)+"/infection_percentages.csv"
    output = open(output_file, "a")
    output.write("Image,K,%Infection")

    # Finding standard values
    inf_values,healthy_values=checkpoint(input_dir,output_dir,healthy,n_rgb,infected,p_rgb,K)
    print('Healthy RGB value: (%s)\nInfected RGB value: (%s)\n' % (str(healthy_values).strip('[]'),str(inf_values).strip('[]')))

    for imagefile in glob.glob(str(input_dir)+"/*"+str(format)):
        image = get_image(imagefile)
        print("- File : %s " % imagefile)
        Z = image.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,labels,center_colours=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        print_image(imagefile,image,center_colours,labels,K,output_dir)
        # plt.clf()

        counts={}
        perc={}
        for i in range(0,K):
            counts[i]=list(labels.flatten()).count(i)
            perc[i]=round((list(labels.flatten()).count(i)/len(labels))*100,2)
        colorlabels=color_name(center_colours)

        # We get ordered colors by iterating through the keys
        ordered_colors = [center_colours[i]/255 for i in counts.keys()]

        # Get rid of black colour (background)
        blackindeces= [i for i, x in enumerate(colorlabels) if x == "black"]
        for blackindex in blackindeces:
            # Delete values for black in all dictionaries and arrays
            counts.pop(blackindex)
            perc.pop(blackindex)
            blackindex2=colorlabels.index('black')
            ordered_colors.pop(blackindex2)
            colorlabels.pop(blackindex2)
            colornames=[]
            for colour in center_colours:
                colornames.append(closest_colour((colour[0],colour[1],colour[2])))
            colornames.pop(blackindex2)
            list(center_colours).pop(blackindex2)
        # Update percentages after removing values from background
        newpercs=[round((i/sum(perc.values()))*100,2) for i in perc.values()]

        #Label pie with percentage for each colour
        pielabel=list(map(': '.join, zip(colorlabels, map(str,newpercs))))
        plt.pie(counts.values(),labels=pielabel,colors=ordered_colors)
        plt.title(os.path.splitext(os.path.basename(imagefile))[0])
        plt.savefig('%s/%s_%i_pie.png' % (output_dir,imagefile.split('.')[0].split('/')[1],K))
        plt.clf()

        # Get colours
        colors={}
        for i in range(0,len(center_colours)):
            rc=center_colours[i][0] ; gc=center_colours[i][1] ; bc=center_colours[i][2]
            rd=(healthy_values[0]-rc)**2
            gd=(healthy_values[1]-gc)**2
            bd=(healthy_values[2]-bc)**2
            healthy_rgb=rd + gd + bd
            rd_inf=(inf_values[0]-rc)**2
            gd_inf=(inf_values[1]-gc)**2
            bd_inf=(inf_values[2]-bc)**2
            inf_rgb=rd_inf + gd_inf + bd_inf
            colors[color_name(center_colours)[i]]=[healthy_rgb,inf_rgb]
        color_name(center_colours)

        # Classify each colour in 'infected' or 'healthy' and estimate percentage of infection
        percinf=0
        totalpercs=sum(newpercs)
        count=0
        if 'black' in colors:
            colors.pop('black')

        for color in colors.keys():
            if colors[color].index(min(colors[color])) == 0 :
                print("Healthy: %s" % color)
            else:
                print("Infected: %s" % color)
                percinf+=newpercs[count]
            count+=1
        print('Percentage of infection: %d ' % percinf)
        # Write csv output file
        output.write("\n%s,%i,%f" % (str(os.path.splitext(os.path.basename(imagefile))[0]),K,percinf))

    output.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k','--k', type=int, nargs='?', help='output directory for results (default=5)',default=5)
    parser.add_argument('-o','--outputdir', type=str, help='output directory for results (default=results)',default="results")
    parser.add_argument('-i','--inputdir', required= True, type=str, help='directory where the images to analyse are found')
    parser.add_argument('-f','--format', type=str, help='format of the pictures (default=.tif)', default=".png")
    parser.add_argument('-p','--positivecontrol', required= False, type=str, help='Image file of an infected leaf')
    parser.add_argument('-n','--negativecontrol', required= False, type=str, help='Image file of a healthy leaf')
    parser.add_argument('-p_rgb','--positiveRGBvalues', required= False, type=str, help='RGB values for the positive control (infected)')
    parser.add_argument('-n_rgb','--negativeRGBvalues', required= False, type=str, help='RGB values for the negative control (healthy)')
    args = parser.parse_args()
    if not (args.positivecontrol or args.positiveRGBvalues):
        parser.error('Positive control needed. Please give a picture (-p) or a RGB value (-p_rgb)')
    if not (args.negativecontrol or args.negativeRGBvalues):
        parser.error('Negative control needed. Please give a picture (-n) or a RGB value (-n_rgb)')

    return parser.parse_args() #parser.parse_args(args)

def checkpoint(input_dir,output_dir,healthy,n_rgb,infected,p_rgb,K):
    if os.path.exists(input_dir) == False:
        parser.error("Directory %s not found" % input_dir)
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)

    # Check arguments and estimate standard values based on image or RGB value
    if healthy:
        try:
            open(healthy, 'r')
        except FileNotFoundError:
            parser.error("File %s not found" % healthy)
        healthy_values = get_threshold_value(healthy,K,output_dir)

    elif n_rgb:
        rgbvalues=n_rgb.split(',')
        healthy_values =[int(i) for i in rgbvalues]

    if infected:
        try:
            open(infected, 'r')
        except FileNotFoundError:
            parser.error("File %s not found" % infected)
        infected_values = get_threshold_value(infected,K,output_dir)
    elif p_rgb:
        rgbvalues=p_rgb.split(',')
        infected_values =[int(i) for i in rgbvalues]    #sys.stdout = open('stdout.txt', 'w')

    print('''
    Output directory : %s
    Input directory : %s
    Images format : %s
    Positive control (infected) image : %s
    Negative control (healthy) image : %s
    Number of clusters to find infection (k) : %i
    Final output: %s
    ''' % (output_dir,input_dir,format,healthy,infected,K,str(output_dir)+"/infection_percentages.csv"))

    return infected_values,healthy_values

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
