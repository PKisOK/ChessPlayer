
#
#   Program to manually control Rajan's CNC using USB port
#   while view video from the RPi camera mounted in the CNC arm
#   and turning on and off the electromagnet using GPIO port to
#   move chess pieces
#
#   One time only:  type  sudo raspi-config on Raspian command line and
#                   select Enable Camera
#                       or
#                   use the Configuration option from the pull down menu
#                   Then reboot.
#
#   Install libray (one time only)
#
#   sudo apt-get update    (makes sure you have latest updates, run 2/6/2017)
#   sudo apt-get install python3-picamera
#   sudo apt-get install python-numpy
#   sudo apt-get install python-matplotlib
#

import os
import picamera
import serial
import RPi.GPIO as GPIO
import numpy as np
import matplotlib.pyplot as plt
import time
import math as math

def talk(cmd):
    saycmd = "echo " + cmd + " | festival --tts"
    os.system(saycmd)
    return

def board_analysis(image,xscal):

    (rows,cols) = image.shape

    low_x = 0
    low_y = 0
    hi_x = int(cols*xscal)
    hi_y = rows
    
    prompt = "Analysis Mode Command (? for list of commands) : "
    cmdlst = "List of commands: \n"
    cmdlst = cmdlst + "    v  : view full image\n"
    cmdlst = cmdlst + "    c  : crop image and analyze derivatives of cropped image\n"
    cmdlst = cmdlst + "    q  : finished analysis return to main menu\n"

    crop_img = 0

    strin = raw_input (prompt);
    while (strin.lower() != "q"):    
        if (strin.lower() == "v") :
            fig, ax = plt.subplots(nrows = 2, ncols = 1) 
            ax[0].imshow(image, aspect='equal', extent=[0,cols*xscal,0,rows], cmap='gray')
            n, bins = np.histogram(image, bins=100)
            print '# bins: ',bins.shape,' Min bin ',np.min(bins),' Max bin : ',np.max(bins)
            ax[1].plot(bins[1:],n)
            plt.show()
            clip = input('Select clip level for masking image (0.0 to 1.0): ')
            mask_img = np.copy(image)
            mask_img[mask_img<=clip] = 0.0
            plt.imshow(mask_img, aspect='equal', extent=[0,cols*xscal,0,rows], cmap='gray')
            plt.show()
        elif (strin.lower() == "c") :            
            print 'Input for cropping (type 0 to use previous values)'
            l_x = input("Enter lower left  hand X value: ")
            l_y = input("Enter lower left  hand Y value: ")
            h_x = input("Enter upper right hand X value: ")
            h_y = input("Enter upper right hand Y value: ")
            if (l_x >0) : low_x = l_x
            if (l_y >0) : low_y = l_y
            if (h_x >0) : hi_x  = h_x
            if (h_y >0) : hi_y  = h_y
           
            lx = np.uint16(low_x/xscal)
            ly = rows - np.uint16(hi_y)
            hx = np.uint16(hi_x/xscal)
            hy = rows - np.uint16(low_y)
            crop_img = image[ly:hy,lx:hx]
            #crop_img = (crop_img - np.min(crop_img))/(np.max(crop_img) - np.min(crop_img))
            
            crop_row,crop_col = crop_img.shape
            print 'size of cropped image (rows,cols) : ',crop_row,' by ',crop_col
            print 'Max inten: ',np.max(crop_img),' Min intens: ',np.min(crop_img)
            
            clip = input('Select intensity level > to mask image (0.0 to 1.0): ')
            
            fig, ax = plt.subplots(nrows = 3, ncols = 3) 
            ax[0,0].set_title('Raw Cropped Image')
            ax[0,0].imshow(crop_img, aspect='equal', extent=[0,crop_col*xscal,0,crop_row], cmap='gray')
            n, bins = np.histogram(crop_img, bins=100)
            norm = np.float16(np.add.accumulate(n))/np.sum(n)
            ax[0,1].set_title('Intensity Histogram')
            ax[0,1].plot(bins[1:],n)

            clip_lev = np.min(bins[norm>clip])
            mask_img = np.copy(crop_img)
            mask_img[mask_img<=clip] = 0.0          
            ax[0,2].set_title('Clipped Image (Pixel Fract > '+str(1-clip)+': Inten Level > '+str(round(clip_lev,3))+')')
            ax[0,2].imshow(mask_img, aspect='equal', extent=[0,crop_col*xscal,0,crop_row], cmap='gray')

            dx_array = abs(crop_img[:,1:]-crop_img[:,:-1])
            ax[1,0].set_title('dX derivative image')
            ax[1,0].imshow(dx_array, aspect='equal', extent=[0,crop_col*xscal,0,crop_row], cmap='gray')

            dx_sum = np.array([np.arange(crop_col-1),np.sum(dx_array,0)]).T
            ax[1,1].set_title('dX sum along Y axis (avg: '+str(round(np.mean(dx_sum[:,1]),1))+')')
            ax[1,1].plot(dx_sum[:,0],dx_sum[:,1])
            n, bins = np.histogram(np.sum(dx_array,0), bins=100)
            norm = np.float16(np.add.accumulate(n))/np.sum(n)
            ax[1,2].set_title('Cummulative Histogram of dX sum')
            ax[1,2].plot(bins[1:],norm)

            dy_array = abs(crop_img[1:,:]-crop_img[:-1,:])
            ax[2,0].set_title('dY derivative image')
            ax[2,0].imshow(dy_array, aspect='equal', extent=[0,crop_col*xscal,0,crop_row], cmap='gray')

            dy_sum = np.array([np.arange(crop_row-1),np.sum(dy_array,1)]).T
            ax[2,1].set_title('dY sum along X axis (avg: '+str(round(np.mean(np.sum(dy_array,1)),1))+')')
            ax[2,1].plot(dy_sum[:,0],dy_sum[:,1])

            n, bins = np.histogram(np.sum(dy_array,1), bins=100)
            norm = np.float16(np.add.accumulate(n))/np.sum(n)
            ax[2,2].set_title('Cummulative Histogram of dY sum')
            ax[2,2].plot(bins[1:],norm)            
            plt.show()

            dXdY_array = dx_array[:-1,:] + dy_array[:,:-1]
            plt.imshow(dXdY_array, aspect='equal', extent=[0,(crop_col-1)*xscal,0,(crop_row-1)], cmap='gray')
            plt.show()
            
        else :
            print cmdlst
        strin = raw_input (prompt);
    return crop_img

def calibrate_board(image,xscal,x0,y0,z0):
#
#   Function does the following
#       
#       Analyze chess board image to physical locations of each square
#       determine image rows/cols to physical distances and x/y scale factor
#

    (rows,cols) = image.shape
    if Eng_mode : print 'Size of full image: ',rows,',',cols

#       Chess board squares dimensions in inches
    no_xsqs = 8
    no_ysqs = 8
    sq_xd = 1.5
    sq_yd = 1.5 + 1.0/16.0/8
    board_border = 0.20

#   find absolute board corners and squares by looking at the bigger image

    lx = 275
    ly = nrows-4
    hx = 882
    hy = nrows-1910
    clip = 0.95

    crop_img = np.copy(image[hy:ly,lx:hx])
    crop_img = (crop_img - np.min(crop_img))/(np.max(crop_img) - np.min(crop_img))
            
    crop_row,crop_col = crop_img.shape          
    n, bins = np.histogram(crop_img, bins=100)
    dx_array = abs(crop_img[:,1:]-crop_img[:,:-1])
    dx_sum = np.sum(dx_array,0)
    
    if Eng_mode :
        fig, ax = plt.subplots(nrows = 3, ncols = 3)
        ax[0,0].imshow(crop_img, aspect='equal', extent=[0,crop_col*xscal,0,crop_row], cmap='gray')
        ax[0,0].set_title('Board Image')
        ax[0,1].plot(bins[1:],n)
        ax[0,1].set_title('Intensity Histogram')            
        ax[1,0].imshow(dx_array, aspect='equal', extent=[0,(crop_col-1)*xscal,0,crop_row], cmap='gray')
        ax[1,0].set_title('dX image')
        ax[1,1].plot(np.arange(crop_col-1),dx_sum)
        ax[1,1].set_title('dX sum along Y axis')
        
    n, bins = np.histogram(dx_sum, bins=100)
    norm = np.float16(np.add.accumulate(n))/np.sum(n)
    if Eng_mode :
        ax[1,2].set_title('Cummulative Histogram of dX sum')
        ax[1,2].plot(bins[1:],norm)
       
    clip_lev = np.min(bins[norm>clip])
    dX_peaks = np.ravel(lx+np.argwhere(dx_sum>clip_lev))
    delta_dX_peaks = dX_peaks[1:]-dX_peaks[:-1]
    threshold = int(0.9 * max(delta_dX_peaks))
    ranks_x_location1 = np.ravel(dX_peaks[np.argwhere(delta_dX_peaks>threshold)])
    ranks_x_location2 = np.ravel(dX_peaks[np.argwhere(delta_dX_peaks>threshold)+1])
    ranks_x_location  = (ranks_x_location1[:-1]+ranks_x_location2[1:])/2
    ranks_x_location  = np.append([ranks_x_location1[0]],[ranks_x_location])
    ranks_x_location  = np.append([ranks_x_location],[ranks_x_location2[-1]])
    if Eng_mode :
        print 'Clip : ',clip,' dX level > : ',clip_lev
        print 'Threshold : ',threshold
        print 'Col location of dX Peaks : ',dX_peaks,' Shape: ',dX_peaks.shape
        print 'Difference : ',dX_peaks[1:]-dX_peaks[:-1]
        print 'Selected Peaks: ',ranks_x_location
 
    dy_array = abs(crop_img[1:,:]-crop_img[:-1,:])
    dy_sum = np.sum(dy_array,1)
    if Eng_mode :
        ax[2,0].imshow(dy_array, aspect='equal', extent=[0,crop_col*xscal,0,(crop_row-1)], cmap='gray')
        ax[2,0].set_title('dY image')
        ax[2,1].plot(np.arange(crop_row-1),dy_sum)
        ax[2,1].set_title('dY sum along X axis')
    n, bins = np.histogram(dy_sum, bins=100)
    norm = np.float16(np.add.accumulate(n))/np.sum(n)
    if Eng_mode :
        ax[2,2].set_title('Cummulative Histogram of dY sum')
        ax[2,2].plot(bins[1:],norm)
    
    clip_lev = np.min(bins[norm>clip])
    dY_peaks = np.ravel(hy+np.argwhere(dy_sum>clip_lev))
    delta_dY_peaks = dY_peaks[1:]-dY_peaks[:-1]
    threshold = int(0.9 * max(delta_dY_peaks))
    ranks_y_location1 = np.ravel(dY_peaks[np.argwhere(delta_dY_peaks>threshold)])
    ranks_y_location2 = np.ravel(dY_peaks[np.argwhere(delta_dY_peaks>threshold)+1])
    ranks_y_location  = (ranks_y_location1[:-1]+ranks_y_location2[1:])/2
    ranks_y_location  = np.append([ranks_y_location1[0]],[ranks_y_location])
    ranks_y_location  = np.append([ranks_y_location],[ranks_y_location2[-1]])
    if Eng_mode :
        print 'Clip : ',clip,' dY level > : ',clip_lev
        print 'Threshold : ',threshold
        print 'Row location of dY Peaks : ',dY_peaks,' Shape: ',dY_peaks.shape
        print 'Difference : ',dY_peaks[1:]-dY_peaks[:-1]
        print 'Selected Peaks: ',ranks_y_location
    
        plt.show()
    
    A8_boardcorner_col = max(ranks_x_location)
    A8_boardcorner_row = min(ranks_y_location)
    H1_boardcorner_col = min(ranks_x_location)
    H1_boardcorner_row = max(ranks_y_location)
    col_inches = no_xsqs*sq_xd/(A8_boardcorner_col-H1_boardcorner_col)
    row_inches = no_ysqs*sq_yd/(H1_boardcorner_row-A8_boardcorner_row)
    xscal = col_inches / row_inches
    
    print 'Board Corners : ',H1_boardcorner_col,',',H1_boardcorner_row,',',A8_boardcorner_col,',',A8_boardcorner_row
    print 'col_inches : ', col_inches
    print 'row_inches : ', row_inches
    print 'xscal : ', xscal

    board_img = np.copy(image)
    board_img[A8_boardcorner_row:H1_boardcorner_row,H1_boardcorner_col:A8_boardcorner_col]=0.0
    board_img[ranks_y_location,:] = 1.0
    board_img[:,ranks_x_location] = 1.0
    
    plt.imshow(board_img, aspect='equal', extent=[0,cols*xscal,0,rows], cmap='gray')
    plt.show()

    if Eng_mode :
        plt.imshow(crop_img, aspect='equal', extent=[0,crop_col*xscal,0,crop_row], cmap='gray')
        plt.show()
    
    return

def absolute_coordinate_moves() :

#   User coordinates use the front right bottom corner as 0,0,0 while machine coordinates use back left bottom as 0,0,0
#   Move signs will be inverted for x and y axis when user moves are translated to machine moves
#   All calibrations units such as backlash are based on user coordinate convention
#   All move units are in inches,  scale factor used to convert inches to GRBL move units for CNC



    prompt = "Absolute Coordinate Move Command (? for list of commands) : "
    cmdlst = "List of commands: \n"
    cmdlst = cmdlst + "    x  : move in x direction\n"
    cmdlst = cmdlst + "    y  : move in y direction\n"
    cmdlst = cmdlst + "    z  : move in z direction\n"
    cmdlst = cmdlst + "    q  : finished return to main menu\n"

    strin = raw_input (prompt);
    while (strin.lower() != "q"):    
        if (strin.lower() == "x") : 
            xloc = input('Enter in X coordinate : (' + repr(round(loc_pre[0],2)) + '->' + repr(round(loc_cur[0],2)) + ') ')
            if xloc > x_limits[1] :
                xloc = x_limits[1]
                print 'Limit reached : ',xloc
            elif xloc < x_limits[0] :
                xloc = x_limits[0]
                print 'Limit reached : ',xloc 
            value = xloc - loc_cur[0]
            if   ( (loc_cur[0] < loc_pre[0]) & (value > 0) ) :
                value = value + x_backlash[0]
                print 'Backlash correction applied'
            elif ( (loc_cur[0] > loc_pre[0]) & (value < 0) ) :
                value = value - x_backlash[1]
                print 'Backlash correction applied'
            GRBLvalue = round(-value/x_scale,2)
            print 'GRBL value = ',repr(GRBLvalue) 
            ser.write('G91 G0 X'+repr(GRBLvalue)+'\r\n')
            reply = ser.readline()
            print 'CNC Reply back: ',reply,
            if (xloc - loc_cur[0]) != 0 :
                loc_pre[0] = loc_cur[0]
                loc_cur[0] = xloc
        elif (strin.lower() == "y") :
            yloc = input('Enter in Y coordinate : (' + repr(round(loc_pre[1],2)) + '->' + repr(round(loc_cur[1],2)) + ') ')
            if yloc > y_limits[1] :
                yloc = y_limits[1]
                print 'Limit reached : ',yloc                
            elif yloc < y_limits[0] :
                yloc = y_limits[0]
                print 'Limit reached : ',yloc   
            value = yloc - loc_cur[1]
            if   ( (loc_cur[1] < loc_pre[1]) & (value > 0) ) :
                value = value + y_backlash[0]
                print 'Backlash correction applied'                
            elif ( (loc_cur[1] > loc_pre[1]) & (value < 0) ) :
                value = value - y_backlash[1]
                print 'Backlash correction applied'
            GRBLvalue = round(-value/y_scale,2)
            ser.write('G91 G0 Y'+repr(GRBLvalue)+'\r\n')
            reply = ser.readline()
            print 'CNC Reply back: ',reply,
            if (yloc - loc_cur[1]) != 0 :
                loc_pre[1] = loc_cur[1]
                loc_cur[1] = yloc     
        elif (strin.lower() == "z") :
            zloc = input('Enter in Z coordinate : (' + repr(round(loc_pre[2],2)) + '->' + repr(round(loc_cur[2],2)) + ') ')
            if zloc > z_limits[1] :
                zloc = z_limits[1]
                print 'Limit reached : ',zloc   
            elif zloc < z_limits[0] :
                zloc = z_limits[0]
                print 'Limit reached : ',zloc 
            value = zloc - loc_cur[2]
            if   ( (loc_cur[2] < loc_pre[2]) & (value > 0) ) :
                value = value + z_backlash[0]
                print 'Backlash correction applied' 
            elif ( (loc_cur[2] > loc_pre[2]) & (value < 0) ) :
                value = value - z_backlash[1]
                print 'Backlash correction applied'                
            GRBLvalue = round(value/z_scale,2)
            ser.write('G91 G0 Z'+repr(GRBLvalue)+'\r\n')
            reply = ser.readline()
            print 'CNC Reply back: ',reply,
            if (zloc - loc_cur[2]) != 0 :
                loc_pre[2] = loc_cur[2]
                loc_cur[2] = zloc     
        else :
            print cmdlst
        strin = raw_input (prompt);
    return 

#   MAIN PROGRAM
#
#   Setup global variables and constants
#
#
#   To Run program in diagnostics mode set Eng_mode to True
#   For audio feedback set Talk_mode to True
#
Eng_mode = True
Talk_mode = False


#       CNC reliable operation motion limits in inches once initial location
#       has been calibrated correctly.  CNC will stop working and possible get
#       damaged if operated beyond these limits.
z_limits = [0.0,3.5]
x_limits = [0.0,15.0]
y_limits = [0.0,12.0]
#       CNC backlash correction in inches (neg. to pos. , pos. to neg)
z_backlash = [0.0, 0.0]
x_backlash = [0.06, 0.06]
y_backlash = [0.06, 0.06]
#       CNC GRBL value conversion to inches
x_scale = 0.97/25.4
y_scale = 1.01/25.4
z_scale = 0.95/25.5
#       Vertical Distance (inches) of Pi Camera above center of Chess board
camera_height = 24.0 + 1.0/8.0
    
#z0 = input('Current Z location (' + repr(round(z_limits[0],1)) + ' <-> '+repr(round(z_limits[1],1)) + ') : ')
#x0 = input('Current X location (' + repr(round(x_limits[0],1)) + ' <-> '+repr(round(x_limits[1],1)) + ') : ')
#y0 = input('Current Y location (' + repr(round(y_limits[0],1)) + ' <-> '+repr(round(y_limits[1],1)) + ') : ')
x0 = 13.5
y0 = 5.0
z0 = 2.5

loc_cur = [x0,y0,z0]
loc_pre = [0,0,0]
col_inches = 1.0
row_inches = 1.0
zero_location_col = 0
zero_location_row = 0

#   GPIO setup
#       use board numbering of the GPIO pins
#       turn off warnings
#       setup pin 38 for output mode to control MOSFET to turn on/off electromagnet
#       set pin 38 to be initally at low value or in the OFF state for the electromagnet
#       use pin 18 to turn on LED at same time as magnet to indicate magnet is ON

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(18, GPIO.OUT,initial = False)
GPIO.setup(38, GPIO.OUT,initial = False)

#   Initialize camera
#   setup camera orientation
#   setup preview window size and location
#       preview window parms(xloc, yloc, width, height)
#   select camera resolution
#       aspect ratio of 4:3 with full FOV
#           ncols = 480
#           nrows = 640
#           col_scal = 1.77778 (16/9)
#       aspect ratio of 4:3 with full FOV
#           ncols = 1944
#           nrows = 2592
#       aspect ratio of 16:9 with partial FOV
#           ncols = 1080
#           nrows = 1920
#           col_scal = 3.0 

nrows = 1920
ncols = 1088
col_scal = 3
camera = picamera.PiCamera()
camera.rotation = 0
camera.preview_fullscreen = False
#camera.preview_window =(0, 240, 960, 720)
camera.preview_window =(60, 540, 720, 540)
camera.resolution = (nrows,ncols)

#   Set up Arduino link to control CNC machine
#       Use USB at 115200 baud
#       ACM0 is USB port for Arduino

ser = serial.Serial('/dev/ttyACM0',115200)

if Talk_mode :
    talk('Hello.   I am Hal the chess playing cyclops')
    talk('Prepare to be crushed')

prompt = "Main Command (? for list of commands) : "
cmdlst =          "    cal  : auto calibrate board and image scales\n"
cmdlst = cmdlst + "    arm  : calibrate arm location\n"
cmdlst = cmdlst + "    mo   : move magnet using abolute coordiates\n"
cmdlst = cmdlst + "    on   : turn on magnet\n"
cmdlst = cmdlst + "    off  : turn off magnet\n"
cmdlst = cmdlst + "    sq   : move to a particular chess square\n"
cmdlst = cmdlst + "    im   : to capture and save JPG image\n"
cmdlst = cmdlst + "    grid : capture image and perform auto detect of chess board squares\n"
cmdlst = cmdlst + "    an   : capture image for manual board analysis\n"
cmdlst = cmdlst + "    g    : enter in GRBL command to CNC\n"
cmdlst = cmdlst + "    q    : to quit program"

#   view video until quit command is entered for Arduino

camera.start_preview()
time.sleep(2)
camera.capture('binary.rgb',format = 'rgb', resize = (ncols,nrows))
image = np.fromfile('binary.rgb',np.uint8, -1, '')
image = image.reshape(nrows,ncols,3)
inten = np.float16(image[:,:,0]) + np.float16(image[:,:,1]) + np.float16(image[:,:,2])
inten = (inten-np.min(inten))/(np.max(inten) - np.min(inten))

strin = raw_input (prompt);
while (strin.lower() != "q"):
    if (strin.lower() == "im") :
        filename = raw_input ("Name of saved image file? ")
        camera.capture(filename + '.jpg')   # store image as JPEG
    elif (strin.lower() == "?") :
        print cmdlst
    elif (strin.lower() == "cal") :
        if Talk_mode : talk('OK, will go into calibration mode')
        camera.stop_preview()
        calibrate_board(inten,col_scal,x0,y0,z0)
        camera.start_preview()
    elif (strin.lower() == "on") :
        if Talk_mode : talk('Turning on magnet')
        GPIO.output(38, True)
        GPIO.output(18, True)
    elif (strin.lower() == "off") :
        if Talk_mode : talk('Turning off magnet')
        GPIO.output(38, False)
        GPIO.output(18, False)
    elif (strin.lower() == "an") :
        if Talk_mode : talk('Capturing binary image and storing data')
        camera.capture('binary.rgb',format = 'rgb', resize = (ncols,nrows))  
        image = np.fromfile('binary.rgb',np.uint8, -1, '')
        image = image.reshape(nrows,ncols,3)
        inten = np.float16(image[:,:,0]) + np.float16(image[:,:,1]) + np.float16(image[:,:,2])
        inten = (inten-np.min(inten))/(np.max(inten) - np.min(inten))
        camera.stop_preview()
        board = board_analysis(inten, col_scal)
        camera.start_preview()
    elif (strin.lower() == "mo") :
        if Talk_mode : talk('Entering into absolute coordinate move mode')
        absolute_coordinate_moves()
    elif (strin.lower() == "g") :
        if Talk_mode : talk('Now in direct GRBL command mode')
        strin = raw_input('Type in GRBL command (e.g X5): ')
        ser.write('G91 G0 '+strin+'\r\n')
        reply = ser.readline()
        print 'GRBL Reply back: ',reply,
    else :
        if Talk_mode : talk('Do not understand command' + prompt)
        print prompt
    strin = raw_input (prompt);                                                             

#   stop camera
#   release all resource used by camers
#   release digital I/O ports

camera.stop_preview()
camera.close()      
GPIO.cleanup()      
