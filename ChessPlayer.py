
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
import math

def talk(cmd):
    saycmd = "echo " + cmd + " | festival --tts"
    os.system(saycmd)
    return

def find_laser (yscal) :
#
#   Find location of laser in image usings of rows and cols
#

    global Image, Inten
    global Laser_row, Laser_col

#
#   inital box size
#
    ly = 200
    hy = 1087
    lx = 0
    hx = 1919
    clip = 0.90
#
#   laser spot detection parameters
#
    min_no_pixels = 50
    max_no_pixels = 250
    max_avg_distance = 30.0
    window_radius = 100.0
    
    crop_img = np.copy(Inten[lx:hx,ly:hy])       
    crop_row,crop_col = crop_img.shape
    mask_img = np.copy(crop_img)
    mask_img[mask_img<=clip] = 0.0
    mask_img[mask_img>clip ] = 1.0
    crop_img = np.copy(Inten[lx:hx,ly:hy])
    
    pixels = np.argwhere(crop_img>clip)
    marks = np.median(pixels,0)
    no_pixels = np.sum(np.sum(mask_img))
    pixel_std = np.std(pixels,axis = 0)

    if Eng_mode :       
        print '# of pixs ',no_pixels,' x std ',pixel_std[0],' y std ',pixel_std[1]*yscal
        print 'size of crop image : ',crop_row,',',crop_col
        print 'Marks : (',marks.shape,'),',marks 

    locs = pixels*1.0
    locs[:,1] = locs[:,1]*yscal

    n_pixels = np.shape(locs)[0]
    locs_mat1x = np.outer(locs[:,0],np.ones(n_pixels))
    locs_mat2x = np.transpose(locs_mat1x)
    locs_mat1y = np.outer(locs[:,1],np.ones(n_pixels))
    locs_mat2y = np.transpose(locs_mat1y)
    dist_mat   = ((locs_mat1x-locs_mat2x)**2 + (locs_mat1y-locs_mat2y)**2)**0.5
    dist_mat[dist_mat > window_radius] = 0.0
    count_mat  = np.copy(dist_mat)
    count_mat[count_mat > 0] = 1.0
    sum_cnt = np.reshape(np.sum(count_mat,axis=1),[n_pixels,1])
    sum_dst = np.reshape(np.sum(dist_mat,axis=1),[n_pixels,1])

    pixels = np.append(pixels,sum_cnt, axis=1)
    pixels = np.append(pixels,sum_dst, axis=1)

    laser_pixels = pixels[np.argwhere((pixels[:,2]<max_no_pixels) & (pixels[:,2]>min_no_pixels))]

    laser_pixels[:,0,3] = laser_pixels[:,0,3] / laser_pixels[:,0,2]
    laser_pixels = laser_pixels[np.argwhere(laser_pixels[:,0,3]< max_avg_distance)]

    laser_row = np.median(laser_pixels[:,:,:,0])
    laser_col = np.median(laser_pixels[:,:,:,1])
    no_pixels = np.shape(laser_pixels)[0]
    
    avg_dist = np.sum(laser_pixels, axis=0)[0,0,3] / no_pixels

    if Eng_mode :
        print '# of pixs ',no_pixels,' avg. distance', avg_dist    
        print 'Laser mark : ',laser_row,' , ',laser_col
    
        fig, ax = plt.subplots(nrows = 1, ncols = 2)
        ax[0].imshow(crop_img, aspect='equal', extent=[0,crop_col*yscal,0,crop_row], cmap='gray')      
        ax[1].set_title('Laser Spot: ('+str(round(laser_col*yscal,1))+','+str(round(crop_row-laser_row,1))+')')   
        ax[1].imshow(mask_img, aspect='equal', extent=[0,crop_col*yscal,0,crop_row], cmap='gray')        
        plt.show()

    laser_row += lx
    laser_col += ly
    
    if Eng_mode :
        print 'Laser mark : ',laser_row,' , ',laser_col
    
    return laser_row,laser_col

def board_analysis(yscal):

#
#   allows manual analysis of images
#

    global Image, Inten
    global Nrows, Ncols

    (rows,cols) = Inten.shape

    low_x = 0
    low_y = 0
    hi_x = int(cols*yscal)
    hi_y = rows
    
    prompt = "Analysis Mode Command (? for list of commands) : "
    cmdlst = "List of commands: \n"
    cmdlst = cmdlst + "    v  : capture new Image and view the full Image in intensity scale\n"
    cmdlst = cmdlst + "    vr  : capture new Image and view the full Image in red scale\n"
    cmdlst = cmdlst + "    vg  : capture new Image and view the full Image in green scaled\n"
    cmdlst = cmdlst + "    vb  : capture new Image and view the full Image in blue scaled\n"
    cmdlst = cmdlst + "    l  : set threshold level for existing Image analysis\n"
    cmdlst = cmdlst + "    c  : crop existing Image and analyze derivatives of cropped Image\n"
    cmdlst = cmdlst + "    fl : find laser mark in image\n"
    cmdlst = cmdlst + "    q  : finished analysis return to main menu\n"

    crop_img = 0

    strin = raw_input (prompt);
    while (strin.lower() != "q"):    
        if (strin.lower() == "v") :
            capture_Image(0)
            plt.imshow(Inten, aspect='equal', extent=[0,cols*yscal,0,rows], cmap='gray')
            plt.show()
        elif (strin.lower() == "vr") :
            capture_Image(1)
            plt.imshow(Inten, aspect='equal', extent=[0,cols*yscal,0,rows], cmap='Reds_r')
            plt.show()
        elif (strin.lower() == "vg") :
            capture_Image(2)
            plt.imshow(Inten, aspect='equal', extent=[0,cols*yscal,0,rows], cmap='Greens_r')
            plt.show()
        elif (strin.lower() == "vb") :
            capture_Image(3)
            plt.imshow(Inten, aspect='equal', extent=[0,cols*yscal,0,rows], cmap='Blues_r')
            plt.show()
        elif (strin.lower() == "fl") :
            laser_row,laser_col = find_laser(yscal)
        elif(strin.lower() == "l") :
            fig, ax = plt.subplots(nrows = 2, ncols = 1) 
            ax[0].imshow(Inten, aspect='equal', extent=[0,cols*yscal,0,rows], cmap='gray')
            n, bins = np.histogram(Inten, bins=100)
            print '# bins: ',bins.shape,' Min bin ',np.min(bins),' Max bin : ',np.max(bins)
            ax[1].plot(bins[1:],n)
            plt.show()
            clip = input('Select clip level for masking Image (0.0 to 1.0): ')
            mask_img = np.copy(Inten)
            mask_img[mask_img<=clip] = 0.0
            plt.imshow(mask_img, aspect='equal', extent=[0,cols*yscal,0,rows], cmap='gray')
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
           
            lx = np.uint16(low_x/yscal)
            ly = rows - np.uint16(hi_y)
            hx = np.uint16(hi_x/yscal)
            hy = rows - np.uint16(low_y)
            crop_img = Inten[ly:hy,lx:hx]
            #crop_img = (crop_img - np.min(crop_img))/(np.max(crop_img) - np.min(crop_img))
            
            crop_row,crop_col = crop_img.shape
            print 'size of cropped Image (rows,cols) : ',crop_row,' by ',crop_col
            print 'Max Inten: ',np.max(crop_img),' Min Intens: ',np.min(crop_img)
            
            clip = input('Select Intensity level > to mask Image (0.0 to 1.0): ')
            
            fig, ax = plt.subplots(nrows = 3, ncols = 3) 
            ax[0,0].set_title('Raw Cropped Image')
            ax[0,0].imshow(crop_img, aspect='equal', extent=[0,crop_col*yscal,0,crop_row], cmap='gray')
            n, bins = np.histogram(crop_img, bins=100)
            norm = np.float16(np.add.accumulate(n))/np.sum(n)
            ax[0,1].set_title('Intensity Histogram')
            ax[0,1].plot(bins[1:],n)

            clip_lev = np.min(bins[norm>clip])
            mask_img = np.copy(crop_img)
            mask_img[mask_img<=clip] = 0.0          
            ax[0,2].set_title('Clipped Image (Pixel Fract > '+str(1-clip)+': Inten Level > '+str(round(clip_lev,3))+')')
            ax[0,2].imshow(mask_img, aspect='equal', extent=[0,crop_col*yscal,0,crop_row], cmap='gray')

            dx_array = abs(crop_img[:,1:]-crop_img[:,:-1])
            ax[1,0].set_title('dX derivative Image')
            ax[1,0].imshow(dx_array, aspect='equal', extent=[0,crop_col*yscal,0,crop_row], cmap='gray')

            dx_sum = np.array([np.arange(crop_col-1),np.sum(dx_array,0)]).T
            ax[1,1].set_title('dX sum along Y axis (avg: '+str(round(np.mean(dx_sum[:,1]),1))+')')
            ax[1,1].plot(dx_sum[:,0],dx_sum[:,1])
            n, bins = np.histogram(np.sum(dx_array,0), bins=100)
            norm = np.float16(np.add.accumulate(n))/np.sum(n)
            ax[1,2].set_title('Cummulative Histogram of dX sum')
            ax[1,2].plot(bins[1:],norm)

            dy_array = abs(crop_img[1:,:]-crop_img[:-1,:])
            ax[2,0].set_title('dY derivative Image')
            ax[2,0].imshow(dy_array, aspect='equal', extent=[0,crop_col*yscal,0,crop_row], cmap='gray')

            dy_sum = np.array([np.arange(crop_row-1),np.sum(dy_array,1)]).T
            ax[2,1].set_title('dY sum along X axis (avg: '+str(round(np.mean(np.sum(dy_array,1)),1))+')')
            ax[2,1].plot(dy_sum[:,0],dy_sum[:,1])

            n, bins = np.histogram(np.sum(dy_array,1), bins=100)
            norm = np.float16(np.add.accumulate(n))/np.sum(n)
            ax[2,2].set_title('Cummulative Histogram of dY sum')
            ax[2,2].plot(bins[1:],norm)            
            plt.show()

            dXdY_array = dx_array[:-1,:] + dy_array[:,:-1]
            plt.imshow(dXdY_array, aspect='equal', extent=[0,(crop_col-1)*yscal,0,(crop_row-1)], cmap='gray')
            plt.show()
            
        else :
            print cmdlst
        strin = raw_input (prompt);
    return crop_img

def calibrate_board(yscal):
#
#   Function does the following
#       
#       Analyze chess board Image to physical locations of each square
#       determine Image rows/cols to physical distances and x/y scale factor
#
#
#   Naming Convention
#       Physical Y Axis = Image Column Axis = Chess Board Rank Axis
#       Physical X Axis = Image Row Axis    = Chess Board File Axis
#
    global Image, Inten
    global X0,Y0,Z0
    global Laser_x_loc, Laser_y_loc
    global Row_inches, Col_inches
    global Zero_location_col
    global Zero_location_row

    (rows,cols) = Inten.shape
    if Eng_mode : print 'Size of full Image: ',rows,',',cols

#       Chess board squares dimensions in inches
    no_xsqs = 8
    no_ysqs = 8
    sq_yd = 1.5
    sq_xd = 1.5 + 1.0/16.0/8
    board_border = 0.20

#   find absolute board corners and squares by looking at the bigger Image

    ly = 275
    hy = 882
    lx = 4
    hx = 1910
    clip = 0.97

    crop_img = np.copy(Inten[lx:hx,ly:hy])
    crop_img = (crop_img - np.min(crop_img))/(np.max(crop_img) - np.min(crop_img))
            
    crop_row,crop_col = crop_img.shape          
    n, bins = np.histogram(crop_img, bins=100)
    dy_array = abs(crop_img[:,1:]-crop_img[:,:-1])
    dy_sum = np.sum(dy_array,0)
    
    if Eng_mode :
        fig, ax = plt.subplots(nrows = 3, ncols = 3)
        ax[0,0].imshow(crop_img, aspect='equal', extent=[0,crop_col*yscal,0,crop_row], cmap='gray')
        ax[0,0].set_title('Chess Board Image')
        ax[0,1].plot(bins[1:],n)
        ax[0,1].set_title('Intensity Histogram')            
        ax[1,0].imshow(dy_array, aspect='equal', extent=[0,(crop_col-1)*yscal,0,crop_row], cmap='gray')
        ax[1,0].set_title('dY Image')
        ax[1,1].plot(np.arange(crop_col-1),dy_sum)
        ax[1,1].set_title('peaks are Col/Rank Boundaries')
        
    n, bins = np.histogram(dy_sum, bins=100)
    norm = np.float16(np.add.accumulate(n))/np.sum(n)
    if Eng_mode :
        ax[1,2].set_title('Cummulative Histogram of dY sum')
        ax[1,2].plot(bins[1:],norm)
       
    clip_lev = np.min(bins[norm>clip])
    dy_peaks = np.ravel(ly+np.argwhere(dy_sum>clip_lev))
    delta_dy_peaks = dy_peaks[1:]-dy_peaks[:-1]
    threshold = int(0.9 * max(delta_dy_peaks))
    squares_y_location1 = np.ravel(dy_peaks[np.argwhere(delta_dy_peaks>threshold)])
    squares_y_location2 = np.ravel(dy_peaks[np.argwhere(delta_dy_peaks>threshold)+1])
    squares_y_location  = (squares_y_location1[:-1]+squares_y_location2[1:])/2
    squares_y_location  = np.append([squares_y_location1[0]],[squares_y_location])
    squares_y_location  = np.append([squares_y_location],[squares_y_location2[-1]])
    if Eng_mode :
        print 'Clip : ',clip,' dY level > : ',clip_lev
        print 'Threshold : ',threshold
        print 'Col location of Rank Boundary Peaks : ',dy_peaks,' Shape: ',dy_peaks.shape
        print 'Difference : ',dy_peaks[1:]-dy_peaks[:-1]
        print 'Selected Peaks: ',squares_y_location
        print '# of Ranks detected: ',len(squares_y_location)-1
 
    dx_array = abs(crop_img[1:,:]-crop_img[:-1,:])
    dx_sum = np.sum(dx_array,1)
    if Eng_mode :
        ax[2,0].imshow(dx_array, aspect='equal', extent=[0,crop_col*yscal,0,(crop_row-1)], cmap='gray')
        ax[2,0].set_title('dX Image')
        ax[2,1].plot(np.arange(crop_row-1),dx_sum)
        ax[2,1].set_title('peaks are File Boundaries')
    n, bins = np.histogram(dx_sum, bins=100)
    norm = np.float16(np.add.accumulate(n))/np.sum(n)
    if Eng_mode :
        ax[2,2].set_title('Cummulative Histogram of dX sum')
        ax[2,2].plot(bins[1:],norm)
    
    clip_lev = np.min(bins[norm>clip])
    dx_peaks = np.ravel(lx+np.argwhere(dx_sum>clip_lev))
    delta_dx_peaks = dx_peaks[1:]-dx_peaks[:-1]
    threshold = int(0.9 * max(delta_dx_peaks))
    squares_x_location1 = np.ravel(dx_peaks[np.argwhere(delta_dx_peaks>threshold)])
    squares_x_location2 = np.ravel(dx_peaks[np.argwhere(delta_dx_peaks>threshold)+1])
    squares_x_location  = (squares_x_location1[:-1]+squares_x_location2[1:])/2
    squares_x_location  = np.append([squares_x_location1[0]],[squares_x_location])
    squares_x_location  = np.append([squares_x_location],[squares_x_location2[-1]])
    if Eng_mode :
        print 'Clip : ',clip,' dX level > : ',clip_lev
        print 'Threshold : ',threshold
        print 'Row location of Row/File Boundary Peaks : ',dx_peaks,' Shape: ',dx_peaks.shape
        print 'Difference : ',dx_peaks[1:]-dx_peaks[:-1]
        print 'Selected Peaks: ',squares_x_location
        print '# of Files detected: ',len(squares_x_location)-1
    
        plt.show()
    
    A8_boardcorner_col = max(squares_y_location)
    A8_boardcorner_row = min(squares_x_location)
    H1_boardcorner_col = min(squares_y_location)
    H1_boardcorner_row = max(squares_x_location)
    Col_inches = no_ysqs*sq_yd/(A8_boardcorner_col-H1_boardcorner_col)
    Row_inches = no_xsqs*sq_xd/(H1_boardcorner_row-A8_boardcorner_row)
    yscal = Col_inches / Row_inches

    files_c = np.float16(squares_x_location[1:]+squares_x_location[:-1])*0.5
    ranks_c = np.float16(squares_y_location[1:]+squares_y_location[:-1])*0.5

    if Eng_mode :   
        print 'File centers : ', files_c   
        print 'Rank centers : ', ranks_c
       
    files_c = np.resize(files_c, (8,8)).T
    ranks_c = np.resize(ranks_c, (8,8))

    Board[:,:,0,0] = files_c
    Board[:,:,0,1] = ranks_c
    Board[:,:,1,0] = (files_c - A8_boardcorner_row) * Row_inches
    Board[:,:,1,1] = (ranks_c - H1_boardcorner_col) * Col_inches

    Zero_location_row = A8_boardcorner_row
    Zero_location_col = H1_boardcorner_col
    
    if Eng_mode :   
        print 'A file ', Board[0,:,:,:]
        print 'H file ', Board[-1,:,:,:]
        print '1st rank ', Board[:,0,:,:]
        print '8th rank ', Board[:,-1,:,:]

        print 'Board Corners : ',H1_boardcorner_col,',',H1_boardcorner_row,',',A8_boardcorner_col,',',A8_boardcorner_row
        print 'Col_inches : ', Col_inches
        print 'Row_inches : ', Row_inches
        print 'yscal : ', yscal

    board_img = np.copy(Inten)
    board_img[A8_boardcorner_row:H1_boardcorner_row,H1_boardcorner_col:A8_boardcorner_col]=0.0
    board_img[squares_x_location,:] = 1.0
    board_img[:,squares_y_location] = 1.0
    
    if Eng_mode :
        plt.imshow(board_img, aspect='equal', extent=[0,cols*yscal,0,rows], cmap='gray')
        plt.show()
        plt.imshow(crop_img, aspect='equal', extent=[0,crop_col*yscal,0,crop_row], cmap='gray')
        plt.show()
    
    return

def calibrate_arm_location(yscale) :

    global Loc_cur
    global X0,Y0,Z0
    global Magnet2laser_offset
    global Magnet_actual_loc
    global Camera_height
    global Laser_height_0
    global Row_inches, Col_inches
    global Zero_location_col
    global Zero_location_row

#
#   turn on laser and capture image
#
    GPIO.output(22, True)
    capture_Image(1)
#
#   find laser spot
#
    laser_row,laser_col = find_laser(yscale)

#    print 'Laser row,col : {0:.3f} , {1:.3f}'.format(laser_row,laser_col)
    
    laser_x_loc = (laser_row-Zero_location_row) * Row_inches
    laser_y_loc = (laser_col-Zero_location_col) * Col_inches

#    print 'Laser location (x,y) {0:.3f} , {1:.3f} : '.format(laser_x_loc,laser_y_loc)
#
#   confirm Z height of laser
#
    laser_ht = input('Height of laser above board plane (inches): ')
    Z0 = laser_ht - Laser_height_0
#
#   calculate X0 and Y0 from geometry of laser location
#
    theta = math.pi*0.5-math.acos(Camera_height/(Camera_height**2+(laser_x_loc-6)**2+(laser_y_loc-6)**2)**0.5)
    
    if (laser_y_loc < 6) :
        theta = -theta
    phi   = math.atan((laser_x_loc-6)/(laser_y_loc-6))

#    print 'theta, phi : {0:.3f}, {1:.3f}'.format(theta,phi)
    
    if theta == math.pi*0.5 :
        shadow = 0
    else :
        shadow = laser_ht / math.tan(theta)

#    print 'shadow : {0:.3f}'.format(shadow)
    
    laser_actual_loc = Magnet_actual_loc
    laser_actual_loc[0] = laser_x_loc - shadow*math.sin(phi)
    laser_actual_loc[1] = laser_y_loc - shadow*math.cos(phi)
    laser_actual_loc[2] = 0.0

#    print 'magnet loc ',Magnet_actual_loc
#    print 'mag2laser offset ',Magnet2laser_offset
    
    Magnet_actual_loc = np.add(laser_actual_loc, Magnet2laser_offset)

#    print 'magnet loc ',Magnet_actual_loc
    
    Magnet_actual_loc[2] = Magnet_actual_loc[2] + laser_ht
    X0 = Magnet_actual_loc[0]
    Y0 = Magnet_actual_loc[1]

    print 'Calculated location of magnet (x,y,z) : {0:.3f} , {1:.3f} , {2:.3f} '.format(Magnet_actual_loc[0],Magnet_actual_loc[1],Magnet_actual_loc[2])

#    Z0 = input('Current Z location (' + repr(round(Z_limits[0],1)) + ' <-> '+repr(round(Z_limits[1],1)) + ') : ')    
#    X0 = input('Current X location (' + repr(round(X_limits[0],1)) + ' <-> '+repr(round(X_limits[1],1)) + ') : ')
#    Y0 = input('Current Y location (' + repr(round(Y_limits[0],1)) + ' <-> '+repr(round(Y_limits[1],1)) + ') : ')

    Loc_cur = [X0,Y0,Z0]

#
#   turn off laser
#
    GPIO.output(22, False)

    return

def absolute_coordinate_moves() :

#   User coordinates use the front left bottom corner as 0,0,0 while machine coordinates use back left bottom as 0,0,0
#   Move signs will be inverted for y axis when user moves are translated to machine moves
#   All calibrations units such as backlash are based on user coordinate convention
#   All move units are in inches,  scale factor used to convert inches to GRBL move units for CNC



    prompt = "Absolute Coordinate Move Command (? for list of commands) : "
    cmdlst = "List of commands: \n"
    cmdlst = cmdlst + "    x  : move in x direction\n"
    cmdlst = cmdlst + "    y  : move in y direction\n"
    cmdlst = cmdlst + "    z  : move in z direction\n"
    cmdlst = cmdlst + "    sq : move to chess square\n"
    cmdlst = cmdlst + "    q  : finished return to main menu\n"

    file_list = 'abcdefgh'
    rank_list = '12345678'
    
    strin = raw_input (prompt);
    while (strin.lower() != "q"):    
        if (strin.lower() == "x") : 
            xloc = input('Enter in X coordinate : (' + repr(round(Loc_pre[0],2)) + '->' + repr(round(Loc_cur[0],2)) + ') ')
            if xloc > X_limits[1] :
                xloc = X_limits[1]
                print 'Limit reached : ',xloc
            elif xloc < X_limits[0] :
                xloc = X_limits[0]
                print 'Limit reached : ',xloc 
            value = xloc - Loc_cur[0]
            if   ( (Loc_cur[0] < Loc_pre[0]) & (value > 0) ) :
                value = value + X_backlash[0]
                print 'Backlash correction applied'
            elif ( (Loc_cur[0] > Loc_pre[0]) & (value < 0) ) :
                value = value - X_backlash[1]
                print 'Backlash correction applied'
            GRBLvalue = round(value/X_scale,2)
            print 'GRBL value = ',repr(GRBLvalue) 
            ser.write('G91 G0 Y'+repr(GRBLvalue)+'\r\n')
            reply = ser.readline()
            print 'CNC Reply back: ',reply,
            if (xloc - Loc_cur[0]) != 0 :
                Loc_pre[0] = Loc_cur[0]
                Loc_cur[0] = xloc
        elif (strin.lower() == "y") :
            yloc = input('Enter in Y coordinate : (' + repr(round(Loc_pre[1],2)) + '->' + repr(round(Loc_cur[1],2)) + ') ')
            if yloc > Y_limits[1] :
                yloc = Y_limits[1]
                print 'Limit reached : ',yloc                
            elif yloc < Y_limits[0] :
                yloc = Y_limits[0]
                print 'Limit reached : ',yloc   
            value = yloc - Loc_cur[1]
            if   ( (Loc_cur[1] < Loc_pre[1]) & (value > 0) ) :
                value = value + Y_backlash[0]
                print 'Backlash correction applied'                
            elif ( (Loc_cur[1] > Loc_pre[1]) & (value < 0) ) :
                value = value - Y_backlash[1]
                print 'Backlash correction applied'
            GRBLvalue = round(-value/Y_scale,2)
            ser.write('G91 G0 X'+repr(GRBLvalue)+'\r\n')
            reply = ser.readline()
            print 'CNC Reply back: ',reply,
            if (yloc - Loc_cur[1]) != 0 :
                Loc_pre[1] = Loc_cur[1]
                Loc_cur[1] = yloc     
        elif (strin.lower() == "z") :
            zloc = input('Enter in Z coordinate : (' + repr(round(Loc_pre[2],2)) + '->' + repr(round(Loc_cur[2],2)) + ') ')
            if zloc > Z_limits[1] :
                zloc = Z_limits[1]
                print 'Limit reached : ',zloc   
            elif zloc < Z_limits[0] :
                zloc = Z_limits[0]
                print 'Limit reached : ',zloc 
            value = zloc - Loc_cur[2]
            if   ( (Loc_cur[2] < Loc_pre[2]) & (value > 0) ) :
                value = value + Z_backlash[0]
                print 'Backlash correction applied' 
            elif ( (Loc_cur[2] > Loc_pre[2]) & (value < 0) ) :
                value = value - Z_backlash[1]
                print 'Backlash correction applied'                
            GRBLvalue = round(value/Z_scale,2)
            ser.write('G91 G0 Z'+repr(GRBLvalue)+'\r\n')
            reply = ser.readline()
            print 'CNC Reply back: ',reply,
            if (zloc - Loc_cur[2]) != 0 :
                Loc_pre[2] = Loc_cur[2]
                Loc_cur[2] = zloc     
        elif (strin.lower() == "sq") :
            square = raw_input('Which chess square to move to? (e.g a1 or f3) ')
            square = square.lower()

            if len(square) == 2 :                
                if square[0] in file_list :
                    file_n = file_list.index(square[0])
                if square[1] in rank_list :
                    rank_n = rank_list.index(square[1])
            
                xloc = Board[file_n,rank_n,1,0]
                yloc = Board[file_n,rank_n,1,1]

                print 'Moving to ',square,' located at (',xloc,',',yloc,')'
            
                if yloc > Y_limits[1] :
                    yloc = Y_limits[1]
                    print 'Limit reached : ',yloc                
                elif yloc < Y_limits[0] :
                    yloc = Y_limits[0]
                    print 'Limit reached : ',yloc   
                value = yloc - Loc_cur[1]
                if   ( (Loc_cur[1] < Loc_pre[1]) & (value > 0) ) :
                    value = value + Y_backlash[0]
                    print 'Backlash correction applied'                
                elif ( (Loc_cur[1] > Loc_pre[1]) & (value < 0) ) :
                    value = value - Y_backlash[1]
                    print 'Backlash correction applied'
                GRBLy = round(-value/Y_scale,2)
    
                if xloc > X_limits[1] :
                    xloc = X_limits[1]
                    print 'Limit reached : ',xloc
                elif xloc < X_limits[0] :
                    xloc = X_limits[0]
                    print 'Limit reached : ',xloc 
                value = xloc - Loc_cur[0]
                if   ( (Loc_cur[0] < Loc_pre[0]) & (value > 0) ) :
                    value = value + X_backlash[0]
                    print 'Backlash correction applied'
                elif ( (Loc_cur[0] > Loc_pre[0]) & (value < 0) ) :
                    value = value - X_backlash[1]
                    print 'Backlash correction applied'
                GRBLx = round(value/X_scale,2)
            
                ser.write('G91 G0 X'+repr(GRBLy)+' Y'+repr(GRBLx)+'\r\n')
                reply = ser.readline()
                print 'CNC Reply back: ',reply,
                if (yloc - Loc_cur[1]) != 0 :
                    Loc_pre[1] = Loc_cur[1]
                    Loc_cur[1] = yloc     
                if (xloc - Loc_cur[0]) != 0 :
                    Loc_pre[0] = Loc_cur[0]
                    Loc_cur[0] = xloc
            else :
                print 'Invalid square : ',square
        else :
            print cmdlst
        strin = raw_input (prompt);
    return

def capture_Image (img_type) :

#
#   captures a new Image and returns a normalized Image file
#
#       img_type = 0, normalized intensity image
#       img_type = 1, normalized red color image
#       img_type = 2, normalized green color image
#       img_type = 3, normalized blue color image
#
#
#
    global Image, Inten
    global Ncols, Nrows
    
    camera.capture('binary.rgb',format = 'rgb', resize = (Ncols,Nrows))
    Image = np.fromfile('binary.rgb',np.uint8, -1, '')
    Image = Image.reshape(Nrows,Ncols,3)

    if img_type in [1,2,3] :
        Inten = np.float16(Image[:,:,img_type-1])
    else :
        Inten = np.float16(Image[:,:,0]) + np.float16(Image[:,:,1]) + np.float16(Image[:,:,2])
        
    Inten = (Inten-np.min(Inten))/(np.max(Inten) - np.min(Inten))

    return

#
#
#   MAIN PROGRAM
#
#   Setup global variables and constants
#
#
#   To Run program in diagnostics mode set Eng_mode to True
#   For audio feedback set Talk_mode to True
#
Eng_mode = False
Talk_mode = False
#
#   Co-ordinate convention
#       zero location will be near chess square A1
#       x axis runs in file direction from a file to h file
#       y axis runs in rank direction from first rant to 8th rank
#       z axis runs upward toward camera with zero being near the board
#
#       CNC reliable operation motion limits in inches once initial location
#       has been calibrated correctly.  CNC will stop working and possible get
#       damaged if operated beyond these limits.
#
Z_limits = [0.0,3.5]
Y_limits = [0.0,15.0]
X_limits = [0.0,12.0]
#       CNC backlash correction in inches (neg. to pos. , pos. to neg)
Z_backlash = [0.0, 0.0]
X_backlash = [0.06, 0.06]
Y_backlash = [0.06, 0.06]
#       CNC GRBL value conversion to inches
Y_scale = 0.97/25.4
X_scale = 1.01/25.4
Z_scale = 0.95/25.5
#       Vertical Distance (inches) of Pi Camera above center of Chess board
Camera_height = 24.0 + 1.0/8.0
#       offset of laser location from magnet in x,y,z (inches)
Magnet2laser_offset = [0.0,0.0,-3.5]
#       height of laser about chess board plane (inches) when Z height of arm is 0.0
Laser_height_0 = 6.5
#
#   X0,Y0,Z0 are the current location of the magnet arm in absolute coordinates (units are in inches)
#
X0 = 6.0 
Y0 = 6.0
Z0 = 0.0

Loc_cur = [X0,Y0,Z0]
Magnet_actual_loc = [X0,Y0,Laser_height_0+Magnet2laser_offset[2]]
Loc_pre = [0,0,0]
Col_inches = 1.0
Row_inches = 1.0
Zero_location_col = 0
Zero_location_row = 0

# location of board squares in units of Image col/row and physical x and y in inches
# array is initialized by the calibrate_board routine
#
#   board[f,r,p,d]
#       where   f = is Chess Board File (0=a , 1= b, etc. 7= h file)
#               r = is Chess Board Rank (0=1 , 1= 2, etc. 7= 8 rank)
#               p = 0 : Image location of chess squares info
#                   1 : phyical location of chess squares info
#               d = 0 : row location (p=0) or x location in inches (p=1)
#                   1 : col location (p=0) or y location in inches (p=1)

Board = np.float16(np.zeros((8,8,2,2)))

#   GPIO setup
#       use board numbering of the GPIO pins
#       turn off warnings
#       setup pin 38 for output mode to control MOSFET to turn on/off electromagnet
#       set pin 38 to be initally at low value or in the OFF state for the electromagnet
#       use pin 18 to turn on LED at same time as magnet to indicate magnet is ON
#       use pin 22 to turn on/off Laser for measuring magnet location relative to chess board

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(18, GPIO.OUT,initial = False)
GPIO.setup(22, GPIO.OUT,initial = False)
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

Nrows = 1920
Ncols = 1088
Col_scal = 3.0
camera = picamera.PiCamera()
camera.rotation = 0
camera.preview_fullscreen = False
camera.crop = (0.0,0.0,1.0,1.0)
# first 2 parameters are location, last 2 are size of window
camera.preview_window =(0, 240, 720,540)
#camera.preview_window =(0, 240, 960, 720)
#camera.preview_window =(60, 540, 720, 540)
camera.resolution = (Nrows,Ncols)

#   Set up Arduino link to control CNC machine
#       Use USB at 115200 baud
#       ACM0 is USB port for Arduino

ser = serial.Serial('/dev/ttyACM0',115200)

if Talk_mode :
    talk('Hello.   I am Hal the chess playing cyclops')
    talk('Prepare to be crushed')

prompt = "Main Command (? for list of commands) : "
cmdlst =          "    cal      : auto calibrate board, chess squares and Image scales\n"
cmdlst = cmdlst + "    arm      : calibrate arm location\n"
cmdlst = cmdlst + "    loc      : outputs current location of magnet\n"
cmdlst = cmdlst + "    mo       : move magnet using abolute coordiates or chess square coordinates\n"
cmdlst = cmdlst + "    magon    : turn on magnet\n"
cmdlst = cmdlst + "    magoff   : turn off magnet\n"
cmdlst = cmdlst + "    laseron  : turn on Laser\n"
cmdlst = cmdlst + "    laseroff : turn off Laser\n"
cmdlst = cmdlst + "    im       : to capture and save JPG Image\n"
cmdlst = cmdlst + "    anal     : capture Image for manual board analysis\n"
cmdlst = cmdlst + "    grbl     : enter in GRBL command to CNC\n"
cmdlst = cmdlst + "    q        : to quit program"

#   view video until quit command is entered for Arduino

camera.start_preview()
time.sleep(2)
capture_Image(0)

strin = raw_input (prompt);
while (strin.lower() != "q"):
    if (strin.lower() == "im") :
        filename = raw_input ("Name of saved Image file? ")
        camera.capture(filename + '.jpg')   # store Image as JPEG
    elif (strin.lower() == "?") :
        print cmdlst
    elif (strin.lower() == "cal") :
        if Talk_mode : talk('OK, will go into calibration mode')
        capture_Image(0)
        camera.stop_preview()
        calibrate_board(Col_scal)
        camera.start_preview()
    elif (strin.lower() == "magon") :
        if Talk_mode : talk('Turning on magnet')
        GPIO.output(38, True)
        GPIO.output(18, True)
    elif (strin.lower() == "magoff") :
        if Talk_mode : talk('Turning off magnet')
        GPIO.output(38, False)
        GPIO.output(18, False)
    elif (strin.lower() == "laseron") :
        if Talk_mode : talk('Turning on Laser')
        GPIO.output(22, True)
    elif (strin.lower() == "laseroff") :
        if Talk_mode : talk('Turning off Laser')
        GPIO.output(22, False)
    elif (strin.lower() == "anal") :
        if Talk_mode : talk('Entering manual board analysis mode')
        camera.stop_preview()
        board = board_analysis(Col_scal)
        camera.start_preview()
    elif (strin.lower() == "mo") :
        if Talk_mode : talk('Entering into absolute coordinate move mode')
        absolute_coordinate_moves()
    elif (strin.lower() == "arm") :
        if Talk_mode : talk('Entering into calibrate arm location mode')
        camera.stop_preview()
        calibrate_arm_location(Col_scal)
        camera.start_preview()
        print 'Current X,Y,Z location : {0:.3f}, {1:.3f}, {2:.3f}'.format(Loc_cur[0],Loc_cur[1],Loc_cur[2])
    elif (strin.lower() == "loc") :
        if Talk_mode : talk('The current location of the magnet')
        print 'Current X,Y,Z location : {0:.3f}, {1:.3f}, {2:.3f}'.format(Loc_cur[0],Loc_cur[1],Loc_cur[2])
    elif (strin.lower() == "grbl") :
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
