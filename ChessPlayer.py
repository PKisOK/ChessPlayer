
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
#   Install libray (one time only, make sure WiFi in connected)
#
#   sudo apt-get dist-upgrade
#   sudo apt-get update    (makes sure you have latest updates, run 8/5/2018)
#   sudo apt-get upgrade
#   sudo apt-get install python3-picamera
#   sudo apt-get install python-numpy
#   sudo apt-get install python-matplotlib
#   sudo apt-get install python-scipy
#   sudo apt-get install python-opencv
#

import os
import picamera
import serial
import RPi.GPIO as GPIO
import numpy as np
import matplotlib.pyplot as plt
#from scipy.interpolate import interp1d
import time
import math
import cv2

def talk(cmd):
    saycmd = "echo " + cmd + " | festival --tts"
    os.system(saycmd)
    return

def find_laser() :
#
#   Find location of laser in image usings of rows and cols
#

    global Image, Inten
    global Laser_row, Laser_col
    global Col_scal
    global Eng_mode, Plt_mode

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
        print '# of pixs ',no_pixels,' x std ',pixel_std[0],' y std ',pixel_std[1]*Col_scal
        print 'size of crop image : ',crop_row,',',crop_col
        print 'Marks : (',marks.shape,'),',marks 

    locs = pixels*1.0
    locs[:,1] = locs[:,1]*Col_scal

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

        if Plt_mode :
            fig, ax = plt.subplots(nrows = 1, ncols = 2)
            ax[0].imshow(crop_img, aspect='equal', extent=[0,crop_col*Col_scal,0,crop_row], cmap='gray')      
            ax[1].set_title('Laser Spot: ('+str(round(laser_col*Col_scal,1))+','+str(round(crop_row-laser_row,1))+')')   
            ax[1].imshow(mask_img, aspect='equal', extent=[0,crop_col*Col_scal,0,crop_row], cmap='gray')        
            plt.show()
            plt.close()       # free up memory

    laser_row += lx
    laser_col += ly
    
    if Eng_mode :
        print 'Laser mark : ',laser_row,' , ',laser_col
    
    return laser_row,laser_col

def plot_inten_hist(img) :

    (rows,cols) = img.shape
        
    fig, ax = plt.subplots(nrows = 1, ncols = 3)
     
    ax[0].set_title('Raw Image')
    ax[0].imshow(img, aspect='equal', extent=[0,cols*Col_scal,0,rows], cmap='gray')
    n, bins = np.histogram(img, bins=100)
    # print '# bins: ',bins.shape,' Min bin ',np.min(bins),' Max bin : ',np.max(bins)
    ax[1].set_title('Inten Hist')
    ax[1].plot(bins[1:],n)
    norm = np.float16(np.add.accumulate(n))/np.sum(n)
    #s_cum = interp1d(bins[1:],norm,kind='cubic')
    #ax[2].set_title('Smootherd Cum Inten Hist')
    #ax[2].plot(s_cum[:,0],s_cum[:,1])
    ax[2].set_title('Cum Inten Hist')
    ax[2].plot(bins[1:],norm)
    plt.show()
    plt.close()       # free up memory

    clip = input('Select clip level for masking Image (0.0 to 1.0) (<0.0 for reverse mask): ')
    mask_img = np.copy(img)
    if clip > 0 :
        mask_img[mask_img<=clip] = 0.0
    else :
        mask_img[mask_img>= (-clip)] = 1.0
    
    fig, ax = plt.subplots(nrows = 1, ncols = 3)
        
    ax[0].set_title('Clipped Image ('+str(clip)+')')
    ax[0].imshow(mask_img, aspect='equal', extent=[0,cols*Col_scal,0,rows], cmap='gray')
    n, bins = np.histogram(mask_img, bins=100)
    ax[1].set_title('Clipped Hist')
    ax[1].plot(bins[1:],n)
    norm = np.float16(np.add.accumulate(n))/np.sum(n)
    ax[2].set_title('Cum Hist')
    ax[2].plot(bins[1:],norm)
    plt.show()
    plt.close()       # free up memory
        
    return clip


def plot_derivative_img(img,der_type) :
#
#   img : input image
#   der_type : type of derivative to perform on image
#

    global Eng_mode, Plt_mode
    
    rows,cols = img.shape
#    print 'Size of img : ',img.shape

    if der_type == 1 :  #   perform Laplacian gradient of the image
        img_scal = np.int16((img - np.min(img))/(np.max(img)-np.min(img))*255)
        dXdY_array = cv2.Laplacian(img_scal,cv2.CV_16S)
        dXdY_array = np.absolute(dXdY_array)
        label = 'Absolute Laplacian Image'
#        dXdY_array = np.absolute((dXdY_array - np.min(dXdY_array))/(np.max(dXdY_array)-np.min(dXdY_array)))
#        label = 'Laplacian Image'
        rows,cols = dXdY_array.shape
    elif der_type == 2 : #   perform Adaptive Gaussian Thresholding of the image
        img_scal = np.uint8((img - np.min(img))/(np.max(img)-np.min(img))*255)
        img_scal = cv2.medianBlur(img_scal,3)
        dXdY_array = cv2.adaptiveThreshold(img_scal,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,5,2)
        rows,cols = dXdY_array.shape
        label = 'Adapt Gaussian Thresh Image'
    else :              # my x-y derivative
        dx_array = abs(img[:,1:]-img[:,:-1])
        dy_array = abs(img[1:,:]-img[:-1,:])
        dXdY_array = dx_array[:-1,:] + dy_array[:,:-1]
#        dx_sum = np.array([np.arange(cols-1),np.sum(dx_array,0)]).T
#        dy_sum = np.array([np.arange(rows-1),np.sum(dy_array,1)]).T
        label = 'Derivative Image'


    if Plt_mode :
        fig, ax = plt.subplots(nrows = 1, ncols = 3)
                
        ax[0].set_title(label)
        ax[0].imshow(dXdY_array, aspect='equal', extent=[0,cols*Col_scal,0,rows], cmap='gray')

        n, bins = np.histogram(dXdY_array, bins=100)
        ax[1].set_title('Derv Hist')
        ax[1].plot(bins[1:],n)
        
        norm = np.float16(np.add.accumulate(n))/np.sum(n)
        ax[2].set_title('Cummulative Derv Hist')
        ax[2].plot(bins[1:],norm)
        
        plt.show()
        plt.close()       # free up memory

        clip = input('Select clip level for masking derivative Image (0.0 to 1.0) (<0.0 for reverse mask): ')
        mask_img = np.copy(dXdY_array)
        if clip > 0 :
            mask_img[mask_img<=clip] = 0.0
        else :
            mask_img[mask_img>= (-clip)] = 1.0
    
        fig, ax = plt.subplots(nrows = 1, ncols = 3)
        
        ax[0].set_title('Clipped Derv Image ('+str(clip)+')')
        ax[0].imshow(mask_img, aspect='equal', extent=[0,cols*Col_scal,0,rows], cmap='gray')
        n, bins = np.histogram(mask_img, bins=100)
        ax[1].set_title('Clipped Derv Hist')
        ax[1].plot(bins[1:],n)
        norm = np.float16(np.add.accumulate(n))/np.sum(n)
        ax[2].set_title('Cum Clipped Derv Hist')
        ax[2].plot(bins[1:],norm)

        plt.show()
        plt.close()       # free up memory        

    return clip


def crop_img(low_x,low_y,hi_x,hi_y) :
    
    print 'Input for cropping (type 0 to use previous cropped values)'
    l_x = input("Enter lower left  hand X value: ("+str(low_x)+") ")
    l_y = input("Enter lower left  hand Y value: ("+str(low_y)+") ")
    h_x = input("Enter upper right hand X value: ("+str(hi_x)+") ")
    h_y = input("Enter upper right hand Y value: ("+str(hi_y)+") ")
    if (l_x >0) : low_x = l_x
    if (l_y >0) : low_y = l_y
    if (h_x >0) : hi_x  = h_x
    if (h_y >0) : hi_y  = h_y
           
    lx = np.uint16(low_x/Col_scal)
    ly = rows - np.uint16(hi_y)
    hx = np.uint16(hi_x/Col_scal)
    hy = rows - np.uint16(low_y)
    crop_img = np.copy(Inten[ly:hy,lx:hx])
    #crop_img = (crop_img - np.min(crop_img))/(np.max(crop_img) - np.min(crop_img))
    
    return low_x,low_y,hi_x,hi_y,crop_img

  

def board_analysis():

#
#   allows manual analysis of images
#

    global Image, Inten
    global Nrows, Ncols
    global Col_scal
    

    (rows,cols) = Inten.shape
    
    low_x = 0
    low_y = 0
    hi_x = int(cols*Col_scal)
    hi_y = rows
    
    prompt = "Analysis Mode Command (? for list of commands) : "
    cmdlst = "List of commands: \n"
    cmdlst = cmdlst + "    v  : capture new Image and view the full Image in intensity scale\n"
    cmdlst = cmdlst + "    vr  : capture new Image and view the full Image in red scale\n"
    cmdlst = cmdlst + "    vg  : capture new Image and view the full Image in green scaled\n"
    cmdlst = cmdlst + "    vb  : capture new Image and view the full Image in blue scaled\n"
    cmdlst = cmdlst + "   chess: update board position and print it out\n"
    cmdlst = cmdlst + "    sq  : crop images for chess square\n"
    cmdlst = cmdlst + "    l  : analyze threshold level of cropped Image\n"
    cmdlst = cmdlst + "    c  : crop existing Image\n"
    cmdlst = cmdlst + "    der  : analyze derivatives of cropped Image\n"
    cmdlst = cmdlst + "    fl : find laser mark in image\n"
    cmdlst = cmdlst + "    q  : finished analysis return to main menu\n"

    crop_img = np.copy(Inten)

    strin = raw_input (prompt);
    while (strin.lower() != "q"):    
        if (strin.lower() == "v") :
            capture_image(0,'all')
            plt.imshow(Inten, aspect='equal', extent=[0,cols*Col_scal,0,rows], cmap='gray')
            plt.show()
            plt.close()       # free up memory
            low_x = 0
            low_y = 0
            hi_x = int(cols*Col_scal)
            hi_y = rows
            crop_img = np.copy(Inten)
        elif (strin.lower() == "vr") :
            capture_image(1,'all')
            plt.imshow(Inten, aspect='equal', extent=[0,cols*Col_scal,0,rows], cmap='Reds_r')
            plt.show()
            plt.close()       # free up memory
            low_x = 0
            low_y = 0
            hi_x = int(cols*Col_scal)
            hi_y = rows
            crop_img = np.copy(Inten)
        elif (strin.lower() == "vg") :
            capture_image(2,'all')
            plt.imshow(Inten, aspect='equal', extent=[0,cols*Col_scal,0,rows], cmap='Greens_r')
            plt.show()
            plt.close()       # free up memory
            low_x = 0
            low_y = 0
            hi_x = int(cols*Col_scal)
            hi_y = rows
            crop_img = np.copy(Inten)
        elif (strin.lower() == "vb") :
            capture_image(3,'all')
            plt.imshow(Inten, aspect='equal', extent=[0,cols*Col_scal,0,rows], cmap='Blues_r')
            plt.show()
            plt.close(ax)       # free up memory
            low_x = 0
            low_y = 0
            hi_x = int(cols*Col_scal)
            hi_y = rows
            crop_img = np.copy(Inten)
        elif (strin.lower() == "chess") :
            update_chess_board()
            print_chess_board()
        elif (strin.lower() == "sq") :
            sq = raw_input('Which square to view and create cropped image (e.g. a6, e8) ? ')
            if len(sq) == 2 :
                code = get_square_image(sq,0)
                crop_img = np.copy(Sq_img)
            else :
                print 'Square (',sq,') not recognized'
        elif (strin.lower() == "fl") :
            laser_row,laser_col = find_laser()
        elif(strin.lower() == "l") :
            clip = plot_inten_hist(crop_img)    
        elif (strin.lower() == "c") :
            (low_x,low_y,hi_x,hi_y,crop_img) = crop_image(low_x,low_y,hi_x,hi_y)
        elif (strin.lower() == "der") :
            der_t = input('Type of derivative (0: x-y, 1: Laplacian, 2: Adaptive Gaussian Thresh) = ')
            plot_derivative_img(crop_img,der_t)           
        else :
            print cmdlst
        strin = raw_input (prompt);
    return crop_img

def update_chess_board() :

    global Chess_board
    
    file_list = 'abcdefgh'
    rank_list = '12345678'

    capture_image(0,'board')
    
    for f in range(8) :
        for r in range(8):
            sq = file_list[f] + rank_list[r]
            code = get_square_image(sq,0)
            Chess_board[f][r] = code            
    return

def print_chess_board() :

    global Chess_board
    global Piece_code
    global Color_code
    
    file_list = 'abcdefgh'
    rank_list = '12345678'

    print '  ',
    for i in range(21) :
        print '-',
    print
    for r in range(7,-1,-1) :
        print rank_list[r],' |',
        for f in range(8):
            print Color_code[Chess_board[f][r][0]]+Piece_code[Chess_board[f][r][1]],'|',
        print
    print '  ',
    for i in range(21) :
        print '-',
    print
    print '   |',
    for f in range(8) :
        print file_list[f],' |',
    print

    return
    

def calibrate_board():
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
#   Array Board[xinfo,yinfo,pix_inches,file_rank]
#       where
#           xinfo   yinfo   pix_inches  file_rank   Comments
#           -----   -----   ----------  ---------   --------
#           row     col         0          0        row & col center loc of files in units of pixels
#           row     col         0          1        row & col center loc of ranks in units of pixels
#           x       y           1          0        x & y center loc of files in units of inches ( 0,0 is lower left hand corner of board - a1 square)
#           x       y           1          1        x & y center loc of ranks in units of inches
#
    global Image, Inten
    global Laser_x_loc, Laser_y_loc
    global Row_inches, Col_inches
    global Zero_location_col
    global Zero_location_row
    global Col_scal
    global Board
    global Squares_x_location
    global Squares_y_location
    global Chess_board
    global Eng_mode, Plt_mode
    global A8_boardcorner_row, A8_boardcorner_col
    global H1_boardcorner_row, H1_boardcorner_col
    

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
    
    if Plt_mode :
        fig, ax = plt.subplots(nrows = 3, ncols = 3)
        ax[0,0].imshow(crop_img, aspect='equal', extent=[0,crop_col*Col_scal,0,crop_row], cmap='gray')
        ax[0,0].set_title('Chess Board Image')
        ax[0,1].plot(bins[1:],n)
        ax[0,1].set_title('Intensity Histogram')            
        ax[1,0].imshow(dy_array, aspect='equal', extent=[0,(crop_col-1)*Col_scal,0,crop_row], cmap='gray')
        ax[1,0].set_title('dY Image')
        ax[1,1].plot(np.arange(crop_col-1),dy_sum)
        ax[1,1].set_title('peaks are Col/Rank Boundaries')
        
    n, bins = np.histogram(dy_sum, bins=100)
    norm = np.float16(np.add.accumulate(n))/np.sum(n)
    if Plt_mode :
        ax[1,2].set_title('Cummulative Histogram of dY sum')
        ax[1,2].plot(bins[1:],norm)
       
    clip_lev = np.min(bins[norm>clip])
    dy_peaks = np.ravel(ly+np.argwhere(dy_sum>clip_lev))
    delta_dy_peaks = dy_peaks[1:]-dy_peaks[:-1]
    threshold = int(0.9 * max(delta_dy_peaks))
    squares_y_location1 = np.ravel(dy_peaks[np.argwhere(delta_dy_peaks>threshold)])
    squares_y_location2 = np.ravel(dy_peaks[np.argwhere(delta_dy_peaks>threshold)+1])
    Squares_y_location  = (squares_y_location1[:-1]+squares_y_location2[1:])/2 + 1
    Squares_y_location  = np.append([squares_y_location1[0]+1],[Squares_y_location])
    Squares_y_location  = np.append([Squares_y_location],[squares_y_location2[-1]])
    if Eng_mode :
        print 'Clip : ',clip,' dY level > : ',clip_lev
        print 'Col Separation Threshold Between Peaks : ',threshold
        print 'Col location of Rank Boundary Peaks : ',dy_peaks,' Shape: ',dy_peaks.shape
        print 'Col Difference between Peaks : ',dy_peaks[1:]-dy_peaks[:-1]
        print 'Selected Peaks: ',Squares_y_location
        print '# of Ranks detected: ',len(Squares_y_location)-1
 
    dx_array = abs(crop_img[1:,:]-crop_img[:-1,:])
    dx_sum = np.sum(dx_array,1)
    if Plt_mode :
        ax[2,0].imshow(dx_array, aspect='equal', extent=[0,crop_col*Col_scal,0,(crop_row-1)], cmap='gray')
        ax[2,0].set_title('dX Image')
        ax[2,1].plot(np.arange(crop_row-1),dx_sum)
        ax[2,1].set_title('peaks are Row/File Boundaries')
    n, bins = np.histogram(dx_sum, bins=100)
    norm = np.float16(np.add.accumulate(n))/np.sum(n)
    if Plt_mode :
        ax[2,2].set_title('Cummulative Histogram of dX sum')
        ax[2,2].plot(bins[1:],norm)
    
    clip_lev = np.min(bins[norm>clip])
    dx_peaks = np.ravel(lx+np.argwhere(dx_sum>clip_lev))
    delta_dx_peaks = dx_peaks[1:]-dx_peaks[:-1]
    threshold = int(0.9 * max(delta_dx_peaks))
    squares_x_location1 = np.ravel(dx_peaks[np.argwhere(delta_dx_peaks>threshold)])
    squares_x_location2 = np.ravel(dx_peaks[np.argwhere(delta_dx_peaks>threshold)+1])
    Squares_x_location  = (squares_x_location1[:-1]+squares_x_location2[1:])/2
    Squares_x_location  = np.append([squares_x_location1[0]],[Squares_x_location])
    Squares_x_location  = np.append([Squares_x_location],[squares_x_location2[-1]])
    if Eng_mode :
        print 'Clip : ',clip,' dX level > : ',clip_lev
        print 'Row Separation Threshold Between Peaks: ',threshold
        print 'Row location of Row/File Boundary Peaks : ',dx_peaks,' Shape: ',dx_peaks.shape
        print 'Row Difference Between Peaks: ',dx_peaks[1:]-dx_peaks[:-1]
        print 'Selected Peaks: ',Squares_x_location
        print '# of Files detected: ',len(Squares_x_location)-1
    if Plt_mode :
        plt.show()
        plt.close()       # free up memory
    
    A8_boardcorner_col = max(Squares_y_location)
    A8_boardcorner_row = min(Squares_x_location)
    H1_boardcorner_col = min(Squares_y_location)
    H1_boardcorner_row = max(Squares_x_location)
    Col_inches = no_ysqs*sq_yd/(A8_boardcorner_col-H1_boardcorner_col)
    Row_inches = no_xsqs*sq_xd/(H1_boardcorner_row-A8_boardcorner_row)
    Col_scal = Col_inches / Row_inches

    files_c = np.float16(Squares_x_location[1:]+Squares_x_location[:-1])*0.5
    ranks_c = np.float16(Squares_y_location[1:]+Squares_y_location[:-1])*0.5

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
        print 'Col_scal : ', Col_scal

    board_img = np.copy(Inten)
    board_img[A8_boardcorner_row:H1_boardcorner_row,H1_boardcorner_col:A8_boardcorner_col]=0.0
    board_img[Squares_x_location,:] = 1.0
    board_img[:,Squares_y_location] = 1.0
    
    if Plt_mode :
        plt.imshow(board_img, aspect='equal', extent=[0,cols*Col_scal,0,rows], cmap='gray')
        plt.show()
        plt.close()       # free up memory
        plt.imshow(crop_img, aspect='equal', extent=[0,crop_col*Col_scal,0,crop_row], cmap='gray')
        plt.show()
        plt.close()       # free up memory

    update_chess_board()
    print_chess_board()
    
    return

def get_square_image(square,show_img) :
#
#   returns the cropped image of square
#
    global Board
    global Image, Inten, Sq_img
    global Col_scal
    global Squares_x_location
    global Squares_y_location
    global Eng_mode, Plt_mode

    file_list = 'abcdefgh'
    rank_list = '12345678'
    error = 0
    r_bord = 8      # row border
    c_bord = 3      # col border

    square = square.lower()
    
    if len(square) == 2 :
        if square[0] in file_list :
            file_n = file_list.index(square[0])
            if square[1] in rank_list :
                rank_n = rank_list.index(square[1])
                row_c = Board[file_n,rank_n,0,0]
                col_c = Board[file_n,rank_n,0,1]
                r1 = int (Squares_x_location[file_n] + r_bord)
                r2 = int (Squares_x_location[file_n+1] - r_bord)
                c1 = int (Squares_y_location[rank_n] + c_bord)
                c2 = int (Squares_y_location[rank_n+1] - c_bord)
                Sq_img = np.copy(Inten[r1:r2,c1:c2])
                (rows,cols) = Sq_img.shape 
        
                if (show_img) :
                    plt.imshow(Sq_img, aspect='equal', extent=[0,cols*Col_scal,0,rows], cmap='gray')
                    plt.show()
                    plt.close()       # free up memory

                code = determine_chess_piece(square)
            else :
                error = 1                
        else :
            error = 1                         
    else :
        error = 1

    if error :
        print 'Invalid chess square : ',square
        crol_img = Inten
        code = [0,0]

    return code

def get_kurtosis(data) :
#
#       calculates the Kurtosis of the data
#

    mean = np.mean(data)
    var  = np.var(data)
    mom4 = np.sum(np.power((data-mean),4))
#    kurt = mom4 / var / var

    return  mom4

def determine_chess_piece(sq) :
#
#   analysis image to determine the following
#
#   Code    =  [1,1] : no pieces or objects on the square
#           =  [0,0] : cannot determine identity of object
#           =  [2,0] : white chess piece on square but cannot determine identity
#           =  [3,0] : black chess piece on square but cannot determine identity
#           =  [2,n] or [3,n] : where n = 7 for King, 6 for Queen, 5 for Rook
#                               4 for Bishop, 3 for Knight and 2 for Pawn
#
#   Use various features of image to determine what chess piece is in the image
#
#   Img_feat[0,0] : chess piece color (1: white, 0: black)
#   Img_feat[0,1] : chess piece (1:empty, 2: pawn, 3: knight, 4: bishop, 5:rook, 6: queen, 7: king)
#   Img_feat[0,2] : chess square (11 : a1, 12:a2, 21: b1, etc up to 88: h8)
#   Img_feat[0,3] : color of chess square (1: white, 0: black)
#   Img_feat[0,4] : radial distance of mid point of square to center of chess board (in inches)
#   Img_feat[0,5] : angle of radial line with the X-axis (chess file direction or image row direction)
#   Img_feat[0,6] : mean intensity of image
#   Img_feat[0,7] : standard deviation intensity of image
#   Img_feat[0,8] : kurtosis of intensity of image
#   Img_feat[0,9] : normalized std. (std/avg)
#   Img_feat[0,10]: range in intensity (max-min)
#   Img_feat[0,11]: pixel footprint of object as % of total pixels in image as determined by image threshold analysis
#   Img_feat[0,12]: radius of the base of object as determined by the derivative image analysis
#   Img_feat[0,13]: radius of the crown of object as determined by the derivative image analysis
#   Img_feat[0,14]: radius of the waist of object as determined by the derivative image analysis
#   Img_feat[0,15]: reserved for future use
#   Img_feat[0,16]: reserved for future use
#   Img_feat[0,17]: reserved for future use
#   Img_feat[0,18]: reserved for future use
#   Img_feat[0,19]: reserved for future use
#
    global Sq_img
    global Eng_mode, Plt_mode
    global Img_feat, Img_feat_header

    
    empty_th  = 0.025
    low_contr = 0.6
    low_kurt  = 10.0

    file_list = 'abcdefgh'
    rank_list = '12345678'


    sq = sq.lower()
    if len(sq) <> 2 :
        sq = 'd1'
        
    file_n = file_list.index(sq[0])
    rank_n = rank_list.index(sq[1])

    Img_feat[0,2] = (file_n+1)*10 + rank_n + 1

    if (sq[0] in ['a','c','e','g']) & (sq[1] in ['1','3','5','7']) :        # dark square
        Img_feat[0,3] = 0.0
    elif (sq[0] in ['b','d','f','h']) & (sq[1] in ['2','4','6','8']) :      # dark square
        Img_feat[0,3] = 0.0
    else :                                                                  # light square
        Img_feat[0,3] = 1.0
    
    Img_feat[0,6] = inten_avg = np.mean(Sq_img)
    Img_feat[0,7] = inten_dev = np.std(Sq_img)
    Img_feat[0,8] = inten_kurt = get_kurtosis(Sq_img)
    
    if inten_avg <> 0.0 :
        norm_dev = inten_dev / inten_avg
    else :
        norm_dev = 1.0

    Img_feat[0,9] = norm_dev
    Img_feat[0,10] = inten_range = np.max(Sq_img) - np.min(Sq_img)

    if inten_dev < empty_th :        # square must be empty
        code = [1,1]
    elif ((inten_range < low_contr) & (inten_kurt < low_kurt)):
        if (sq[0] in ['a','c','e','g']) & (sq[1] in ['1','3','5','7']) :        # dark piece
            code = [3,0]
        elif (sq[0] in ['b','d','f','h']) & (sq[1] in ['2','4','6','8']) :      # dark piece
            code = [3,0]
        else :
            code = [2,0]                                                        # light piece
    else :
        if (sq[0] in ['a','c','e','g']) & (sq[1] in ['1','3','5','7']) :        # light piece
            code = [2,0]
        elif (sq[0] in ['b','d','f','h']) & (sq[1] in ['2','4','6','8']) :      # light piece
            code = [2,0]
        else :
            code = [3,0]                                                        # dark piece

    

    if Eng_mode :
        print 'Avg. Inten: {0:.3f}, Std. Inten: {1:.3f}, Kurtosis: {2:.3f}'.format(inten_avg,inten_dev,inten_kurt)
        print 'Normalized Std : {0:.3f}, Range : {1:.3f} '.format(norm_dev,inten_range)
        print 'Max inten: {0:.3f}, Min inten: {1:.3f}'.format(np.max(Sq_img),np.min(Sq_img))
        print 'Code : ',code
        
    if Plt_mode :
        clip = plot_inten_hist(Sq_img)    
        plot_derivative_img(Sq_img,1)           

    return code

def calibrate_arm_location() :

    global Loc_cur
    global X0,Y0,Z0
    global Magnet2laser_offset
    global Magnet_actual_loc
    global Camera_height
    global Laser_height_0
    global Row_inches, Col_inches
    global Zero_location_col
    global Zero_location_row
    global Col_scal

#
#   turn on laser and capture image
#
    GPIO.output(22, 1)
    capture_image(1,'all')
#
#   find laser spot
#
    laser_row,laser_col = find_laser()

    if Eng_mode :
        print 'Zero row,col  : {0:.3f} , {1:.3f}'.format(Zero_location_row,Zero_location_col)
        print 'Laser row,col : {0:.3f} , {1:.3f}'.format(laser_row,laser_col)
    
    laser_x_loc = (laser_row-Zero_location_row) * Row_inches
    laser_y_loc = (laser_col-Zero_location_col) * Col_inches

    if Eng_mode :
        print 'Laser location (x,y) {0:.3f} , {1:.3f} : '.format(laser_x_loc,laser_y_loc)
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
    if (laser_y_loc <> 6) :
        phi   = math.atan((laser_x_loc-6)/(laser_y_loc-6))
    else :
        phi   = math.pi*0.5

    if Eng_mode :
        print 'theta, phi : {0:.3f}, {1:.3f}'.format(theta,phi)
    
    if theta == math.pi*0.5 :
        shadow = 0
    else :
        shadow = laser_ht / math.tan(theta)

    if Eng_mode :
        print 'shadow : {0:.3f}'.format(shadow)
    
    laser_zero_plane_loc = Magnet_actual_loc
    laser_zero_plane_loc[0] = laser_x_loc - shadow*math.sin(phi)
    laser_zero_plane_loc[1] = laser_y_loc - shadow*math.cos(phi)
    laser_zero_plane_loc[2] = 0.0

    if Eng_mode :
        print 'prev magnet loc ',Magnet_actual_loc
        print 'mag2laser offset ',Magnet2laser_offset
    
    Magnet_actual_loc = np.add(laser_zero_plane_loc, Magnet2laser_offset)
    Magnet_actual_loc[2] = Magnet_actual_loc[2] + laser_ht

    if Eng_mode :
        print 'new magnet loc ',Magnet_actual_loc
    
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
    GPIO.output(22, 0)

    return

def act_on_chesspiece(action, piece) :

#
#   move pickup or place chess piece
#
#       action = 'pickup'
#                'place'
#
#       piece = 'k','q','r','b','k','p'
#
    global Piece_hts
    global Hover_ht
    global Magnet2laser_offset
    global Laser_height_0

    piece_lst = 'kqrbkp'

    p = piece[0].lower()
    
    if p in piece_lst :
        n = piece_lst.index(p)
        zloc = Piece_hts[n]-Laser_height_0-Magnet2laser_offset[2]
        
        print 'Moving to (',zloc,')'

        loc = [zloc]
        move2loc('Z',loc)
        time.sleep(20)

        if action == 'pickup' :
            magnet_on_off(1)
        else :
            magnet_on_off(0)

        zloc = Hover_ht - Laser_height_0 - Magnet2laser_offset[2]

        print 'Moving to (',zloc,')'

        loc = [zloc]
        move2loc('Z',loc)
        time.sleep(20)
              
    else :
        print 'Invalid chess piece : ',piece
                                     
    return

def move2square(square) :

#
#   move arm to chess square
#
    global Board

    file_list = 'abcdefgh'
    rank_list = '12345678'
    
    if len(square) == 2 :
        if square[0] in file_list :
            file_n = file_list.index(square[0])
        if square[1] in rank_list :
            rank_n = rank_list.index(square[1])
        xloc = Board[file_n,rank_n,1,0]
        yloc = Board[file_n,rank_n,1,1]

        print 'Moving to ',square,' located at (',xloc,',',yloc,')'

        loc = [xloc,yloc]
        move2loc('XY',loc)                                    
    else :
        print 'Invalid chess square : ',square
                                     
    return

def move2loc(axis, loc) :
    
#
#   move arm to absolute X,Y,Z coordinates
#
#   axis = 'X'  for x axis motion
#          'Y'  for y axis motion
#          'Z'  for z axis motion
#          'XY' for simultaneous x and y axis motion
#          'XYZ' for simultaneous x,y,z axis motion
#
    global XYZ_limits
    global Loc_cur, Loc_pre
    global CNC_scale
    global Backlash

    axis_lst = 'XYZ'
    cmd_str = ''

    for i in range(len(axis)) :
    
        ax = axis[i].upper()

        if ax in axis_lst :
            n = axis_lst.index(ax)
   
            if loc[i] > XYZ_limits[n][1] :
                loc[i] = XYZ_limits[n][1]
                print 'Upper limit for axis '+ax+' reached : ',loc[i]
            elif loc[i] < XYZ_limits[n][0] :
                loc[i] = XYZ_limits[n][0]
                print 'Lower limit for axis '+ax+' reached : ',loc[i] 
            value = loc[i] - Loc_cur[n]
        
            if   ( (Loc_cur[n] < Loc_pre[n]) & (value > 0) ) :
                value = value + Backlash[n][0]
                print 'Backlash correction applied'
            elif ( (Loc_cur[n] > Loc_pre[n]) & (value < 0) ) :
                value = value - Backlash[n][1]
                print 'Backlash correction applied'
            
            if (loc[i] - Loc_cur[n]) != 0 :
                Loc_pre[n] = Loc_cur[n]
                Loc_cur[n] = loc[i]

            GRBLvalue = round(value/CNC_scale[n],2)

#            print 'loc : '+ax+':',loc
#            print 'GRBL : ',GRBLvalue

            if n == 0 :
                cmd_str = cmd_str+' Y'+repr(GRBLvalue)
            elif n == 1 :
                cmd_str = cmd_str+' X'+repr(-GRBLvalue)
            elif n == 2 :
                cmd_str = cmd_str+' Z'+repr(GRBLvalue)
        
        else :
            print '*** Motion Axis "'+axis+'" not valid *** '

    cmd_str = 'G91 G0'+cmd_str+'\r\n'

    print 'GRBL cmd : '+cmd_str,

    ser.write(cmd_str)        
    reply = ser.readline()
    print 'CNC Reply back: ',reply,
        
    return

def chess_move(move)    :
#
#   takes chess move command in full algebraic notation and
#   executes moves
#
#       e.g.  Nb1-c3  moves night on b1 to c3 square
#             Qd1xBh4  queen capture bishop on h4 square by first removing
#                       piece on h4 square and then moving queen to h4
#

    global Hover_ht
    global Captured_piece_loc

    if len(move) in [6,7] :
        p_m = move[0].lower()
        sq0 = move[1:3].lower()
        if len(move) == 7 :
            p_c = move[4].lower()
            sq1 = move[5:7].lower()
            move2loc('z',[Hover_ht])    
            move2square(sq1)
            act_on_chesspiece('pickup',p_c)
            move2loc('xy', Captured_piece_loc)
            act_on_chesspiece('place',p_c)
            Captured_piece_loc[0] += 1.0
        else :
            sq1 = move[4:6].lower()
        move2square(sq0)
        act_on_chesspiece('pickup',p_m)
        move2square(sq1)
        act_on_chesspiece('place',p_m)
    else :
        print 'Chess move : ('+move+') not recoqnized'
        
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
    cmdlst = cmdlst + "    pick : pick up chess piece\n"
    cmdlst = cmdlst + "    place : place chess piece\n"
    cmdlst = cmdlst + "    loc : display current location\n"
    cmdlst = cmdlst + "    chess : enter a chess move command\n"    
    cmdlst = cmdlst + "    q  : finished return to main menu\n"
    
    strin = raw_input (prompt);
    while (strin.lower() != "q"):    
        if (strin.lower() == "x") : 
            loc = input('Enter in X coordinate : (' + repr(round(Loc_pre[0],2)) + '->' + repr(round(Loc_cur[0],2)) + ') ')
            loc_l = [loc]
            move2loc(strin,loc_l)
        elif (strin.lower() == "y") :
            loc = input('Enter in Y coordinate : (' + repr(round(Loc_pre[1],2)) + '->' + repr(round(Loc_cur[1],2)) + ') ')
            loc_l = [loc]
            move2loc(strin,loc_l)
        elif (strin.lower() == "z") :
            loc = input('Enter in Z coordinate : (' + repr(round(Loc_pre[2],2)) + '->' + repr(round(Loc_cur[2],2)) + ') ')
            loc_l = [loc]
            move2loc(strin,loc_l)
        elif (strin.lower() == "sq") :
            square = raw_input('Which chess square to move to? (e.g a1 or f3) ')
            square = square.lower()
            move2square(square)
        elif (strin.lower() == "pick") :
            act_on_chesspiece('pickup','q')
        elif (strin.lower() == "place") :
            act_on_chesspiece('place','q')
        elif (strin.lower() == "chess") :
            move = raw_input('Enter chess move (e.g. Qd8-g6')
            move = move.lower()
            chess_move(move)
        elif (strin.lower() == "loc")  :
            print 'Current X,Y,Z location : {0:.3f}, {1:.3f}, {2:.3f}'.format(Loc_cur[0],Loc_cur[1],Loc_cur[2])           
        else :
            print cmdlst
        strin = raw_input (prompt);
    return

def capture_image (img_type,scope) :

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
    global A8_boardcorner_row, A8_boardcorner_col
    global H1_boardcorner_row, H1_boardcorner_col
    
    camera.capture('binary.rgb',format = 'rgb', resize = (Ncols,Nrows))
#   camera.capture('binary.rgb',format = 'rgb')
    Image = np.fromfile('binary.rgb',np.uint8, -1, '')
    Image = Image.reshape(Nrows,Ncols,3)
    
    if img_type in [1,2,3] :
        Inten = np.float16(Image[:,:,img_type-1])
    else :                              # convert RGB to luminosity gray scale 
        Inten = np.float16(Image[:,:,0])*0.3 + np.float16(Image[:,:,1])*0.59 + np.float16(Image[:,:,2])*0.11
        
    Inten = (Inten-np.min(Inten))/(np.max(Inten) - np.min(Inten))

    if scope == 'board' :
        board_max = np.max(Inten[A8_boardcorner_row:H1_boardcorner_row,H1_boardcorner_col:A8_boardcorner_col])
        board_min = np.min(Inten[A8_boardcorner_row:H1_boardcorner_row,H1_boardcorner_col:A8_boardcorner_col])
        Inten[A8_boardcorner_row:H1_boardcorner_row,H1_boardcorner_col:A8_boardcorner_col] = (Inten[A8_boardcorner_row:H1_boardcorner_row,H1_boardcorner_col:A8_boardcorner_col]-board_min)/(board_max - board_min)

    return

def magnet_on_off(value) :

    if value == 1 :
        if Talk_mode : talk('Turning on magnet')
        GPIO.output(38, 1)
        GPIO.output(18, 1)
    else :
        if Talk_mode : talk('Turning off magnet')
        GPIO.output(38, 0)
        GPIO.output(18, 0)
    
    return

def laser_on_off(value) :

    if value == 1 :
        if Talk_mode : talk('Turning on laser')
        GPIO.output(22, 1)
    else :
        if Talk_mode : talk('Turning off laser')
        GPIO.output(22, 0)
    
    return


#
#
#   MAIN PROGRAM
#
#   Setup global variables and constants
#
#
#   To Run program in diagnostics mode set Eng_mode to 1
#   For audio feedback set Talk_mode to 1
#
Eng_mode = 0
Plt_mode = 0
Talk_mode = 0
#
#   Board Orientation
#       Board_O  =  0 : human will play with white pieces from side  of CNC machine/robot (default)
#                   1 : human will play with black pieces from side  of CNC machine/robot
#                   2 : human will play with white pieces from front of CNC machine/robot
#                   3 : human will play with black pieces from front of CNC machine/robot
#
Board_O = 0
#
#   Co-ordinate convention
#       physical zero location is fixed indendent of board oriention and will be located at
#           Board_O =   0 : zero is upper left corner of A8 square
#                       1 : zero is lower right corner of H1 square
#                       2 : zero is lower left corner of A1 square
#                       3 : zero is upper right corner of H8 square
#       x axis runs reverse direction but parallel camera row axis
#       y axis runs same direction and parallel to camera column axis
#       z axis runs upward toward camera with zero being near the board
#
#       CNC reliable operation motion limits in inches once initial location
#       has been calibrated correctly.  CNC will stop working and possible get
#       damaged if operated beyond these limits.
#
#       Stage locations as tracked by X0, Y0 and Z0 are kept within these limits
#           Locations are calibrated during the setup phase
#
XYZ_limits = [[0.0,12.0],[0.0,15.0],[0.0,3.50]]
#       CNC backlash correction in inches (neg. to pos. , pos. to neg)
Z_backlash = [0.0, 0.0]
X_backlash = [0.06, 0.06]
Y_backlash = [0.06, 0.06]
Backlash = [[0.06, 0.06],[0.06,0.06],[0.0,0.0]]
#       CNC GRBL value conversion to inches
Y_scale = 0.97/25.4
X_scale = 1.01/25.4
Z_scale = 1.0/25.4
CNC_scale = [X_scale,Y_scale,Z_scale]
#       Vertical Distance (inches) of Pi Camera above center of Chess board
Camera_height = 24.0 + 1.0/8.0 - 0.1
#       offset of laser location from magnet in x,y,z (inches)
Magnet2laser_offset = [0.0,-0.5,-3.5]
#       height of laser about chess board plane (inches) when Z height of arm is 0.0
Laser_height_0 = 4.75
#
#   create an array for storing feature properties that will be used to determined identity of chess piece from image
#
Img_feat = np.reshape(np.zeros(20),[1,20])
feat_log = Img_feat
log_fname = 'testlog.txt'
Img_feat_header  = 'Piece_Color'
Img_feat_header += '\tChess_Piece'
Img_feat_header += '\tChess_Sq'
Img_feat_header += '\tColor_Sq'
Img_feat_header += '\tDistance_Ctr'
Img_feat_header += '\tAngle_Ctr'
Img_feat_header += '\tInten_Mean'
Img_feat_header += '\tInten_Std'
Img_feat_header += '\tInten_Kurtosis'
Img_feat_header += '\tInten_Norm_Std'
Img_feat_header += '\tInten_Range'
Img_feat_header += '\tFootprint'
Img_feat_header += '\tRadius_Base'
Img_feat_header += '\tRadius_Crown'
Img_feat_header += '\tRadius_Waist'
Img_feat_header += '\treserved'
Img_feat_header += '\treserved'
Img_feat_header += '\treserved'
Img_feat_header += '\treserved'
Img_feat_header += '\treserved'
#
#   X0,Y0,Z0 are the current location of the 3 stages in absolute coordinates (units are in inches)
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

Board = np.zeros([8,8,2,2])
Piece_hts = [2.0,2.0,2.25,2.25,2.25,2.25]     # ht of piece above planecalibrate_board of board - k,q,r,b,k,p
Hover_ht  = 4.00                              # ht of magnet above plane of board when holding piece
Captured_piece_loc = [6.0,14.5]
Chess_board = np.zeros([8,8,2],dtype=int)
Color_code = ['X','0','W','B']
Piece_code = ['X','0','P','K','B','R','Q','K']
#   GPIO setup
#       use board numbering of the GPIO pins
#       turn off warnings
#       setup pin 38 for output mode to control MOSFET to turn on/off electromagnet
#       set pin 38 to be initally at low value or in the OFF state for the electromagnet
#       use pin 18 to turn on LED at same time as magnet to indicate magnet is ON
#       use pin 22 to turn on/off Laser for measuring magnet location relative to chess board

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(0)
GPIO.setup(18, GPIO.OUT,initial = 0)
GPIO.setup(22, GPIO.OUT,initial = 0)
GPIO.setup(38, GPIO.OUT,initial = 0)

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
#
#   setup camera
#
camera = picamera.PiCamera()
camera.rotation = 0
camera.preview_fullscreen = 0
camera.crop = (0.0,0.0,1.0,1.0)
camera.preview_window =(0, 240, 640,320)

camera.resolution = (Nrows,Ncols)

#   Set up Arduino link to control CNC machine
#       Use USB at 115200 baud
#       ACM0 is USB port for Arduino

ser = serial.Serial('/dev/ttyACM0',115200)

try:
    prompt = raw_input('Enter y to run in Engineering Mode : ')
    if prompt.lower() == 'y' :
        Eng_mode = 1
        try:
            prompt = raw_input('Enter y to run in Engineering Mode without Plots : ')
            if prompt.lower() == 'y' :
                Plt_mode = 0
                print '*** running in Engineering Mode with Plotting Off ***'
            else :
                Plt_mode = 1
                print '*** running in Engineering Mode with Plotting On ***'
        except ValueError:
            Plt_mode = 1
            print '*** running in Engineering Mode with Plotting On ***'
    else :
        Eng_mode = 0
        Plt_mode = 0
except ValueError:
    Eng_mode = 0
    Plt_mode = 0
    
try:
    prompt = raw_input('Enter y to run in Talking Mode : ')
    if prompt.lower() == 'y' :
        Talk_mode = 1
        print '*** running in Talking Mode ***'
    else :
        Talk_mode = 0
except ValueError:
    Talk_mode = 0    

if Talk_mode :
    talk('Hello.   I am Hal the chess playing cyclops')
    talk('Prepare to be crushed')

prompt = "Main Command (? for list of commands) : "
cmdlst =          "    cal      : auto calibrate board, chess squares and Image scales\n"
cmdlst = cmdlst + "    arm      : calibrate arm location\n"
cmdlst = cmdlst + "    ref_b    : refreshed chess board position and prints it\n"
cmdlst = cmdlst + "    loc      : outputs current location of magnet\n"
cmdlst = cmdlst + "    mo       : move magnet using abolute coordiates or chess square coordinates\n"
cmdlst = cmdlst + "    magon    : turn on magnet\n"
cmdlst = cmdlst + "    magoff   : turn off magnet\n"
cmdlst = cmdlst + "    laseron  : turn on Laser\n"
cmdlst = cmdlst + "    laseroff : turn off Laser\n"
cmdlst = cmdlst + "    im       : to capture and save JPG Image\n"
cmdlst = cmdlst + "    load_log : load a log file for appending feature analysis data\n"
cmdlst = cmdlst + "    make_log : create a new log file for storing feature analysis data\n"
cmdlst = cmdlst + "    sq_img   : analyze image of a chess square and store feature data into log file\n"
cmdlst = cmdlst + "    anal     : capture Image for manual board analysis\n"
cmdlst = cmdlst + "    grbl     : enter in GRBL command to CNC\n"
cmdlst = cmdlst + "    q        : to quit program"

#   view video until quit command is entered for Arduino

time.sleep(2)
capture_image(0,'all')

if Eng_mode :
    Eng_mode = 0
    if Plt_mode :
        Plt_mode = 0
        calibrate_board()
        Eng_mode = 1
        Plt_mode = 1
    else :
        calibrate_board()
        Eng_mode = 1    
else :
    calibrate_board()
    
camera.start_preview()

Sq_image = np.copy(Inten)

strin = raw_input (prompt);
while (strin.lower() != "q"):
    if (strin.lower() == "im") :
        filename = raw_input ("Name of saved Image file? ")
        camera.capture(filename + '.jpg')   # store Image as JPEG
    elif (strin.lower() == "?") :
        print cmdlst
    elif (strin.lower() == "cal") :
        if Talk_mode : talk('OK, will go into calibration mode')
        capture_image(0,'all')
        camera.stop_preview()
        calibrate_board()
        camera.start_preview()
    elif (strin.lower() == "load_log") :
        log_fname = raw_input('What is name of log file to load? ')
        feat_log = np.loadtxt(log_fname,delimiter='\t')
    elif (strin.lower() == "make_log") :
        log_fname = raw_input('What is name of log file to create? ')
        np.savetxt(log_fname, feat_log, fmt='%1.4e',delimiter='\t',header= Img_feat_header)
    elif (strin.lower() == "sq_img") :
        camera.stop_preview()
        sq = raw_input('Which square to view (e.g. a6, e8) ? ')
        temp_plt_mode = Plt_mode
        Plt_mode = 1
            
        while len(sq) == 2 :
            capture_image(0,'all')
            code = get_square_image(sq,1)
            Img_feat[0,0] = input('Color of chess piece (0:black, 1:white) = ')
            Img_feat[0,1] = input('Chess piece ID (1: empty, 2: pawn, 3: knight, 4: bishop, 5: rook, 6: queen, 7: king) = ')
            feat_log = np.append(feat_log,Img_feat,axis = 0)
            np.savetxt(log_fname, feat_log, fmt='%1.2e',delimiter='\t',header= Img_feat_header)
            sq = raw_input('Which square to view (e.g. a6, e8) ? ')
            
        camera.start_preview()
        Plt_mode = temp_plt_mode
    elif (strin.lower() == "ref_b") :
        camera.stop_preview()
        update_chess_board()
        print_chess_board()
        camera.start_preview()
    elif (strin.lower() == "magon") :
        magnet_on_off(1)
    elif (strin.lower() == "magoff") :
        magnet_on_off(0)
    elif (strin.lower() == "laseron") :
        laser_on_off(1)
    elif (strin.lower() == "laseroff") :
        laser_on_off(0)
    elif (strin.lower() == "anal") :
        if Talk_mode : talk('Entering manual board analysis mode')
        camera.stop_preview()
        board = board_analysis()
        camera.start_preview()
    elif (strin.lower() == "mo") :
        if Talk_mode : talk('Entering into absolute coordinate move mode')
        absolute_coordinate_moves()
    elif (strin.lower() == "arm") :
        if Talk_mode : talk('Entering into calibrate arm location mode')
        camera.stop_preview()
        calibrate_arm_location()
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
