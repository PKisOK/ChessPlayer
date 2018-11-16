
#
#   Program to manually control Rajan's CNC using USB port
#   while view video from the RPi camera mounted in the CNC arm
#   and turning on and off the electromagnet using GPIO port to
#   move chess pieces using CNC
#
#   Added Stockfish chess program to code : 10/31/2018
#       Source code for chess program : www.chess.fortherapy.co.uk
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
#   sudo apt-get install stockfish   (installs Stockfish 5.0)
#       copy ChessBoard.py(.pyc) and Maxchessdemo.py into Chess sub directory from Windows PC
#       using Filezilla
#       additional recommended packages for chess are polyglot, xboard,update_chess_board scid
#       scid has graphical chessboard routines
#

import os
import picamera
import serial
import RPi.GPIO as GPIO
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time, subprocess
import math
import cv2

# initiate chessboard

from ChessBoard import ChessBoard
import subprocess, time
maxchess = ChessBoard()

# initiate stockfish chess engine

engine = subprocess.Popen(
    'stockfish',
    universal_newlines=True,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,)

def get():

    global Show_Stockfish_Analysis
    
    # using the 'isready' command (engine has to answer 'readyok')
    # to indicate current last line of stdout
    stx=""
    engine.stdin.write('isready\n')
#    print('\nengine:')
    while True :
        text = engine.stdout.readline().strip()
        if text == 'readyok':
            break
        if text !='':   
            if Show_Stockfish_Analysis :
                print('\t'+text)
        if text[0:8] == 'bestmove':        
            return text

def sget():

    global Show_Stockfish_Analysis
    
    # using the 'isready' command (engine has to answer 'readyok')
    # to indicate current last line of stdout
    stx=""
    engine.stdin.write('isready\n')
#    print('\nengine:')
    while True :
        text = engine.stdout.readline().strip()
        #if text == 'readyok':
         #   break
        if text !='':   
            if Show_Stockfish_Analysis :
                print('\t'+text)
        if text[0:8] == 'bestmove':
            mtext=text
            return mtext
def getboard():

    global Human_move
    
    """ gets a text string from the board """
    
    Human_move = raw_input("\n Make move and then enter CNC chess move (syntax ps1-s2 or ps1xps2) : ").lower()

    btxt = translate_CNC2Stockfish_move(Human_move)

    print 'CNC Move Notation = ',Human_move,' Stockfish Move Notation = ',btxt
    
#    raw_input("\n\nPress the enter key to continue")
    
    return btxt
    
def sendboard(stxt):
    """ sends a text string to the board """
    print("\n send to board: " +stxt)

def newgame():

    global Skill
    
    get ()
    put('uci')
    get ()
    put('setoption name Skill Level value ' +Skill)
    get ()
    put('setoption name Hash value 128')
    get()
    put('setoption name Best Book Move value true')
    get()
    put('setoption name OwnBook value true')
    get()
    put('uci')
    get ()
    put('ucinewgame')
    maxchess.resetBoard()
    fmove=""
    return fmove


def bmove(fmove):

    global Movetime
    global Computer_move, Human_move
    global B_message
    global Show_Stockfish_Analysis
    
    """ assume we get a command of the form ma1a2 from board"""    
    fmove=fmove
    # Get a move from the board
    brdmove = B_message[1:5].lower()
    # now validate move
    # if invalid, get reason & send back to board
      #  maxchess.addTextMove(move)
    if maxchess.addTextMove(brdmove) == False :
                        etxt = "error"+ str(maxchess.getReason())+brdmove
                        maxchess.printBoard()
                        sendboard(etxt)
                        return "error"
                       
#  elif valid  make the move and send Fen to board
    
    else:
        maxchess.printBoard()
        # maxfen = maxchess.getFEN()
        # sendboard(maxfen)
       # remove line below when working
#        raw_input("\n\nPress the enter key to continue")

        if Show_Stockfish_Analysis :
            print ("fmove")
            print(fmove)
            print ("brdmove")
            print(brdmove)

        fmove =fmove+" " +brdmove

        cmove = "position startpos moves"+fmove
        if Show_Stockfish_Analysis :
            print (cmove)

            #        if fmove == True :
            #                move = "position startpos moves "+move
            #        else:
            #               move ="position fen "+maxfen

        # put('ucinewgame')
        # get()

       
        put(cmove)
        # send move to engine & get engines move

        
        put("go movetime " +Movetime)
        # time.sleep(6)
        # text = get()
        # put('stop')
        text = sget()
        print (text)
        smove = text[9:13]
        
        Computer_move = translate_Stockfish2CNC_move(smove)
        print 'Computer move : ',Computer_move
        
        hint = text[21:25]
        if maxchess.addTextMove(smove) != True :
                        stxt = "e"+ str(maxchess.getReason())+move
                        maxchess.printBoard()
                        sendboard(stxt)

        else:
                        temp=fmove
                        fmove =temp+" " +smove
                        stx = smove+hint      
                        sendboard(stx)
                        maxchess.printBoard()
                        # maxfen = maxchess.getFEN()
                        print ("computer move: " +smove)
                        return fmove
        

def put(command):
#    print('\nyou:\n\t'+command)
    engine.stdin.write(command+'\n')


def talk(cmd):
    saycmd = "echo " + cmd + " | festival --tts"
    os.system(saycmd)
    return

def find_laser(box,diff) :
#
#   Find location of laser in image usings of rows and cols
#
#   box != 0 : then use smaller image box around last known laser location
#                   to look for laser spot
#   diff != 0   : image is a derivative image so use lower threshold to
#                   find laser spot

    global Image, Inten
    global Col_scal
    global Eng_mode, Plt_mode
    global Laser_row, Laser_col
    global Laser_actual_loc
    global Row_inches, Col_inches
    global Zero_location_row, Zero_location_col
    global Laser_window_col_width
    global Laser_window_row_width
    global Laser_box_col_min, Laser_box_col_max
    global Laser_box_row_min, Laser_box_row_max
    global Ncols,Nrows

    if box :
#
#   select image box size around previous known location of laser
#
        hy = Laser_col + Laser_window_col_width
        if hy > (Ncols-1) :
            hy = (Ncols-1)
        ly = Laser_col - Laser_window_col_width
        if ly < 0 :
            ly = 0
        lx = Laser_row - Laser_window_row_width
        if lx < 0 :
            lx = 0
        hx = Laser_row + Laser_window_row_width
        if hx > (Nrows-1) :
            hx = (Nrows-1)

        if diff :
            clip = 0.75
        else :
            clip = 0.85
        
        if Eng_mode :
            print 'Laser window ly,hy,lx,hx: ',ly,' ',hy,' ',lx,' ',hx

        Laser_box_col_min = ly
        Laser_box_col_max = hy
        Laser_box_row_min = lx
        Laser_box_row_max = hx

    else :
#
#   inital box size to find laser for first time
#
        ly = 330
        hy = (Ncols-1)
        lx = 0
        hx = (Nrows-1)
        if diff :
            clip = 0.75
        else :
            clip = 0.90

#
#   laser spot detection parameters
#
    min_no_pixels = 10
    if diff :
        max_no_pixels = 350
        max_avg_distance = 25.0
    else :
        max_no_pixels = 350
        max_avg_distance = 25.0
    window_radius = 100.0
    possible_error = 0
    
    crop_img = np.copy(Inten[lx:hx,ly:hy])       
    crop_row,crop_col = crop_img.shape
    mask_img = np.copy(crop_img)

    clip = np.max(mask_img)*clip+(1-clip)*np.min(mask_img)
    
    mask_img[mask_img<=clip] = 0.0
    mask_img[mask_img>clip ] = 1.0
    
    pixels = np.argwhere(crop_img>clip)
    marks = np.median(pixels,0)
    no_pixels = np.sum(np.sum(mask_img))
    pixel_std = np.std(pixels,axis = 0)

    if (no_pixels < 10)| (no_pixels > 340):
        print '*** possible error in laser finding routine ***'
        if Eng_mode :
            print '# of pixs ',no_pixels,' x std ',pixel_std[0],' y std ',pixel_std[1]*Col_scal
            print 'size of crop image : ',crop_row,',',crop_col
            print 'Marks : (',marks.shape,'),',marks
            print 'Abort laser find and keep last laser position'
        
            fig, ax = plt.subplots(nrows = 1, ncols = 2)
            ax[0].imshow(Image[lx:hx,ly:hy], aspect='equal', extent=[0,crop_col*Col_scal,0,crop_row], cmap='gray')      
            ax[1].imshow(mask_img, aspect='equal', extent=[0,crop_col*Col_scal,0,crop_row], cmap='gray')        
            plt.show()
            plt.close()       # free up memory
        else :
            return

        try:
            prompt = raw_input('Enter y to retry laser search using entire image : ')
            if prompt.lower() == 'y' :
                print '*** re-analyzing  fulliamge for laser location ***'

                ly = 330
                hy = (Ncols-1)
                lx = 0
                hx = (Nrows-1)
                if diff :
                    clip = 0.5
                else :
                    clip = 0.90

                min_no_pixels = 10
                if diff :
                    max_no_pixels = 350
                    max_avg_distance = 25.0
                else :
                    max_no_pixels = 350
                    max_avg_distance = 25.0
                window_radius = 100.0
                possible_error = 0
    
                crop_img = np.copy(Inten[lx:hx,ly:hy])       
                crop_row,crop_col = crop_img.shape
                mask_img = np.copy(crop_img)

                clip = np.max(mask_img)*clip+(1-clip)*np.min(mask_img)
    
                mask_img[mask_img<=clip] = 0.0
                mask_img[mask_img>clip ] = 1.0
    
                pixels = np.argwhere(crop_img>clip)
                marks = np.median(pixels,0)
                no_pixels = np.sum(np.sum(mask_img))
                pixel_std = np.std(pixels,axis = 0)

                print '** New results **'
                print '# of pixs ',no_pixels,' x std ',pixel_std[0],' y std ',pixel_std[1]*Col_scal
                print 'size of crop image : ',crop_row,',',crop_col
                print 'Marks : (',marks.shape,'),',marks

            else :
                print '*** ignoring error message ***'
        except ValueError:
            print '*** ignoring error message ***'          
        
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
    
    new_laser_row = np.median(laser_pixels[:,:,:,0])
    new_laser_col = np.median(laser_pixels[:,:,:,1])
    old_laser_row = Laser_row
    old_laser_col = Laser_col

    if (abs(Laser_row-new_laser_row-lx) > Laser_window_row_width) :
        possible_error = 1
    if (abs(Laser_col-new_laser_col-ly) > Laser_window_col_width) :
        possible_error = 1

    Laser_row = new_laser_row
    Laser_col = new_laser_col
    nof_pixels = int(np.shape(laser_pixels)[0])

    if nof_pixels > 0 :
        avg_dist = np.sum(laser_pixels, axis=0)[0,0,3] / nof_pixels
    else :
        avg_dist = 0.0
        possible_error = 1

    if (Eng_mode | possible_error):
        print '# of pixels {0:d} -> {1:d}, avg. distance = {2:.3f}'.format(int(no_pixels),int(nof_pixels),avg_dist)    
        print 'previous Laser mark (row,col)    : ',old_laser_row,' , ',old_laser_col
        print 'new      Laser mark (row,col   ) : ',(Laser_row+lx),' , ',(Laser_col+ly)
        print 'Laser detection window (row,col) : ',Laser_window_row_width,' , ',Laser_window_col_width

        if possible_error :
            print '*** Possible laser find error ***'

        if (Plt_mode | possible_error):
            fig, ax = plt.subplots(nrows = 1, ncols = 2)
            ax[0].imshow(Image[lx:hx,ly:hy], aspect='equal', extent=[0,crop_col*Col_scal,0,crop_row], cmap='gray')      
            ax[1].set_title('Laser Spot: ('+str(round(Laser_col*Col_scal,1))+','+str(round(crop_row-Laser_row,1))+')')   
            ax[1].imshow(mask_img, aspect='equal', extent=[0,crop_col*Col_scal,0,crop_row], cmap='gray')        
            plt.show()
            plt.close()       # free up memory

    Laser_row += lx
    Laser_col += ly
    
    if Eng_mode :
        print 'Laser mark : ',Laser_row,' , ',Laser_col
    
    return

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

    clip = input('Select clip level for masking Image (0.0 to 1.0) (<0.get_square_im0 for reverse mask): ')
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

  

def update_chess_board(init) :
#
#   updates the chess board position
#
#   init <> 0 : then initialize the board to starting chess position without analyzing
#         = 0 : then analyze chess board image and determine chess position
#

    global Chess_board
    global Computer_move, Human_move
    
    file_list = 'abcdefgh'
    rank_list = '12345678'
    chess_pc  = [5,3,4,6,7,4,3,5]

    if init <> 0 :
        capture_image(0,'board','all_scale')

    for f in range(8) :
        for r in range(8):
            if init :
                if r in [2,3,4,5] : code = [1,1]
                elif r == 0 : code = [2,chess_pc[f]]
                elif r == 7 : code = [3,chess_pc[f]]
                elif r == 1 : code = [2,2]
                else        : code = [3,2] 
            else :
                sq = file_list[f] + rank_list[r]
                code = get_square_image(sq,0)       # image processing routine
            Chess_board[f][r] = code
        
    return

def print_chess_board() :

    global Chess_board
    global Piece_code
    global Color_code
    
    rank_list = '12345678'

    print "  PI Robot board view "
    print "  +-----------------+"
    for r in range(7,-1,-1) :
        print rank_list[r],'|',
        for f in range(8):
            if (Color_code.upper()[Chess_board[f][r][0]] == 'W') :
                print Piece_code.upper()[Chess_board[f][r][1]],
            elif (Color_code.upper()[Chess_board[f][r][0]] == 'B') :
                print Piece_code.lower()[Chess_board[f][r][1]],
            else :
                print Piece_code.lower()[Chess_board[f][r][1]],
        print '|'
    print "  +-----------------+"
    print "    A B C D E F G H"  

    return
    

def calibrate_board():
#
#   Function does the following
#       get_square_im
#       Analyze chess board Image to physical locations of each square
#       determine Image rows/cols to physical distances and x/y scale factor
#
#
#   Naming Convention
#       Physical Y Axis = Image Column Axis = Chess Board Rank or File Axis Depending on Board Orientation
#       Physical X Axis = Image Row Axis    = Chess Board File or Rank Axis Depending on Board Orientation
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
    global Col_scal,Ncols,Nrows
    global Board
    global Squares_x_location
    global Squares_y_location
    global Chess_board
    global Eng_mode, Plt_mode
    global Boardcorner_row_min, Boardcorner_col_max
    global Boardcorner_row_max, Boardcorner_col_min
    global Board_O
    global Laser_window_col_width
    global Laser_window_row_width


    (rows,cols) = Inten.shape
    if Eng_mode : print 'Size of full Image: ',rows,',',cols

#       Chess board squares dimensions in inches
    no_xsqs = 8
    no_ysqs = 8
    sq_yd = 1.5
    sq_xd = 1.5 + 1.0/16.0/8
    board_border = 0.20

#   find absolute board corners and squares by looking at the bigger Image

    ly = int (0.25*Ncols)
    hy = int (0.80*Ncols)
    lx = int (0.1*Nrows)
    hx = int (0.9*Nrows)
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
        ax[1,1].set_title('peaks are Col Boundaries')
        
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
        print 'Col location of Y-Axis Boundary Peaks : ',dy_peaks,' Shape: ',dy_peaks.shape
        print 'Col Difference between Peaks : ',dy_peaks[1:]-dy_peaks[:-1]
        print 'Selected Peaks: ',Squares_y_location
        print '# of Y-Axis Squares detected: ',len(Squares_y_location)-1
 
    dx_array = abs(crop_img[1:,:]-crop_img[:-1,:])
    dx_sum = np.sum(dx_array,1)
    if Plt_mode :
        ax[2,0].imshow(dx_array, aspect='equal', extent=[0,crop_col*Col_scal,0,(crop_row-1)], cmap='gray')
        ax[2,0].set_title('dX Image')
        ax[2,1].plot(np.arange(crop_row-1),dx_sum)
        ax[2,1].set_title('peaks are Row Boundaries')
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
        print 'Row location of X-Axis Boundary Peaks : ',dx_peaks,' Shape: ',dx_peaks.shape
        print 'Row Difference Between Peaks: ',dx_peaks[1:]-dx_peaks[:-1]
        print 'Selected Peaks: ',Squares_x_location
        print '# of X-Axis Squares detected: ',len(Squares_x_location)-1
    if Plt_mode :
        plt.show()
        plt.close()       # free up memory
        
    Boardcorner_col_max = max(Squares_y_location)
    Boardcorner_row_min = min(Squares_x_location)
    Boardcorner_col_min = min(Squares_y_location)
    Boardcorner_row_max = max(Squares_x_location)
    
    Col_inches = no_ysqs*sq_yd/(Boardcorner_col_max-Boardcorner_col_min)
    Row_inches = no_xsqs*sq_xd/(Boardcorner_row_max-Boardcorner_row_min)
    Col_scal = Col_inches / Row_inches

    if   Board_O == 0 :     # human playing with white pieces from side of robot - zero is located at upper left of A8
        ranks_c = np.float16(Squares_x_location[1:]+Squares_x_location[:-1])*0.5
        ranks_c = ranks_c[np.arange((np.size(ranks_c)-1),-1,-1)]
        files_c = np.float16(Squares_y_location[1:]+Squares_y_location[:-1])*0.5
    elif Board_O == 1 :     # human playing with black pieces from side of robot - zero is located at lower right of H1
        ranks_c = np.float16(Squares_x_location[1:]+Squares_x_location[:-1])*0.5
        files_c = np.float16(Squares_y_location[1:]+Squares_y_location[:-1])*0.5
        files_c = files_c[np.arange((np.size(files_c)-1),-1,-1)]
    elif Board_O == 2 :     # human playing with white pieces from front of robot - zero is located at lower left of A1      
        files_c = np.float16(Squares_x_location[1:]+Squares_x_location[:-1])*0.5
        ranks_c = np.float16(Squares_y_location[1:]+Squares_y_location[:-1])*0.5
    elif Board_O == 3 :     # human playing with black pieces from front of robot - zero is located at upper right of H8
        files_c = np.float16(Squares_x_location[1:]+Squares_x_location[:-1])*0.5
        files_c = files_c[np.arange((np.size(files_c)-1),-1,-1)]
        ranks_c = np.float16(Squares_y_location[1:]+Squares_y_location[:-1])*0.5
        ranks_c = ranks_c[np.arange((np.size(ranks_c)-1),-1,-1)]

    if Eng_mode :   
        print 'File centers : ', files_c   
        print 'Rank centers : ', ranks_c
       
    files_c = np.resize(files_c, (8,8)).T
    ranks_c = np.resize(ranks_c, (8,8))

    if (Board_O == 0)|(Board_O == 1) :
        Board[:,:,0,0] = ranks_c
        Board[:,:,0,1] = files_c
        Board[:,:,1,0] = (ranks_c - Boardcorner_row_min) * Row_inches
        Board[:,:,1,1] = (files_c - Boardcorner_col_min) * Col_inches
    else :  
        Board[:,:,0,0] = files_c
        Board[:,:,0,1] = ranks_c
        Board[:,:,1,0] = (files_c - Boardcorner_row_min) * Row_inches
        Board[:,:,1,1] = (ranks_c - Boardcorner_col_min) * Col_inches

    Zero_location_row = Boardcorner_row_min
    Zero_location_col = Boardcorner_col_min
    
    if Eng_mode :   
        print 'A file ', Board[0,:,:,:]
        print 'H file ', Board[-1,:,:,:]
        print '1st rank ', Board[:,0,:,:]
        print '8th rank ', Board[:,-1,:,:]

        print 'Board Corners : ',Boardcorner_col_min,',',Boardcorner_row_max,',',Boardcorner_col_max,',',Boardcorner_row_min
        print 'Col_inches : ', Col_inches
        print 'Row_inches : ', Row_inches
        print 'Col_scal : ', Col_scal

    board_img = np.copy(Inten)
#
#   blank out board area
#
    board_img[Boardcorner_row_min:Boardcorner_row_max,Boardcorner_col_min:Boardcorner_col_max]=0.0
#
#   draw the lines for the chess squares
#
    board_img[Squares_x_location,:] = 1.0
    board_img[:,Squares_y_location] = 1.0
#
#       mark zero location on images
#
    board_img[(Zero_location_row-10):(Zero_location_row+10), (Zero_location_col-3):(Zero_location_col+3)] = 1.0
    crop_img[(Zero_location_row-lx-10):(Zero_location_row-lx+10), (Zero_location_col-ly-3):(Zero_location_col-ly+3)] = 1.0
#
#       mark square center on image
#
    for i in [-2,-1,0,1,2] :
        for j in [-2,-1,0,1,2] :
            crop_img[(Board[:,:,0,0]-lx+i).astype(int),(Board[:,:,0,1]-ly+j).astype(int)] = 1.0
    
    if (Plt_mode):
        plt.imshow(board_img, aspect='equal', extent=[0,cols*Col_scal,0,rows], cmap='gray')
        plt.show()
        plt.close()       # free up memory
    plt.imshow(crop_img, aspect='equal', extent=[0,crop_col*Col_scal,0,crop_row], cmap='gray')
    plt.show()
    plt.close()       # free up memory

    update_chess_board(0)
    print_chess_board()
#
#   image box size to search around previous known location of laser
#
    Laser_window_col_width = int(round(3.0/Col_inches,0))
    Laser_window_row_width = int(round(3.0/Row_inches,0))
        
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
    global Board_O
    global Col_scal

    file_list = 'abcdefgh'
    rank_list = '12345678'
    error = 0
    r_bord = 8      # row border
    c_bord = int(r_bord/Col_scal+0.5)      # col border

    square = square.lower()
    
    if len(square) == 2 :
        if square[0] in file_list :
            file_n = file_list.index(square[0])
            if Board_O == 0 :
                y_index = file_n
            elif Board_O == 1:
                y_index = 7-file_n
            elif Board_O == 2:
                x_index = file_n
            elif Board_O == 3:
                x_index == 7-file_n
            if square[1] in rank_list :
                rank_n = rank_list.index(square[1])
                row_c = Board[file_n,rank_n,0,0]
                col_c = Board[file_n,rank_n,0,1]
                if Board_O == 0 :
                    x_index = 7-rank_n
                elif Board_O == 1 :
                    x_index = rank_n
                elif Board_O == 2 :
                    y_index = rank_n
                elif Board_O == 3 :
                    y_index = 7-rank_n
                r1 = int (Squares_x_location[x_index] + r_bord)
                r2 = int (Squares_x_location[x_index+1] - r_bord)
                c1 = int (Squares_y_location[y_index] + c_bord)
                c2 = int (Squares_y_location[y_index+1] - c_bord)
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
#   Img_feat[0,11]: normalized kurtosis
#   Img_feat[0,12]: normalized range
#   Img_feat[0,13]: pixel footprint of object as % of total pixels in image as determined by image threshold analysis
#   Img_feat[0,14]: radius of the base of object as determined by the derivative image analysis
#   Img_feat[0,15]: radius of the crown of object as determined by the derivative image analysis
#   Img_feat[0,16]: radius of the waist of object as determined by the derivative image analysis
#   Img_feat[0,17]: reserved for future use
#   Img_feat[0,18]: reserved for future use
#   Img_feat[0,19]: reserved for future use
#
    global Sq_img
    global Eng_mode, Plt_mode
    global Img_feat, Img_feat_header
    global Log_fname, Feat_log

    
    empty_th  = 0.10
    low_kurt  = 20.0
    high_kurt = 120.0
    bk_pawn_kurt = 160.0
    bk_pawn_range = 1.7
    wh_pawn_dev = 0.5
    wh_pawn_range = 0.75

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
    Img_feat[0,10] = inten_range = np.max(Sq_img) - np.min(Sq_img)
    
    if inten_avg <> 0.0 :
        norm_dev = inten_dev / inten_avg
        norm_kurt = inten_kurt / inten_avg
        norm_range = inten_range / inten_avg
    else :
        norm_dev = 0.0
        norm_kurt = 0.0
        norm_range = 0.0

    Img_feat[0,9] = norm_dev
    Img_feat[0,11] = norm_kurt
    Img_feat[0,12] = norm_range

    if norm_dev < empty_th :        # square must be empty
        code = [1,1]
    elif Img_feat[0,3] == 0 :       # dark square
        if norm_kurt < low_kurt :
            code = [3,0]                # piece is dark
            if norm_range < bk_pawn_range :
                code = [3,2]            # piece is a dark pawn
        else :
            code = [2,0]                # piece is light
            if norm_dev < wh_pawn_dev :
                code = [2,2]            # piece is a light pawn
    else :
        if norm_kurt > high_kurt :  # light square
            code = [3,0]                # piece is dark
            if inten_kurt > bk_pawn_kurt :
                code = [3,2]            # piece is a dark pawn
        else :
            code = [2,0]                # piece is light                
            if norm_range < wh_pawn_range :
                code = [2,2]            # piece is a light pawn

    if Eng_mode :
        print 'Avg. Inten: {0:.3f}, Std. Inten: {1:.3f}, Norm Std: {2:.3f}'.format(inten_avg,inten_dev,norm_dev)
        print 'Kurtosis: {0:.3f}, Norm Kurtosis: {1:.3f} '.format(inten_kurt,norm_kurt)
        print 'Range : {0:.3f}, Norm Range : {1:.3f} '.format(inten_range, norm_range)
        print 'Max inten: {0:.3f}, Min inten: {1:.3f}'.format(np.max(Sq_img),np.min(Sq_img))
        print 'Code : ',code
        Img_feat[0,0] = 0
        Img_feat[0,1] = 0
        Feat_log = np.append(Feat_log,Img_feat,axis = 0)
        np.savetxt(Log_fname, Feat_log, fmt='%1.2e',delimiter='\t',header= Img_feat_header)
        
    if Plt_mode :
        clip = plot_inten_hist(Sq_img)    
        plot_derivative_img(Sq_img,1)           

    return code

def get_z_height_info() :
#
#   get user input on Z height
#
    global Laser_height_0
    global Loc_cur
    global XYZ_limits
    
    try:
        laser_ht = input('Height of laser above board plane (inches) ('+str(Loc_cur[2] + Laser_height_0)+'): ')
    except :
        laser_ht = Loc_cur[2] + Laser_height_0
        
    if laser_ht < Laser_height_0 :
        laser_ht = Laser_height_0
        print 'Laser ht. is invalid.  Will use min laser ht. of {0:.2f}'.format(laser_ht)
        abort = input('Abort by hitting enter or type any character to contineu : ')
    if laser_ht > (Laser_height_0 + XYZ_limits[2][1]) :
        laser_ht = Laser_height_0 + XYZ_limits[2][1]
        print 'Laser ht. is invalid.  Will use max laser ht. of {0:.2f}'.format(laser_ht)
        abort = input('Abort by hitting enter or type any character to contineu : ')

    return laser_ht

def calibrate_arm_location(prompt, box, static) :
#
#       prompt <> 0 : will prompt user for Z height
#       box    =  0 : will search entire camera view area
#                else search in a box around last know laser location
#       static <> 0 : use difference in laser on vs off image
#                   else keep laser on and use one image to find laser location

    global Loc_cur, Loc_temp
    global Z_loc_apparent
    global Magnet2laser_offset
    global Magnet_actual_loc
    global Laser_actual_loc, Laser_apparent_loc
    global Camera_height
    global Laser_height_0
    global Row_inches, Col_inches
    global Zero_location_col
    global Zero_location_row
    global Col_scal
    global Laser_col, Laser_row
    global Laser_window_col_width
    global Laser_window_row_width
    global Arm_ht_offset
    global XYZ_limits
    global Image, Inten

#
#   turn on laser and capture image
#
    GPIO.output(22, 1)      # turn on laser
    if static :         # if arm is motionless, use differential image of laser on vs off to detect laser location
        capture_image(1,'all','no_scale')
        Image = np.copy(Inten)
        GPIO.output(22, 0) # turn off laser
        Inten_laser_on = np.copy(Inten)
        capture_image(1,'all','no_scale')
        Inten = Inten_laser_on - Inten
        Inten = (Inten-np.min(Inten))/(np.max(Inten) - np.min(Inten))
        Inten[Inten>1.0] = 1.0
        Inten[Inten<0.0] = 0.0        
        find_laser(box,1)
    else :      
        capture_image(1,'all','laser_scale')
        Image = np.copy(Inten)
        GPIO.output(22, 0)  # turn off laser
        find_laser(box,0)

    if Eng_mode :
        print 'Zero row,col  : {0:.3f} , {1:.3f}'.format(Zero_location_row,Zero_location_col)
        print 'Laser row,col : {0:.3f} , {1:.3f}'.format(Laser_row,Laser_col)
    
    Laser_apparent_loc[0] = (Laser_row-Zero_location_row) * Row_inches
    Laser_apparent_loc[1] = (Laser_col-Zero_location_col) * Col_inches

    if Eng_mode :
        print 'Laser location (x,y) {0:.3f} , {1:.3f} : '.format(Laser_apparent_loc[0],Laser_apparent_loc[1])
#
#   confirm Z height of laser if prompt option is enabled, otherwise use existing laser_ht information
#
    if (prompt) :
        laser_ht = get_z_height_info()
    else :
        laser_ht = Loc_cur[2] + Laser_height_0
        
    Loc_cur[2] = laser_ht - Laser_height_0
#
#   calculate X0 and Y0 from geometry of laser location
#
    theta = math.pi*0.5-math.acos(Camera_height/(Camera_height**2+(Laser_apparent_loc[0]-6)**2+(Laser_apparent_loc[1]-6)**2)**0.5)
    
    if (Laser_apparent_loc[1] < 6) :
        theta = -theta
    if (Laser_apparent_loc[1] <> 6) :
        phi   = math.atan((Laser_apparent_loc[0]-6)/(Laser_apparent_loc[1]-6))
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

    shadow_apparent = ((Laser_apparent_loc[0]-Loc_cur[0])**2+(Laser_apparent_loc[1]-Loc_cur[1])**2)**0.5
    Z_loc_apparent = shadow_apparent * math.tan(theta) - Laser_height_0

    if Eng_mode :
        print 'Current Z loc : {0:.2f}, Apparent Z loc : {1:.2f}'.format(Loc_cur[2],Z_loc_apparent)
    
    laser_zero_plane_loc = Magnet_actual_loc
    laser_zero_plane_loc[0] = Laser_apparent_loc[0] - shadow*math.sin(phi)
    laser_zero_plane_loc[1] = Laser_apparent_loc[1] - shadow*math.cos(phi)
    laser_zero_plane_loc[2] = 0.0

    if Eng_mode :
        print 'prev magnet loc ',Magnet_actual_loc
        print 'mag2laser offset ',Magnet2laser_offset
    
    Magnet_actual_loc = np.add(laser_zero_plane_loc, Magnet2laser_offset)
    Magnet_actual_loc[2] += laser_ht
    Magnet_actual_loc[2] += Arm_ht_offset*Magnet_actual_loc[0]
    Laser_actual_loc = laser_zero_plane_loc
    Laser_actual_loc[2] += laser_ht

    if Eng_mode :
        print 'new magnet loc ',Magnet_actual_loc
        print 'new laser loc ',Laser_actual_loc
    
    if static == 0 :
        Loc_temp[0] = Magnet_actual_loc[0]
        Loc_temp[1] = Magnet_actual_loc[1]        
    else :
        Loc_cur[0] = Magnet_actual_loc[0]
        Loc_cur[1] = Magnet_actual_loc[1]
        Loc_temp[0] = Magnet_actual_loc[0]
        Loc_temp[1] = Magnet_actual_loc[1]        

    return

def act_on_chesspiece(action, piece) :

#
#   move pickup or place chess piece
#
#       action = 'pickup_cap'
#                'pickup'
#                'place'
#                'up'
#
#       piece = 'k','q','r','b','n','p'
#
    global Piece_hts
    global Hover_ht, Move_ht, Capture_ht
    global Magnet2laser_offset
    global Laser_height_0
    global Arm_ht_offset
    global Slew_rate
    global Loc_cur, Loc_pre

    piece_lst = 'pnbrqk'

    p = piece[0].lower()
    
    if p in piece_lst :
        n = piece_lst.index(p)
        if action == 'up' :
            zloc = Hover_ht - Laser_height_0 - Magnet2laser_offset[2]

            if Eng_mode :
                print 'Moving to (',zloc,')'

            if (abs(Loc_cur[2]-zloc)> 0.05) :
                loc = [zloc]
                move2loc('Z',loc)

        elif (action == 'pickup') | (action == 'place') | (action == 'pickup_cap') :   

            zloc = Piece_hts[n] - Laser_height_0 - Magnet2laser_offset[2] + Arm_ht_offset*Loc_cur[0]        
            if Eng_mode :
                print 'Moving to zloc = {0:.3f} ',zloc
            
            if (abs(Loc_cur[2]-zloc)> 0.05) :
                loc = [zloc]
                move2loc('Z',loc)

            if action == 'pickup' :
                magnet_on_off(1)
                if n == 1 :     #  for knight move higher
                    zloc = Move_ht  - Laser_height_0 - Magnet2laser_offset[2]
                else :          #  for other pieces move lower
                    zloc = Hover_ht - Laser_height_0 - Magnet2laser_offset[2]
            elif action == 'pickup_cap' :
                magnet_on_off(1)
                zloc = Capture_ht - Laser_height_0 - Magnet2laser_offset[2]                                    
            else :
                magnet_on_off(0)
                zloc = Hover_ht - Laser_height_0 - Magnet2laser_offset[2]
            
            if Eng_mode :
                print 'Moving to zloc = {0:.3f} ',zloc        
            loc = [zloc]
            move2loc('Z',loc)
              
    else :
        print 'Invalid chess piece : ',piece
                                     
    return

def move2square(square) :

#
#   move arm to chess square
#
    global Board
    global Loc_cur
    global Two_step_motion

    file_list = 'abcdefgh'
    rank_list = '12345678'
    
    if len(square) == 2 :
        if square[0] in file_list :
            file_n = file_list.index(square[0])
        if square[1] in rank_list :
            rank_n = rank_list.index(square[1])
        
        xloc = Board[file_n,rank_n,1,0]
        yloc = Board[file_n,rank_n,1,1]

        if Eng_mode :
            print 'Moving to square {0:.2s} located at {1:.3f}, {2:.3f}'.format(square,xloc,yloc)
            
#      In 2 step mode, move 4/5 way to allow recalibration of location and reduce backlash or sticking effects

        if Two_step_motion :
            move2loc('XY',[(4.0*xloc+Loc_cur[0])*0.2,(4.0*yloc+Loc_cur[1])*0.2])
        
#       then move to actual location
        
        loc = [xloc,yloc]
        move2loc('XY',loc)

        if Laser_backlash_correction :
            count = 0
            while ((abs(xloc-Loc_cur[0])>0.0625) | (abs(yloc-Loc_cur[1])>0.0625)) & (count < 10) :    
                print 'Backlash correct #',(count+1)
                move2loc('XY',[(4.0*xloc+Loc_cur[0])/5.0,(yloc*4.0+Loc_cur[1])/5.0])  # adjust position for backlash
                count += 1
                
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
#          Simultaneous x,y,z axis motion not currently allowed
#
    global XYZ_limits
    global Loc_cur, Loc_pre, Loc_temp
    global Z_loc_apparent, Laser_apparent_loc
    global CNC_scale
    global Backlash, Laser_backlash_correction
    global Slew_rate
    global Laser_row,Laser_col
    global Laser_box_col_min, Laser_box_col_max
    global Laser_box_row_min, Laser_box_row_max
    global Col_scal

    axis_lst = 'XYZ'
    cmd_str = ''
    xloc = Loc_cur[0]
    yloc = Loc_cur[1]
    last_laser_row = Laser_row
    last_laser_col = Laser_col
    waitxy = 0

    if Eng_mode :
        print 'Move2loc cmd : ',axis,' Loc = ',loc

    if len(axis) < 3 :
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
                    if Eng_mode :
                        print '+ backlash correction applied to {0:.2s} axis motion'.format(ax)
                elif ( (Loc_cur[n] > Loc_pre[n]) & (value < 0) ) :
                    value = value - Backlash[n][1]
                    if Eng_mode :
                        print '- backlash correction applied to {0:.2s} axis motion'.format(ax)
            

                Loc_pre[n] = Loc_cur[n]

                GRBLvalue = round(value/CNC_scale[n],2)

                if n == 0 :
                    cmd_str = cmd_str+' Y'+repr(GRBLvalue)
                    Loc_cur[0] = loc[i]
                    waitxy = 1
                elif n == 1 :
                    cmd_str = cmd_str+' X'+repr(-GRBLvalue)
                    Loc_cur[1] = loc[i]
                    waitxy = 1
                elif n == 2 :
                    cmd_str = cmd_str+' Z'+repr(GRBLvalue)
                    zloc = loc[i]
                    Loc_cur[n] = loc[i]
                    if len(axis) > 1 :
                        print '*** Cannot make a Z axis move simulatenously with X or Y motion ***'
                        print '*** Motion Command "'+axis+'" not valid *** '
                        return 
        
            else :
                print '*** Motion Axis "'+axis[i]+'" not valid *** '
    else :        
        print '*** Motion Command "'+axis+'" not valid *** '
        return         

#    calibrate_arm_location(0,1,1)
            
    cmd_str = 'G91 G0'+cmd_str+'\r\n'

    if Eng_mode :
        print 'GRBL cmd : '+cmd_str,
    ser.write(cmd_str)        
    reply = ser.readline()
    if Eng_mode :
        print 'CNC Reply back: ',reply,
        
#
#       wait till arm has reached target location for x and y motion only 
#
    if waitxy :
        print 'Waiting for XY motion to complete',

        max_motion_time = abs(Loc_cur[0]-Loc_pre[0])/Slew_rate[0] + abs(Loc_cur[1]-Loc_pre[1])/Slew_rate[1] + 2.0
#        max_motion_time *= 0.5
        dx_thres = abs(Loc_cur[0]-Loc_pre[0])*0.5
        dy_thres = abs(Loc_cur[1]-Loc_pre[1])*0.5
        
        start_time = time.clock()
        time_count = 0.0
        motion = 1
        last_x_loc = Loc_cur[0]
        last_y_loc = Loc_cur[1]
        Loc_temp[0] = 0.0
        Loc_temp[1] = 0.0

        if Eng_mode :
            print 'dx_thres: {0:.2f}, dy_thres: {1:.2f}'.format(dx_thres,dy_thres)
            print 'Loc_cur : {0:.2f}, {1:.2f} Loc_temp : {2:.2f}, {3:.2f}'.format(Loc_cur[0],Loc_cur[1],Loc_temp[0],Loc_temp[1])

#   make sure motion has started by checking on laser location to insure motion as occured

        while ( (abs(Loc_cur[0]-Loc_temp[0])> dx_thres) | (abs(Loc_cur[1]-Loc_temp[1])> dy_thres) ) & (time_count < max_motion_time) :
            
            calibrate_arm_location(0,1,0)
            
            if Eng_mode :
                print 'moving to target loc: ({0:.3f}->{1:.3f}),({2:.3f}->{3:.3f})'.format(Loc_cur[0],Loc_cur[0],Loc_cur[1],Loc_cur[1])
            else :
                print '.',

            time.sleep(1)
            time_count = (time.clock()-start_time)*1.0

        if (time_count > max_motion_time) & Eng_mode :
            print '*** move timed out ***'
            print 'Failed move from current loc to target loc: ({0:.2f}->{1:.2f}),({2:.2f}->{3:.2f})'.format(Loc_cur[0],Loc_cur[0],Loc_cur[1],Loc_cur[1])
            print 'in {0:.3f} seconds.'.format(time_count)

#   insure motion has completed by seeing if laser is moving

        last_x_loc = Loc_temp[0]
        last_y_loc = Loc_temp[1]
        while motion :
            calibrate_arm_location(0,1,0)
            if (abs(last_x_loc-Loc_temp[0])<0.05) & (abs(last_y_loc-Loc_temp[1])<0.05) :
                motion = 0
            last_x_loc = Loc_temp[0]
            last_y_loc = Loc_temp[1]
            print '+',
                        
        calibrate_arm_location(0,1,1)
#        Loc_cur[0] = Loc_temp[0]
#        Loc_cur[1] = Loc_temp[1]

        print ' '    
        print '** motion completed **',
        if Eng_mode :
            print ' New loc : ({0:.2f}, {1:.2f}, {2:.2f})'.format(Loc_cur[0],Loc_cur[1],Loc_cur[2])
        else :
            print ' '
    else :
        print 'Waiting for Z motion to complete ',
        
        max_motion_time = abs(zloc-Loc_pre[2])/Slew_rate[2] + 10
        time_count = 0.0
        motion = 1
        capture_image(0,'all','no_scale')
        ref_img = np.copy(Inten[Laser_box_row_min:Laser_box_row_max,Laser_box_col_min:Laser_box_col_max])
        ref_row,ref_col = ref_img.shape

        if ((Loc_cur[0]-6.0)**2 + (Loc_cur[1]-6.0)**2)< 4.0 :
            image_jitter = 0.0005           # for locations beneath camera, smaller jitter threshold for lack of Z motion
        elif ((Loc_cur[0]-6.0)**2 + (Loc_cur[1]-6.0)**2)> 16.0 :
            image_jitter = 0.002           # for locations far from underneath camera, larger jitter threshold for lack of Z motion            
        else :
            image_jitter = 0.001

        time.sleep(3)           # wait for arm to start moving
        start_time = time.clock()
        
        while motion :
            capture_image(0,'all','no_scale')
            test_img = np.copy(Inten[Laser_box_row_min:Laser_box_row_max,Laser_box_col_min:Laser_box_col_max])
            diff_img = np.copy (test_img-ref_img)
            img_dif_res = abs(np.sum(diff_img, dtype=float))
            ref_sum = abs(np.sum(ref_img, dtype = float))            

            if Eng_mode :
                print 'Ref Sum : {0:.6f}, Test Sum : {1:.6f}, Diff Sum : {2:.6f}'.format(abs(np.sum(ref_img,dtype=float)),abs(np.sum(test_img, dtype=float)),img_dif_res)
                fig, ax = plt.subplots(nrows = 1, ncols = 3)
                ax[0].set_title('Reference Image')   
                ax[0].imshow(ref_img, aspect='equal', extent=[0,ref_col*Col_scal,0,ref_row], cmap='gray')      
                ax[1].set_title('Test Image')   
                ax[1].imshow(test_img, aspect='equal', extent=[0,ref_col*Col_scal,0,ref_row], cmap='gray')      
                ax[2].set_title('Difference Image')   
                ax[2].imshow(diff_img, aspect='equal', extent=[0,ref_col*Col_scal,0,ref_row], cmap='gray')      
                plt.show()
                plt.close()       # free up memory
            
            ref_img = np.copy(test_img)
            
            time_count = (time.clock()-start_time)*60.0
            time.sleep(1)
            
             
            if (img_dif_res/ref_sum) < image_jitter :
                motion = 0

            if Eng_mode :
                print 'moving to target zloc: ({0:.3f}->{1:.3f})'.format(Loc_cur[2],zloc)
                print 'Ref sum : {0:.3f}, Diff sum : {1:.3f}, Diff/Ref : {2:.5f}'.format(ref_sum,img_dif_res,img_dif_res/ref_sum)
            else :
                print '%',
                
   #         print 'Time count : {0:.1f},  Max count : {1:.1f}'.format(time_count,max_motion_time)
            
   #         if (time_count > max_motion_time) :
   #             motion = 0

   #     if (time_count > max_motion_time) :
   #         print '*** move timed out ***'
   #         print 'Failed move to target zloc ({0:.3f}->{1:.3f}) within {2:.3f} seconds'.format(Loc_cur[2],zloc, time_count)
                        
        print ' '    
        print '** motion completed **',
        if Eng_mode :
            print ' New loc : ({0:.2f}, {1:.2f}, {2:.2f})'.format(Loc_cur[0],Loc_cur[1],Loc_cur[2])
        else :
            print ' '
    return

def go2home_loc()   :
#
#   move arm to home location to wait for next chess move
#

    move2loc('xy',[6.0,12.0])
    
    return


def chess_move(move,black_pc)    :
#
#   takes chess move command in full algebraic notation and
#   executes moves
#
#       e.g.  Nb1-c3  moves night on b1 to c3 square
#             Qd1xBh4  queen capture bishop on h4 square by first removing
#                       piece on h4 square and then moving queen to h4
#

    global Hover_ht, Move_ht, Capture_ht
    global Captured_piece_loc
    global Laser_actual_loc, Magnet_actual_loc
    global Laser_height_0
    global Magnet2laser_offset
    global Slew_rate
    global Loc_cur

    file_list = 'abcdefgh'
    rank_list = '12345678'
    pc_list   = 'pnbrqk'
    
    if len(move) in [6,7] :
        if (Human_manual == 0) | black_pc :
            zloc = Hover_ht - Laser_height_0 - Magnet2laser_offset[2]
            if (abs(Loc_cur[2]-zloc)> 0.05) :
                move2loc('z',[zloc])
            
        p_m = move[0].lower()
        sq0 = move[1:3].lower()

        if len(move) == 7 :     # move involves a piece capture
            p_c = move[4].lower()
            sq1 = move[5:7].lower()
            if (Human_manual == 0) | black_pc :       
                move2square(sq1)
                act_on_chesspiece('pickup_cap',p_c)
                move2loc('xy', Captured_piece_loc)
                calibrate_arm_location(1,0,1)
                move2loc('xy', Captured_piece_loc)
                act_on_chesspiece('place',p_c)
                Captured_piece_loc[0] -= 1.0
                if Captured_piece_loc[0]< 0.5 :
                    Captured_piece_loc[0] = 6.0
                    Captured_piece_loc[1] += 1.0
        else :
                sq1 = move[4:6].lower()
                
        if (Human_manual == 0) | black_pc :       
            move2square(sq0)
            act_on_chesspiece('pickup',p_m)
            move2square(sq1)
            act_on_chesspiece('place',p_m)
            go2home_loc()

#
# update chess board
#
        if sq0[0] in file_list :
            f = file_list.index(sq0[0])
        if sq0[1] in rank_list :
            r = rank_list.index(sq0[1])        
        Chess_board[f][r] = [1,1]       # mark square as being empty

        if sq1[0] in file_list :
            f = file_list.index(sq1[0])
        if sq1[1] in rank_list :
            r = rank_list.index(sq1[1])
        pc = pc_list.index(p_m)+2
        if black_pc :                   # update new location of piece
            Chess_board[f][r] = [3,pc]
        else :
            Chess_board[f][r] = [2,pc]
        
    else :
        print 'Chess move : ('+move+') not recoqnized'
        
    return


def capture_image (img_type,scope,scale) :

#
#   captures a new Image and returns a normalized Image file
#
#       img_type = 0, normalized intensity image
#       img_type = 1, normalized red color image
#       img_type = 2, normalized green color image
#       img_type = 3, normalized blue color image
#
#       scope = 'all' : capture entire image
#               'board' : capture only board image
#
#       scale = 'all_scale' : rescale image using max and min from entire image
#               'laser_scale' : rescale iamge using only max and min from box around last known position of laser
#               'no_scale'  : do not rescale image
#
    global Image, Inten
    global Ncols, Nrows
    global Boardcorner_row_min, Boardcorner_col_max
    global Boardcorner_row_max, Boardcorner_col_min
    global Laser_window_col_width
    global Laser_window_row_width
    global Laser_row, Laser_col
    
    camera.capture('binary.rgb',format = 'rgb', resize = (Ncols,Nrows))
#   camera.capture('binary.rgb',format = 'rgb')
    Image = np.fromfile('binary.rgb',np.uint8, -1, '')
    Image = Image.reshape(Nrows,Ncols,3)
    
    if img_type in [1,2,3] :
        Inten = np.float16(Image[:,:,img_type-1])
    else :                              # convert RGB to luminosity gray scale 
        Inten = np.float16(Image[:,:,0])*0.3 + np.float16(Image[:,:,1])*0.59 + np.float16(Image[:,:,2])*0.11

    if str.lower(scale) == 'laser_scale' :        
        hy = Laser_col + Laser_window_col_width
        if hy > (Ncols-1) :
            hy = (Ncols-1)
        ly = Laser_col - Laser_window_col_width
        if ly < 0 :
            ly = 0
        lx = Laser_row - Laser_window_row_width
        if lx < 0 :
            lx = 0
        hx = Laser_row + Laser_window_row_width
        if hx > (Nrows-1) :
            hx = (Nrows-1)
        scale_img = Inten[lx:hx,ly:hy]
    else :
        scale_img = Inten

    if str.lower(scale) != 'no_scale' :  
        Inten = (Inten-np.min(scale_img))/(np.max(scale_img) - np.min(scale_img))
        Inten[Inten>1.0] = 1.0
        Inten[Inten<0.0] = 0.0

    if scope == 'board' :
        board_max = np.max(Inten[Boardcorner_row_min:Boardcorner_row_max,Boardcorner_col_min:Boardcorner_col_max])
        board_min = np.min(Inten[Boardcorner_row_min:Boardcorner_row_max,Boardcorner_col_min:Boardcorner_col_max])
        Inten[Boardcorner_row_min:Boardcorner_row_max,Boardcorner_col_min:Boardcorner_col_max] = (Inten[Boardcorner_row_min:Boardcorner_row_max,Boardcorner_col_min:Boardcorner_col_max]-board_min)/(board_max - board_min)

    return

def magnet_on_off(value) :

    if value == 1 :
        if Eng_mode :
            if Talk_mode : talk('Turning on magnet')
            print 'Turning on magnet'
        GPIO.output(38, 1)
        GPIO.output(18, 1)
        time.sleep(0.1)
    else :
        if Eng_mode :
            if Talk_mode : talk('Turning off magnet')
            print 'Turning off magnet'
        GPIO.output(38, 0)
        GPIO.output(18, 0)
        time.sleep(0.1)
    
    return

def laser_on_off(value) :

    if value == 1 :
        if Talk_mode & Eng_mode : talk('Turning on laser')
        GPIO.output(22, 1)
    else :
        if Talk_mode & Eng_mode : talk('Turning off laser')
        GPIO.output(22, 0)
    
    return

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
    cmdlst = cmdlst + "    v  : capture new Image and view the full Imaupdate_chess_boardge in intensity scale\n"
    cmdlst = cmdlst + "    vr  : capture new Image and view the full Imget_square_image in red scale\n"
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
            capture_image(0,'all','all_scale')
            plt.imshow(Inten, aspect='equal', extent=[0,cols*Col_scal,0,rows], cmap='gray')
            plt.show()
            plt.close()       # free up memory
            low_x = 0
            low_y = 0
            hi_x = int(cols*Col_scal)
            hi_y = rows
            crop_img = np.copy(Inten)
        elif (strin.lower() == "vr") :
            capture_image(1,'all','all_scale')
            plt.imshow(Inten, aspect='equal', extent=[0,cols*Col_scal,0,rows], cmap='Reds_r')
            plt.show()
            plt.close()       # free up memory
            low_x = 0
            low_y = 0
            hi_x = int(cols*Col_scal)
            hi_y = rows
            crop_img = np.copy(Inten)
        elif (strin.lower() == "vg") :
            capture_image(2,'all','all_scale')
            plt.imshow(Inten, aspect='equal', extent=[0,cols*Col_scal,0,rows], cmap='Greens_r')
            plt.show()
            plt.close()       # free up memory
            low_x = 0
            low_y = 0
            hi_x = int(cols*Col_scal)
            hi_y = rows
            crop_img = np.copy(Inten)
        elif (strin.lower() == "vb") :
            capture_image(3,'all','all_scale')
            plt.imshow(Inten, aspect='equal', extent=[0,cols*Col_scal,0,rows], cmap='Blues_r')
            plt.show()
            plt.close(ax)       # free up memory
            low_x = 0
            low_y = 0
            hi_x = int(cols*Col_scal)
            hi_y = rows
            crop_img = np.copy(Inten)
        elif (strin.lower() == "chess") :
            update_chess_board(1)
            print_chess_board()
        elif (strin.lower() == "sq") :
            sq = raw_input('Which square to view and create cropped image (e.g. a6, e8) ? ')
            if len(sq) == 2 :
                code = get_square_image(sq,0)
                crop_img = np.copy(Sq_img)
            else :
                print 'Square (',sq,') not recognized'
        elif (strin.lower() == "fl") :
            find_laser(0,0)
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

def get_pc_code_square(sq) :

    global Chess_board

    file_list = 'abcdefgh'
    rank_list = '12345678'

    if sq[0] in file_list :
        f = file_list.index(sq[0])
    if sq[1] in rank_list :
        r = rank_list.index(sq[1])        
    code = Chess_board[f][r][1]

    return code

def translate_Stockfish2CNC_move(move) :

    pc_list = 'pnbrqk'

    move = move.lower()

    pc = get_pc_code_square(move[0:2])
    if pc > 1 :
        pc_label = pc_list[pc-2]
    else :
        print 'no piece located on square selected for move'
        pc_label = 'x'
        
    t_move = pc_label + move[0:2]
        
    if (get_pc_code_square(move[2:4]) == 1) :
        t_move = t_move + '-' + move[2:4]
    else :
        pc = get_pc_code_square(move[2:4])
        if pc > 1 :
            pc_label = pc_list[pc-2]
        else :
            print 'piece '+move[2:4]+' not recognized'
            pc_label = 'x'
        t_move = t_move + 'x' + pc_label + move[2:4]

    return t_move

def translate_CNC2Stockfish_move(move) :

    move = move.lower()

    if len(move) == 6 :
        t_move = 'm'+move[1:3]+move[4:6]
    elif len(move) == 7 :
        t_move = 'm'+move[1:3]+move[5:7]
    else :
        t_move = move

    return t_move


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
    cmdlst = cmdlst + "    up : move to magnet to hover position above board\n"
    cmdlst = cmdlst + "    sq : move to chess square\n"
    cmdlst = cmdlst + "    pick : pick up chess piece\n"
    cmdlst = cmdlst + "    place : place chess piece\n"
    cmdlst = cmdlst + "    loc : display current location\n"
    cmdlst = cmdlst + "    chess : enter a chess move command\n"    
    cmdlst = cmdlst + "    arm : calibrate arm position using laser (no Z ht. prompt)\n"    
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
        elif (strin.lower() == "up") :
            act_on_chesspiece('up','q')
        elif (strin.lower() == "pick") :
            pc = raw_input ('Piece to be picked up ("k","q","r","b","n","p") : ')
            act_on_chesspiece('pickup',pc[0])
        elif (strin.lower() == "place") :
            pc = raw_input ('Piece to be placed ("k","q","r","b","n","p") : ')
            act_on_chesspiece('place',pc[0])
        elif (strin.lower() == "chess") :
            move = raw_input('Enter chess move (e.g. Qd8-g6, Bc1xNh6) ')
            move = move.lower()
            chess_move(move,0)
        elif (strin.lower() == "loc")  :
            print 'Current X,Y,Z location : {0:.3f}, {1:.3f}, {2:.3f}'.format(Loc_cur[0],Loc_cur[1],Loc_cur[2])           
            print 'Current Laser location : {0:.3f}, {1:.3f}, {2:.3f}'.format(Laser_actual_loc[0],Laser_actual_loc[1],Laser_actual_loc[2])
            print 'Current Magnet location: {0:.3f}, {1:.3f}, {2:.3f}'.format(Magnet_actual_loc[0],Magnet_actual_loc[1],Magnet_actual_loc[2])        
        elif (strin.lower() == "arm") :
            calibrate_arm_location(0,1,1)
            print 'Current X,Y,Z location : {0:.3f}, {1:.3f}, {2:.3f}'.format(Loc_cur[0],Loc_cur[1],Loc_cur[2])
            print 'Current Laser location : {0:.3f}, {1:.3f}, {2:.3f}'.format(Laser_actual_loc[0],Laser_actual_loc[1],Laser_actual_loc[2])
            print 'Current Magnet location: {0:.3f}, {1:.3f}, {2:.3f}'.format(Magnet_actual_loc[0],Magnet_actual_loc[1],Magnet_actual_loc[2])
        else :
            print cmdlst
        strin = raw_input (prompt);
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
Human_manual = 1    # human will manually make his move
Human_move = 'pe2-e4'
Computer_move = 'pe7-e6'
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
#   Stage locations as tracked by X0, Y0 and Z0 are kept within these limits
#   Locations are calibrated during the setup phase
#
XYZ_limits = [[0.0,12.0],[0.0,15.0],[0.0,3.50]]
#       CNC backlash correction in inches for X,Y,Z (neg. to pos. , pos. to neg)
#Backlash = [[0.06, 0.06],[0.06,0.06],[0.0,0.0]]
Backlash = [[0.00, 0.00],[0.00,0.00],[0.0,0.0]]
#       CNC GRBL value conversion to inches
Y_scale = 0.97/25.4
X_scale = 1.01/25.4
Z_scale = 1.0/25.4
CNC_scale = [X_scale,Y_scale,Z_scale]
#       Vertical Distance (inches) of Pi Camera above center of Chess board
Camera_height = 24.0 + 1.0/4.0 + 1.0/4.0
#       offset of laser location from magnet in x,y,z (inches)
#Magnet2laser_offset = [0.0,0.0,-3.5]
Magnet2laser_offset = [0.125, 0.0625,-3.5]
#       height of laser about chess board plane (inches) when Z height of arm is 0.0
Laser_height_0 = 4.750
#
#   create an array for storing feature properties that will be used to determined identity of chess piece from image
#
Img_feat = np.reshape(np.zeros(20),[1,20])
Feat_log = Img_feat
Log_fname = 'testlog.txt'
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
Img_feat_header += '\tNorm_Kurtosis'
Img_feat_header += '\tNorm_Range'
Img_feat_header += '\tFootprint'
Img_feat_header += '\tRadius_Base'
Img_feat_header += '\tRadius_Crown'
Img_feat_header += '\tRadius_Waist'
Img_feat_header += '\treserved'
Img_feat_header += '\treserved'
Img_feat_header += '\treserved'
#
#   X0,Y0,Z0 are the current location of the 3 stages in absolute coordinates (units are in inches)
#
Loc_cur = [6.0,12.0,1.5]
Magnet_actual_loc = [Loc_cur[0],Loc_cur[1],Laser_height_0+Magnet2laser_offset[2]]
Laser_actual_loc = np.copy(Magnet_actual_loc)
Laser_actual_loc = np.add(Laser_actual_loc,np.multiply(-1.0,Magnet2laser_offset))
Laser_apparent_loc = np.copy(Laser_actual_loc)
Loc_pre = [0,0,0]
Loc_temp = np.copy(Loc_cur)
Col_inches = 1.0
Row_inches = 1.0
Zero_location_col = 0
Zero_location_row = 0
Laser_row = 0
Laser_col = 0

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
Piece_hts = [1.25,1.75,1.8125,1.375,2.125,(2.5-1.0/16.0)]     # ht of piece above plane of board - p,n,p,r,q,k
Arm_ht_offset = 0.125/10.5*0.0                      # arm motion is not planar, arm gets closer to board as x goes to zero
                                                # value is calibrated slope  (z ht. offset/(xloc))
Hover_ht = 2.75                                 # ht of magnet above plane of board to move pieces without any pieces in the way
Move_ht  = 4.25                                 # ht of magnet above plane of board when holding a knight to clear over a king
Capture_ht= 4.75                                # ht of magnet above plane of board when removing a captured piece to side of board

Slew_rate = [4.0/25.0,4.0/25.0,4.0/25.0]                  # slew rate of x,y,z motion in inches/second
Captured_piece_loc = [6.0,14.5]
Chess_board = np.zeros([8,8,2],dtype=int)
Color_code = 'x.WB'
Piece_code = 'x.PNBRQK'
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
#       aspect ratio of 4:3 with full FOV
#           ncols = 1920
#           nrows = 1440
#           col_scal = 1.778
Nrows = 1920
Ncols = 1440
Col_scal = 1.778
#
#   setup camera
#
camera = picamera.PiCamera()
camera.rotation = 0
camera.preview_fullscreen = 0
camera.crop = (0.0,0.0,1.0,1.0)
camera.preview_window =(0, 200, 768,1024)

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
    prompt = raw_input('Enter y to Enable 2 Step Motion : ')
    if prompt.lower() == 'y' :
        Two_step_motion = 1
    else :
        Two_step_motion = 0
except ValueError:    
    Two_step_motion = 0


try:
    prompt = raw_input('Enter y to Disable Correct Backlash with Laser feedback : ')
    if prompt.lower() == 'y' :
        Laser_backlash_correction = 0
    else :
        Laser_backlash_correction = 1
except ValueError:    
    Laser_backlash_correction = 1
    
try:
    prompt = raw_input('Enter y to run in Talking Mode : ')
    if prompt.lower() == 'y' :
        Talk_mode = 1
        print '*** running in Talking Mode ***'
    else :
        Talk_mode = 0
except ValueError:
    Talk_mode = 0    

try:
    prompt = raw_input('Enter y to run in Show Chess Engine Analysis during the game : ')
    if prompt.lower() == 'y' :
        Show_Stockfish_Analysis = 1
        print '*** Will show Stockfish analysis during chess game ***'
    else :
        Show_Stockfish_Analysis = 0
except ValueError:
    Show_Stockfish_Analysis = 0
        
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
cmdlst = cmdlst + "    sq_img   : analyze image of chess square & store features in log\n"
cmdlst = cmdlst + "    anal     : capture Image for manual board analysis\n"
cmdlst = cmdlst + "    grbl     : enter in GRBL command to CNC\n"
cmdlst = cmdlst + "    stock    : play chess against Stockfish chess engine\n"
cmdlst = cmdlst + "    q        : to quit program"

chess_cmds = "List of chess commands: \n"
chess_cmds = chess_cmds + "    mxxyy  : chess move xx is initial sq & yy is destentation sq\n"
chess_cmds = chess_cmds + "             use alg notation for sqs (e.g me2e4 moves pc on e2 to e4\n"
chess_cmds = chess_cmds + "    n      : start a new chess game\n"   
chess_cmds = chess_cmds + "    q      : finished chess game, return to main menu\n"

#   view video until quit command is entered for Arduino

time.sleep(2)
capture_image(0,'all','all_scale')

if Eng_mode :
    Eng_mode = 0
    if Plt_mode :
        Plt_mode = 0
        calibrate_board()
        calibrate_arm_location(1,0,1)
        calibrate_arm_location(0,1,1)
        Eng_mode = 1
        Plt_mode = 1
    else :
        calibrate_board()
        calibrate_arm_location(1,0,1)
        calibrate_arm_location(0,1,1)
        Eng_mode = 1    
else :
    calibrate_board()
    calibrate_arm_location(1,0,1)
    calibrate_arm_location(0,1,1)
 
print 'Current X,Y,Z location : {0:.3f}, {1:.3f}, {2:.3f}'.format(Loc_cur[0],Loc_cur[1],Loc_cur[2])
print 'Current Laser location : {0:.3f}, {1:.3f}, {2:.3f}'.format(Laser_actual_loc[0],Laser_actual_loc[1],Laser_actual_loc[2])
print 'Current Magnet location: {0:.3f}, {1:.3f}, {2:.3f}'.format(Magnet_actual_loc[0],Magnet_actual_loc[1],Magnet_actual_loc[2])        
    
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
        capture_image(0,'all','all_scale')
        camera.stop_preview()
        calibrate_board()
        camera.start_preview()
    elif (strin.lower() == "load_log") :
        Log_fname = raw_input('What is name of log file to load? ')
        Feat_log = np.loadtxt(Log_fname,delimiter='\t')
    elif (strin.lower() == "make_log") :
        Log_fname = raw_input('What is name of log file to create? ')
        np.savetxt(Log_fname, Feat_log, fmt='%1.4e',delimiter='\t',header= Img_feat_header)
    elif (strin.lower() == "sq_img") :
        camera.stop_preview()
        sq = raw_input('Which square to view (e.g. a6, e8) ? ')
        temp_plt_mode = Plt_mode
        Plt_mode = 1
            
        while len(sq) == 2 :
            capture_image(0,'all','all_scale')
            code = get_square_image(sq,1)
            Img_feat[0,0] = input('Color of chess piece (0:black, 1:white) = ')
            Img_feat[0,1] = input('Chess piece ID (1: empty, 2: pawn, 3: knight, 4: bishop, 5: rook, 6: queen, 7: king) = ')
            Feat_log = np.append(Feat_log,Img_feat,axis = 0)
            np.savetxt(Log_fname, Feat_log, fmt='%1.2e',delimiter='\t',header= Img_feat_header)
            sq = raw_input('Which square to view (e.g. a6, e8) ? ')
            
        camera.start_preview()
        Plt_mode = temp_plt_mode
    elif (strin.lower() == "ref_b") :
        camera.stop_preview()
        update_chess_board(1)
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
        if Eng_mode : camera.stop_preview()
        calibrate_arm_location(1,0,1)
        if Eng_mode : camera.start_preview()
        print 'Current X,Y,Z location : {0:.3f}, {1:.3f}, {2:.3f}'.format(Loc_cur[0],Loc_cur[1],Loc_cur[2])
        print 'Current Laser location : {0:.3f}, {1:.3f}, {2:.3f}'.format(Laser_actual_loc[0],Laser_actual_loc[1],Laser_actual_loc[2])
        print 'Current Magnet location: {0:.3f}, {1:.3f}, {2:.3f}'.format(Magnet_actual_loc[0],Magnet_actual_loc[1],Magnet_actual_loc[2])
    elif (strin.lower() == "loc") :
        if Talk_mode : talk('The current location of the magnet')
        print 'Current X,Y,Z location : {0:.3f}, {1:.3f}, {2:.3f}'.format(Loc_cur[0],Loc_cur[1],Loc_cur[2])
        print 'Current Laser location : {0:.3f}, {1:.3f}, {2:.3f}'.format(Laser_actual_loc[0],Laser_actual_loc[1],Laser_actual_loc[2])
        print 'Current Magnet location: {0:.3f}, {1:.3f}, {2:.3f}'.format(Magnet_actual_loc[0],Magnet_actual_loc[1],Magnet_actual_loc[2])        
    elif (strin.lower() == "grbl") :
        if Talk_mode : talk('Now in direct GRBL command mode')
        strin = raw_input('Type in GRBL command (e.g X5): ')
        ser.write('G91 G0 '+strin+'\r\n')
        reply = ser.readline()
        print 'GRBL Reply back: ',reply,
    elif (strin.lower() == "stock") :       # initiate chess playing program Stockfish

        update_chess_board(1)
        print ("\n Chess Program \n")
        print chess_cmds
        Skill = "10"
        Movetime = "6000"
        fmove = newgame()
        
        talk('New game started. Please make your first move and best of luck in the game.')
          
        while True:

    # Get  message from board
            B_message = getboard()
    # Message options   Move, Newgame, level, style
            code = B_message[0]
           
    # decide which function to call based on first letter of txt
            fmove=fmove
            if code == 'm':
                chess_move(Human_move,0)            # use CNC to make human move on the board if Human_manual <> 0
                if Eng_mode :
                    print_chess_board()
                cmove = bmove(fmove)                # chess engine move
                if cmove == 'error' :
                    print 'Invalid move.  Renter the chess move'
                else : 
                    chess_move(Computer_move,1)         # use CNC to make computer move on the board
                    if Eng_mode :
                        print_chess_board()
                    fmove = cmove
#                if Talk_mode : talk('It is your move. Move carefully!')
                    talk('I have made my move. It is your turn')
            elif code == 'n': newgame()
            elif code == '?': print chess_cmds
            elif code == 'q': break
            else :  sendboard('error at option')

        talk('Thanks for the game. Bye Bye')    
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
