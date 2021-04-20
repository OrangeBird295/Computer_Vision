import numpy as np
import cv2
import time
image_hsv = None   # global ;(
#pixel = (20,60,80) # อันนี้คือค่าที่เขาตั้งขึ้นมามั่วๆอะ

def capimg():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while(True):
        ret, frame = cap.read()
        color = cv2.cvtColor(frame, 0)

        cv2.imshow('capture img [when you ready you must pass S : save]',color)

        #key = cv2.waitKey(1)
        if cv2.waitKey(33) == ord('s'):
            cv2.imwrite("img.jpg", frame)
            break
    cap.release()
    cv2.destroyAllWindows()

def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #คลิกเม้าซ้าย
        #print("pick_color", x, y, flags, param)
        pixel_0rigin = image_hsv[y,x];     
        pixel_1 = image_hsv[y, x+1];        
        pixel_2 = image_hsv[y-1, x+1];     
        pixel_3 = image_hsv[y-1, x];       
        pixel_4 = image_hsv[y-1, x-1];    
        pixel_5 = image_hsv[y, x-1];       
        pixel_6 = image_hsv[y+1, x-1];     
        pixel_7 = image_hsv[y+1, x];      
        pixel_8 = image_hsv[y+1, x+1];    
        H = (int(pixel_0rigin[0])+int(pixel_1[0])+int(pixel_2[0])+int(pixel_3[0])+int(pixel_4[0])+int(pixel_5[0])+int(pixel_6[0])+int(pixel_7[0])+int(pixel_8[0]))/9
        S = (int(pixel_0rigin[1])+int(pixel_1[1])+int(pixel_2[1])+int(pixel_3[1])+int(pixel_4[1])+int(pixel_5[1])+int(pixel_6[1])+int(pixel_7[1])+int(pixel_8[1]))/9
        V = (int(pixel_0rigin[2])+int(pixel_1[2])+int(pixel_2[2])+int(pixel_3[2])+int(pixel_4[2])+int(pixel_5[2])+int(pixel_6[2])+int(pixel_7[2])+int(pixel_8[2]))/9
        print("H->", H, "S->", S, "V->", V)
        upper =  np.array([H+10, S+10, V+20])
        lower =  np.array([H-10, S-10, V-20])
        thearray = [upper, lower] 
        print(" loxer ->", lower, '\n', "upper ->", upper, '\n')
        np.save('penval',thearray)
        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow("MASK img  [when you ready you must pass S or ESC : save]",image_mask)

capimg()

image_src = cv2.imread("img.jpg") 
if image_src is None:
    print ("the image read is None............")
cv2.imshow("BGR img",image_src)

## NEW ##
cv2.namedWindow('HSV img') #สร้างหร้าต่างใหม่โดยใช้ชื่อ hsv (ชื่อ, ขนาดของหน้าต่าง)
cv2.setMouseCallback('HSV img', pick_color)

# now click into the hsv img , and look at values:
image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
print(image_hsv)
cv2.imshow("HSV img",image_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()

def Draw():
    load_from_disk = True
    if load_from_disk:
        penval = np.load('penval.npy')

    cap = cv2.VideoCapture(0)

    # Load these 2 images and resize them to the same size.
    pen_img = cv2.resize(cv2.imread('pen.png',1), (50, 50))
    eraser_img = cv2.resize(cv2.imread('eraser.png',1), (50, 50))

    blue_img = cv2.resize(cv2.imread('blue.png',1), (50, 50))
    green_img = cv2.resize(cv2.imread('green.png',1), (50, 50))
    red_img = cv2.resize(cv2.imread('red.png',1), (50, 50))
    blue = [255,0,0]
    green = [0,255,0]
    red = [0,0,255]
    pen_color = blue

    kernel = np.ones((5,5),np.uint8)

    # Making window size adjustable
    cv2.namedWindow('image [when you want to exit you must pass ESC : exit]', cv2.WINDOW_NORMAL)

    # This is the canvas on which we will draw upon
    canvas = None

    # Create a background subtractor Object  เป็นการลบพื้นหลัง เงา=false
    backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

    # This threshold determines the amount of disruption in the background.
    background_threshold = 800

    # A variable which tells you if you're using a pen or an eraser.
    switch = 'Pen'

    # With this variable we will monitor the time between previous switch.
    last_switch = time.time()

    # Initilize x1,y1 points
    x1,y1=0,0

    # Threshold for noise
    noiseth = 800

    # Threshold for wiper, the size of the contour must be bigger than this for # us to clear the canvas
    wiper_thresh = 40000

    # A variable which tells when to clear canvas
    clear = False

    while(1):
        _, frame = cap.read()
        frame = cv2.flip( frame, 1 )
        
        # Initilize the canvas as a black image
        if canvas is None:
            canvas = np.zeros_like(frame)
            
        # Take the top left of the frame and apply the background subtractor
        # there    กรอบรูปยางลบและปากกา
        top_left = frame[0: 50, 0: 50]
        fgmask = backgroundobject.apply(top_left)
        
        blue_func =  frame[70: 120, 0: 50] 
        green_func = frame[140: 190, 0: 50] 
        red_func = frame[210: 260, 0: 50]
        bluemask = backgroundobject.apply(blue_func)
        greenmask = backgroundobject.apply(green_func)
        redmask = backgroundobject.apply(red_func)

        #np.sum ซึ่งเป็นฟังก์ชันที่จะรวมผลบวกของ array 
        switch_blue = np.sum(bluemask==255)
        switch_green = np.sum(greenmask==255)
        switch_red = np.sum(redmask==255)
    
        if switch_blue>background_threshold :
            if pen_color == green or pen_color == red:
                pen_color = blue
        if switch_green>background_threshold :
            if pen_color == blue or pen_color == red:
                pen_color = green
        if switch_red>background_threshold :
            if pen_color == blue or pen_color == green:
                pen_color = red

        # Note the number of pixels that are white, this is the level of 
        # disruption.
        switch_thresh = np.sum(fgmask==255)
        
        # If the disruption is greater than background threshold and there has 
        # been some time after the previous switch then you. can change the 
        # object type.
        if switch_thresh>background_threshold and (time.time()-last_switch) > 1:

            # Save the time of the switch.  # delay 1 Second.
            last_switch = time.time()
            #print(" last_switch = "+ last_switch)
            if switch == 'Pen':
                switch = 'Eraser'
            else:
                switch = 'Pen'

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # If you're reading from memory then load the upper and lower ranges 
        # from there
        if load_from_disk:
            lower_range = penval[1]
            upper_range = penval[0]
           
                
        # Otherwise define your own custom values for upper and lower range.
        else:             
            lower_range  = np.array([15,206,161])
            upper_range = np.array([35,226,241])
        
        # print("Low", lower_range)
        # print("Upp", upper_range)
        mask = cv2.inRange(hsv, lower_range, upper_range)
        #print(lower_range, upper_range)

        # Perform morphological operations to get rid of the noise
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask = cv2.dilate(mask,kernel,iterations = 2)
        
        # Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)
        
        # Make sure there is a contour present and also it size is bigger than 
        # noise threshold.
        if contours and cv2.contourArea(max(contours,
                                        key = cv2.contourArea)) > noiseth:
                    
            c = max(contours, key = cv2.contourArea)    
            x2,y2,w,h = cv2.boundingRect(c)
            
            # Get the area of the contour
            area = cv2.contourArea(c)
            
            # If there were no previous points then save the detected x2,y2 
            # coordinates as x1,y1. 
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2

            #cv2.circle(img, center, radius, color[, thicknes])    
            else:  
                if switch == 'Pen':
                    # Draw the line on the canvas
                    canvas = cv2.line(canvas, (x1,y1),
                    (x2,y2), pen_color, 5)
                    
                else:
                    cv2.circle(canvas, (x2, y2), 20,
                    (0,0,0), -1)
                
                
            
            # After the line is drawn the new points become the previous points.
            x1,y1= x2,y2
            
            # Now if the area is greater than the wiper threshold then set the 
            # clear variable to True
            if area > wiper_thresh:
                cv2.putText(canvas,'Clearing Canvas',(0,200), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 1, cv2.LINE_AA)
                clear = True 

        else:
            # If there were no contours detected then make x1,y1 = 0
            x1,y1 =0,0
        
    
        # Now this piece of code is just for smooth drawing. (Optional)

        #cv2.bitwise_and(frame,frame,mask=fgmask) คือการนำภาพระดับบิตที่ชื่อว่า frame มาทำการ AND กับ Mask ที่ชื่อว่า fgmask โดยมีหลักการว่า 0 AND กับอะไรก็จะได้ 0

        #cv2.bitwise_not(fgmask) ใช้ในการสลับภาพ fgmask จาก 0 เป็น 1 จาก 1 เป็น 0 หรือ การสลับจากภาพขาวเป็นดำ จากดำเป็นขาว

        #cv2.add(inv2,res)เป็นฟังก์ชันที่ใช้ในการรวมภาพที่ชื่อว่า inv2 และ res โดยการ add เป็นการกระทำการของnumpy 
        #เป็นการบวกกันของบิตของภาพทั้งสอง โดยค่าสีในแต่ละpixelจะมีค่า0-255  ถ้าค่าที่บวกได้ต่ำกว่า 0 ให้เป็น 0 และ ถ้าค่าสูงกว่า 255 
        # ให้เป็น 255
        _ , mask = cv2.threshold(cv2.cvtColor (canvas, cv2.COLOR_BGR2GRAY), 20, 
        255, cv2.THRESH_BINARY)
        foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
        background = cv2.bitwise_and(frame, frame,
        mask = cv2.bitwise_not(mask))
        frame = cv2.add(foreground,background)

        # Switch the images depending upon what we're using, pen or eraser.
        if switch != 'Pen':
            cv2.circle(frame, (x1, y1), 20, (255,255,255), -1)
            frame[0: 50, 0: 50] = eraser_img
        else:
            cv2.circle(frame, (x1, y1), 8, pen_color, -1)
            frame[0: 50, 0: 50] = pen_img

        
        frame[70: 120, 0: 50] = blue_img
        frame[140: 190, 0: 50] = green_img
        frame[210: 260, 0: 50] = red_img

        stacked = np.hstack((canvas,frame))
        cv2.imshow('image [when you want to exit you must pass ESC : exit]',cv2.resize(stacked,None,fx=1.6,fy=1.6))

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        
        # Clear the canvas after 1 second, if the clear variable is true
        if clear == True: 
            time.sleep(1)
            canvas = None
            
            # And then set clear to false
            clear = False
            
    cv2.destroyAllWindows()
    cap.release()

Draw()

