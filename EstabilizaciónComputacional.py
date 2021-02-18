##Estabilizador de los fotogramas de un vídeo respecto a otro de referencia
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob

### 1-CONFIGURACIONES Y RECURSOS
frame_ref = 80
nfeatures=1500
distance=0.8
vidcap = cv2.VideoCapture('hibrid30_new2.mp4')
count = 0
good_points_i=[]

while count != frame_ref:
    success,img1 = vidcap.read()
    count += 1
last_img_good=img1

while success:
    success,img2 = vidcap.read()

    ### 2-EXTRACCIÓN DE CARACTERÍSTICAS
    orb = cv2.ORB_create(nfeatures)
    kp_orb1, desc1 = orb.detectAndCompute(img1, None)
    kp_orb2, desc2 = orb.detectAndCompute(img2, None)
    if success:
        desc1 = desc1.astype('float32')
        desc2 = desc2.astype('float32')
    
        ### 3-EMPAREJADO DE PUNTOS
        index_params = dict (algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher (index_params, search_params)
        
        #Dibujar los puntos en las imágenes
        #img1=cv2.drawKeypoints(img1, kp_orb1, img1)
        #img2=cv2.drawKeypoints(img2, kp_orb2, img2)
        
        ### 4-EMPAREJADO DE COINCIDENCIAS
        ## 4.1-Emparejado de puntos por distancia knn
        matches= flann.knnMatch(desc1, desc2, k=2)
        good_points = []
        ## 4.2-Filtrado de las mejores coincidencias
        for m, n in matches: 
            if m.distance <distance*n.distance:
                good_points.append(m)
    
    
        
        ### 5-CÁLCULO DE LA HOMOGRÁFICA Y SU INVERSA
        ## 5.1-Filtrado de puntos para el cálculo de la homográfica
        if len(good_points)>10:
            query_pts = np.float32([kp_orb1[m.queryIdx].pt for m in good_points]).reshape(-1,1,2)
            train_pts = np.float32([kp_orb2[m.trainIdx].pt for m in good_points]).reshape(-1,1,2)
          
        ## 5.2-Dibujar las coincidencias de puntos(correspondences)
        corr= cv2.drawMatches(img1, kp_orb1, img2, kp_orb2, good_points,img2)    
        ## 5.3-Calculo de la homográfica
        m, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask= mask.ravel().tolist()
        
        ## 5.4-Cálculo de la homográfica inversa
        n = np.linalg.inv(m)
               
        ## 5.5-Extracción del ángulo
        theta = - math.atan2(m[0,1], m[0,0]) * 180 / math.pi
        
        
        ### 6-APLICACIÓN DE LAS TRANSFORMACIONES
        ## 6.1-Cáculo de parámetros
        rows,cols,ch = img1.shape
        pts= np.float32([[0, 0], [0, rows], [cols, rows], [cols,0]]).reshape (-1,1,2)
        
        ## 6.2-Transformación de puntos
        dst= cv2.perspectiveTransform(pts, m)
        dst_inv= cv2.perspectiveTransform(pts, n)
        
        ## 6.3-Aplicar la transformada de perspectiva
        img3=cv2.warpPerspective(img1, m, (cols,rows))
        img4=cv2.warpPerspective(img2, n, (cols,rows))

        ## 6.4-Dibujar la homográfica con perspectiva sobre la imagen        
        img4_ln = cv2.polylines(img4, [np.int32(dst_inv)], True, (0, 255, 255), 3)
        img5 = cv2.addWeighted(img1,1,img4_ln,0.5,0)
        
        ### 7- IMÁGENES DE RESULTADOS
        ## 7.1-Generar las imágenes:
    #    cv2.imwrite("3-RelaciónDePuntos.png", corr)
    #    cv2.imwrite("4-ImagenFinalAplicandoHomInv.png", img4)
    #    cv2.imwrite("5-Combinación.png", img5)
  
        
        ## 7.2-Mostrar las imágenes en el terminal
    #    plt.subplot(231),plt.imshow(img1),plt.title('1-Input')
    #    plt.subplot(232),plt.imshow(img2),plt.title('2-Next Frame')
    #    plt.subplot(233),plt.imshow(corr),plt.title('Relación de puntos')
    #    plt.subplot(234),plt.imshow(img3),plt.title('3-Con homográfica')
    #    plt.subplot(235),plt.imshow(img4),plt.title('4-Con inversa')
    #    plt.subplot(236),plt.imshow(img5),plt.title('5-Superpuestas')
    #    plt.show()
    
        ## 7.3-Tratamiento de malas correspondencias
        #Si el cálculo de la homográfica es malo se coge la última mejor imagen
        if len(good_points)<10:
            print("***Fotogrma con pocos puntos")
            img5= last_img_good
        last_img_good=img5
        
        ## 7.4-Código para crear los fotogramas para luego formar un vídeo
        cv2.imwrite("frame%d.jpg" % count,img4)
        print('Fotograma %d leído: '% count, success)
        count += 1
        good_points_i.append(len(good_points))
        
###8-RECREACIÓN DEL VÍDEO
## 8.1-Código para crear un vídeo a partir de los fotogramas        
print('Creando el vídeo... ')        
img_array = []
for filename in glob.glob('*.jpg'):
    img = cv2.imread(filename)
    img_array.append(img)

height, width, layers = img4.shape
size = (width,height)
out = cv2.VideoWriter('hibrid30_new2_inv.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
print('Fin del programa ')


