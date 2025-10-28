import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from scipy.ndimage import convolve, label
from scipy.ndimage import binary_dilation

##########################################
## FONCTIONS
##########################################

def eros(img,h):
    img_cv = convolve(img,h)
    height,width = img.shape
    img_eros = np.zeros(img.shape)
    for i in range(height):
        for j in range(width):
            if img_cv[i,j]==1:
                img_eros[i,j] = 1
            else :
                img_eros[i,j] = 0
    return img_eros

def dila(img,h):
    img_cv = convolve(img,h)
    height,width = img.shape
    img_dila = np.zeros(img.shape)
    for i in range(height):
        for j in range(width):
            if img_cv[i,j]>0:
                img_dila[i,j] = 1
            else :
                img_dila[i,j] = 0
    return img_dila

def compute_gradient(image,sig):
    X,Y=np.meshgrid(range(int(np.ceil(-3*sig)),int(np.ceil(3*sig))+1 ),range(int(np.ceil(-3*sig)),int(np.ceil(3*sig))+1 ))
    dGx = -X/(2*np.pi*sig**4)*np.exp(-(X**2 + Y**2)/(2*sig**2))
    #Gy = -Y/(2*np.pi*sig**4)*np.exp(-(Y**2+X**2)/(2*sig**2))

    grad_x=convolve(image,dGx)
    grad_y=convolve(image,np.transpose(dGx))
    return grad_x, grad_y


def compute_structure_tensor(grad_x, grad_y, sigma_T,temp):
    I_x = grad_x / (np.sqrt(grad_x**2 + grad_y**2))  #on normalise
    I_y = grad_y / (np.sqrt(grad_x**2 + grad_y**2))
    
    taille=int(np.ceil(temp));
    
    X,Y=np.meshgrid(range(-taille,taille+1),range(-taille,taille+1))
    Gauss=1/(2*np.pi*sigma_T**2)*np.exp(-(X**2 + Y**2)/(2*sigma_T**2))
    T_xx=convolve(I_x**2,Gauss)
    T_yy=convolve(I_y**2,Gauss)
    T_xy=convolve(I_x*I_y,Gauss)

    return T_xx,T_yy,T_xy

def max_distance(points):
    n = len(points)
    max_dist = 0
    point1, point2 = None, None
    
    for i in range(n):
        for j in range(i + 1, n):  # On évite les doublons
            dist = np.linalg.norm(points[i] - points[j])  # Distance Euclidienne
            if dist > max_dist:
                max_dist = dist
                point1, point2 = points[i], points[j]
    
    return max_dist, point1, point2


def contours(img,h):
    erosion = eros(img,h)  # Érosion de la matrice
    contours = img - erosion
    points = np.argwhere(contours == 1)

    # Calculer toutes les distances entre les points
    max_dist,point1,point2 = max_distance(points) #Utiliser max_dist pour vérifier que c'est le bon nombre de points ?
    return point1,point2   
    

def compute_coherence(T_xx, T_yy, T_xy):
    numerator = (T_xx - T_yy)**2 + 4 * (T_xy)**2
    denominator = T_xx + T_yy
    D = np.sqrt(numerator) / denominator
    return D

def distance_euclidienne(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def extraire_contours(img,h):
    img_dilat=dila(img,h)
    img_erod=eros(img, h)
    
    contours = img_dilat - img_erod
    return contours

def pt_eloigne(img,h):
    contour_pixels = extraire_contours(img,h)
    plt.figure()
    plt.imshow(contour_pixels)
    y,x=np.where(contour_pixels==1)
    coord=list(zip(x,y))
    
    max_distance = 0
    points_eloignes = (None, None)
    
    # Comparer chaque paire de points pour calculer la distance euclidienne
    for i in range(len(coord)):
        for j in range(i + 1, len(coord)):
            dist = distance_euclidienne(coord[i], coord[j])
            if dist > max_distance:
                max_distance = dist
                points_eloignes = (coord[i], coord[j])
    
    return points_eloignes
    

##########################################
## MAIN
##########################################

def main(image_path, sigma_G, sigma_T):
    plt.close('all')
    image_source=plt.imread(image_path)
    if image_source.shape[2] == 4:
        image_source = image_source[:, :, :3]
    image_ycbr=ski.color.rgb2ycbcr(image_source)
    image=image_ycbr[:,:,0]
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title("Image originale")
    
    noise=0.8
    image=image+noise*np.random.normal(0,1,image.shape) #ajout du bruit

    grad_x, grad_y = compute_gradient(image,sigma_G)
    T_xx,T_yy,T_xy = compute_structure_tensor(grad_x, grad_y, sigma_T,sigma_T*3)
    coherence = compute_coherence(T_xx,T_yy,T_xy)
    print(np.max(grad_x))
    print(np.max(grad_y))
    plt.figure()
    plt.imshow(coherence,cmap='gray')
    plt.title("Cohérence")
    
    seuil=0.58
    coh_seuil=np.ones_like(coherence)
    coh_seuil[(coherence > seuil) & (coherence <= 1)] = 0
    coh_seuil = 1-coh_seuil
    #Utiliser label pour garder seulement la zone du code barre
    labeled_array,num_label = label(coh_seuil)
    print(labeled_array)
    print(num_label)
    unique, counts = np.unique(labeled_array, return_counts=True)
    print(unique)
    print(counts)
    
    
    valeur_code = np.max(counts[1:len(counts)]) 
    counts[0] = -1
    # La valeur du fond est changé, la valeur maximale est maintenant celle du code barre
    indice_code = np.unravel_index(np.argmax(counts), counts.shape)
    print(indice_code)
    #Masque que sur indice code : là où il y a le code barre
    
    labeled_array[labeled_array != indice_code] = 0
    #On s'assure de bien rester entre 0 et 1 malgré le label
    labeled_array[labeled_array == indice_code] = 1
    
    plt.figure()
    plt.imshow(labeled_array, cmap='gray')
    plt.title("Cohérence avec uniquement code barre")
    
    # dilatation
    h_=np.array([[0, 1, 0],
                 [1, 1, 1],
                 [0, 1, 0]
                 ])*(1/5)
    
    for i in range(20):
        labeled_array=dila(labeled_array,h_)
    
    pt_loin=pt_eloigne(labeled_array,h_)
    print(pt_loin)
    
    # Plot de l'image de base
    plt.imshow(labeled_array, cmap='gray', origin='upper')  # Affiche l'image en fond
    plt.plot([pt_loin[0][0], pt_loin[1][0]], [pt_loin[0][1], pt_loin[1][1]], color='blue', label='Distance max')  # Ligne entre les deux points

    # Marquer les deux points
    plt.scatter([pt_loin[0][0]], [pt_loin[0][1]], color='green', label='Point 1', zorder=1)
    plt.scatter([pt_loin[1][0]], [pt_loin[1][1]], color='orange', label='Point 2', zorder=1)

    # Légendes et affichage
    plt.legend()
    plt.title("Distance maximale sur le contour")
    plt.show()


    h=np.array([[0,1,0],[1,1,1],[0,1,0]])*(1/5)
    
    p1,p2 = contours(labeled_array,h)
    
    # Plot de l'image de base
    plt.imshow(labeled_array, cmap='gray', origin='upper')  # Affiche l'image en fond
    plt.plot([p1[1], p2[1]], [p1[0], p2[0]], color='blue', label='Distance max')  # Ligne entre les deux points

    # Marquer les deux points
    plt.scatter([p1[1]], [p1[0]], color='green', label='Point 1', zorder=1)
    plt.scatter([p2[1]], [p2[0]], color='orange', label='Point 2', zorder=1)

    # Légendes et affichage
    plt.legend()
    plt.title("Distance maximale sur le contour")
    plt.show()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Traitement d\'image et mesure de cohérence.')
#     parser.add_argument('image', type=str, help='Chemin de l\'image à traiter')
#     args = parser.parse_args()

#     main(args.image, sigma_G=1.5, sigma_T=2.0)
    
sig=0.5
main("image_doc_code_barre.png",sigma_G=sig, sigma_T=sig*30)

