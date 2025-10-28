import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from scipy.ndimage import convolve, label
from skimage.transform import resize

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
    
    #X,Y=np.meshgrid(range(-taille,taille+1),range(-taille,taille+1))
    #Gauss=1/(2*np.pi*sigma_T**2)*np.exp(-(X**2 + Y**2)/(2*sigma_T**2))
    Y=np.linspace(-taille,taille,taille*2);
    gauss_=np.array(1/(2*np.pi*sigma_T**2)*np.exp(-(Y**2)/(2*sigma_T**2)))
    
    T_xx=convolve(convolve(I_x**2,gauss_.reshape(-1,1)),gauss_.reshape(1, -1))
    T_yy=convolve(convolve(I_y**2,gauss_.reshape(-1,1)),gauss_.reshape(1, -1))
    T_xy=convolve(convolve(I_x*I_y,gauss_.reshape(-1,1)),gauss_.reshape(1, -1))

    return T_xx,T_yy,T_xy


def compute_coherence(T_xx, T_yy, T_xy):
    numerator = (T_xx - T_yy)**2 + 4 * (T_xy)**2
    denominator = T_xx + T_yy
    D = np.sqrt(numerator) / denominator
    return D


def acp(img):
    
    x,y=np.where(img==1)
    coord=list(zip(x,y))
    points_array = np.array(coord)
    print("Points array :",points_array)
    mean = np.mean(points_array, axis=0) #centre
    points_centered = points_array - mean
    
    covariance_matrix = np.cov(points_centered.T)
    
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix) #valeurs propres
    
    max_index = np.argmax(eigenvalues)  
    principal_component = eigenvectors[:, max_index]
    
    #projected_points = np.dot(points_centered, principal_component) #projection
    
    return mean,principal_component,eigenvalues, eigenvectors 
    
    

##########################################
## MAIN
##########################################

def position(image_path):
    plt.close('all')
    image_source=plt.imread(image_path)
    image_origin = image_source
    shape_origin = image_source.shape
    height, width, channels = shape_origin

    #paramètres

    new_height = 1080  # On définit la hauteur de la nouvelle image à partir de laquelle on cherche le code barre
    sig=1.34
    sigma_G=sig
    sigma_T=sig*15
    noise=1
    seuil=0.58
    dilatation=20
    nb_point_random=10
    dispersion= np.floor(height/50)

    # Calculer le facteur de réduction

    aspect_ratio = width / height

    # Définir la nouvelle hauteur 

    new_width = int(new_height * aspect_ratio)

    # Redimensionner l'image en gardant l'aspect ratio
    image_source = resize(image_source, (new_height, new_width), anti_aliasing=True)

    # Affichage
    plt.imshow(image_source)
    
    # Convertir en YCbCr
    if image_source.shape[2] == 4:
        image_source = image_source[:, :, :3]
    image_ycbr=ski.color.rgb2ycbcr(image_source)
    image=image_ycbr[:,:,0]  

    image=image+noise*np.random.normal(0,1,image.shape) #ajout du bruit

    grad_x, grad_y = compute_gradient(image,sigma_G)
    T_xx,T_yy,T_xy = compute_structure_tensor(grad_x, grad_y, sigma_T,sigma_T*3)
    coherence = compute_coherence(T_xx,T_yy,T_xy)
        
    #plt.figure()
    #plt.imshow(coherence,cmap='gray')
    #plt.title("Coherence")
        

    coh_seuil=np.ones_like(coherence)
    coh_seuil[(coherence > seuil) & (coherence <= 1)] = 0
    coh_seuil = 1-coh_seuil

    #Labeliser pour obtenir la zone contenant le code-barre  
    labeled_array,num_label = label(coh_seuil)
    unique, counts = np.unique(labeled_array, return_counts=True)
        
    counts[0] = -1 #pour ne pas sélectionner le fond noir qui possède le plus de pixel
    
    # La valeur du fond est changé, la valeur maximale est maintenant celle du code barre
    indice_code = np.unravel_index(np.argmax(counts), counts.shape)
    #Masque que sur indice code : là où il y a le code barre
        
    labeled_array[labeled_array != indice_code] = 0
    labeled_array[labeled_array == indice_code] = 1
    plt.figure()
    plt.imshow(labeled_array, cmap='gray')
    plt.title("Cohérence avec uniquement code barre")

    # dilatation
    h_=np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]])*(1/5)    

    for i in range(dilatation):
        labeled_array=dila(labeled_array,h_)

    # Trouver valeur centre du code-barre et son orientation    
    centre, p1, valeur_avance, eigenvectors  = acp(labeled_array)
    #print(p1)
    #print(valeur_avance, eigenvectors)
        

    #plt.figure()
    #plt.imshow(labeled_array, cmap='gray')
    #plt.title("Cohérence avec uniquement code barre")
        
    #plt.figure()
    #plt.imshow(image_source, cmap='gray')
    #plt.title("Image originale")

    # Recherche des bords du code-barre
    centre = np.round(np.array([centre[1],centre[0]])).astype('uint64')
    valeur_avance = np.round(valeur_avance/min(valeur_avance)).astype('uint64')
    valeur_avance = (valeur_avance[1],valeur_avance[0])
    bord = np.array([centre,centre])
    #plt.scatter([centre[0]], [centre[1]], color='green', label='Point 3', zorder=1)
    while ((labeled_array[bord[0][1].astype('uint64')][bord[0][0].astype('uint64')]!=0) & (labeled_array[bord[1][1].astype('uint64')][bord[1][0].astype('uint64')]!=0)):
        bord[0] = bord[0] + valeur_avance
        bord[1] = bord[1] - valeur_avance


    #plt.scatter([bord[0][0]], [bord[0][1]], color='blue', label='Point 1', zorder=1)
    #plt.scatter([bord[1][0]], [bord[1][1]], color='red', label='Point 2', zorder=1)
    #plt.plot([bord[1][0], bord[0][0]],[bord[1][1],bord[0][1]])

    # On veut se ramener à l'image de base et donc au bon format
    ratio_height = height/new_height
    ratio_width = width/new_width
    bord_origin1, bord_origin2 = [[bord[0][0]*ratio_height,bord[0][1]*ratio_width], [bord[1][0]*ratio_height,bord[1][1]*ratio_width]]

    # Image originale

    plt.figure()
    plt.imshow(image_origin, cmap='gray')
    plt.title("Image originale")
    plt.scatter([bord_origin1[0]], [bord_origin1[1]], color='blue', label='Point 1', zorder=1)
    plt.scatter([bord_origin2[0]], [bord_origin2[1]], color='red', label='Point 2', zorder=1)
    plt.plot([bord_origin1[0], bord_origin1[0]],[bord_origin2[1],bord_origin2[1]])


    # Chercher plusieurs points autour des bords trouvés
    matrice_point1=np.zeros((nb_point_random,2))
    matrice_point2=np.zeros((nb_point_random,2))

    for j in range(nb_point_random):
        matrice_point1[j][0]=bord_origin1[0] +np.random.randn()*dispersion #x
        matrice_point1[j][1]=bord_origin1[1] +np.random.randn()*dispersion #y
        
        matrice_point2[j][0]=bord_origin2[0] +np.random.randn()*dispersion #x
        matrice_point2[j][1]=bord_origin2[1] +np.random.randn()*dispersion #y
        
        plt.plot([matrice_point1[j][0], matrice_point2[j][0]],[matrice_point1[j][1],matrice_point2[j][1]])
    
    # On renvoit tous les points trouvés
    bord_origin1 = np.array([bord_origin1])
    bord_origin2 = np.array([bord_origin2])
    bords1 = np.concatenate([bord_origin1,matrice_point1])
    bords2 = np.concatenate([bord_origin2,matrice_point2])
    
    bords1 = np.round(bords1)
    bords2 = np.round(bords2)
    
    return (bords1, bords2)




    


