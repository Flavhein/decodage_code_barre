import numpy as np

# Otsu's algorithm for binarization
def otsu(signal):
    # Step 1: Calculate the histogram
    hist = np.histogram(signal, bins=256, range=(0, 256))[0]
    hist = hist / len(signal)  # Normalize to get probabilities

    # Step 2: Cumulative probabilities and cumulative means
    prob_cumul = np.cumsum(hist)  # Calculate cumulative probabilities
    mean_cumul = np.cumsum(np.arange(256) * hist)  # Calculate cumulative means

    # Global mean
    mean_global = mean_cumul[-1] # moyenne globale

    # Step 3: Calculate inter-class variance for each threshold
    inter_class_variance = np.zeros(256)
    for t in range(256):
        if prob_cumul[t] == 0 or prob_cumul[t] == 1:
            inter_class_variance[t] = 0
            continue
        weight_bg = prob_cumul[t]  # Background weight
        weight_fg = 1 - weight_bg  # Foreground weight

        mean_bg = mean_cumul[t] / weight_bg  # Background mean
        mean_fg = (mean_global - mean_cumul[t]) / weight_fg  # Foreground mean

        inter_class_variance[t] = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

    # Find the threshold that maximizes the inter-class variance
    threshold = np.argmax(inter_class_variance)
    return threshold

def extract_code_barre(point1, point2, img):
    x1, y1 = point1
    x2, y2 = point2
    
    # Calculate the distance between the two points
    distance = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2)) # un entier car on le mettra dans une boucle
    
    # Create a set of points between the two points
    x_points = [x1 + (x2 - x1) / (distance - 1) * i for i in range(distance)]
    y_points = [y1 + (y2 - y1) / (distance - 1) * i for i in range(distance)]
    
    # Extract the signal V
    V = []
    for x, y in zip(x_points, y_points):
        if 0 <= int(round(y)) < img.shape[0] and 0 <= int(round(x)) < img.shape[1]: # on regarde si les coordonnées sont dans l'image du code barre
            V.append(img[int(round(y)), int(round(x))]) # si oui, on le rajoute à V (la valeurs du pixel, compris entre 0 et 255)
    
    print(V)
    # Apply Otsu's algorithm to get a threshold
    threshold = otsu(V)
    
    # Binarize the signal
    binary_signal = (V > threshold).astype(int) # si la valeur du pixel est supérieur au seuil, on met 1, sinon 0
    
    # Identify the limits (guard regions)
    start = np.argmax(binary_signal)  # First black region
    end = len(binary_signal) - np.argmax(binary_signal[::-1]) - 1  # Last black region
    
    # Extract the useful portion (barcode without unnecessary parts)
    useful_signal = binary_signal[start:end+1] # enleve les parties blanches inutiles
    
    return useful_signal, threshold
