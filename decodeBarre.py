import click_seg as click
import extract_code_barre as cb

import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import cv2

plt.close()

#### FAMILLE ###
A = np.array(
    [
        [1, 1, 1, 0, 0, 1, 0],  # 0
        [1, 1, 0, 0, 1, 1, 0],  # 1
        [1, 1, 0, 1, 1, 0, 0],  # 2
        [1, 0, 0, 0, 0, 1, 0],  # 3
        [1, 0, 1, 1, 1, 0, 0],  # 4
        [1, 0, 0, 1, 1, 1, 0],  # 5
        [1, 0, 1, 0, 0, 0, 0],  # 6
        [1, 0, 0, 0, 1, 0, 0],  # 7
        [1, 0, 0, 1, 0, 0, 0],  # 8
        [1, 1, 1, 0, 1, 0, 0],  # 9
    ]
)

EAN_TABLE = {
    'A': {
        '0001101': 0, '0011001': 1, '0010011': 2, '0111101': 3,
        '0100011': 4, '0110001': 5, '0101111': 6, '0111011': 7,
        '0110111': 8, '0001011': 9
    },
    'B': {
        '0100111': 0, '0110011': 1, '0011011': 2, '0100001': 3,
        '0011101': 4, '0111001': 5, '0000101': 6, '0010001': 7,
        '0001001': 8, '0010111': 9
    },
    'C': {
        '1110010': 0, '1100110': 1, '1101100': 2, '1000010': 3,
        '1011100': 4, '1001110': 5, '1010000': 6, '1000100': 7,
        '1001000': 8, '1110100': 9
    }
}

def bresenham_line(x1, y1, x2, y2):
    """
    Implémente l'algorithme de Bresenham pour tracer une droite discrète entre deux points.

    :param x1: int, coordonnée x du premier point
    :param y1: int, coordonnée y du premier point
    :param x2: int, coordonnée x du deuxième point
    :param y2: int, coordonnée y du deuxième point
    :return: list of tuple, les coordonnées de la ligne
    """
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return points

def select_values_between_points(matrix, x1, y1, x2, y2):
    """
    Sélectionne les valeurs dans une matrice se trouvant entre deux points (x1, y1) et (x2, y2).

    :param matrix: np.ndarray, la matrice à traiter
    :param x1: int, coordonnée x du premier point
    :param y1: int, coordonnée y du premier point
    :param x2: int, coordonnée x du deuxième point
    :param y2: int, coordonnée y du deuxième point
    :return: np.ndarray, un vecteur contenant les valeurs entre les deux points
    """
    # Obtenir les points entre (x1, y1) et (x2, y2) avec l'algorithme de Bresenham
    line_points = bresenham_line(x1, y1, x2, y2)

    # Extraire les valeurs correspondantes de la matrice
    values_vector = [matrix[y, x] for x, y in line_points]

    return np.array(values_vector)

def determine_first_digit(decoded_left):
    # Define the parity patterns for the first digit
    PARITY_PATTERNS = {
        'AAAAAA': 0,
        'AABABB': 1,
        'AABBBA': 2,
        'AABBAB': 3,
        'ABAABB': 4,
        'ABBAAB': 5,
        'ABBBAA': 6,
        'ABABAB': 7,
        'ABABBA': 8,
        'ABBABA': 9
    }

    # Identify the parity (L/G) of each left digit
    parity_sequence = ''

    
    for block in decoded_left:
        parity_sequence += block[1]
    
    # Match the parity sequence to determine the first digit
    if parity_sequence in PARITY_PATTERNS:
        return PARITY_PATTERNS[parity_sequence]
    else:
        print(f"Unknown parity sequence: {parity_sequence}")

# Function to decode left and right blocks
def decode_block(block):
    binary_str = ''.join(map(str, block.astype(int)))  # Convert block to binary string
    if binary_str in EAN_TABLE['A']:
        return EAN_TABLE['A'][binary_str], 'A'
    elif binary_str in EAN_TABLE['C']:
        return EAN_TABLE['C'][binary_str], 'C'
    elif binary_str in EAN_TABLE['B']:
        return EAN_TABLE['B'][binary_str], 'B'
    else :
        print(f"Block does not match L nor G nor R encoding: {binary_str}")
        return -1, 'Z'
        

def mean_nb(Seq, len_car):
    return sum(Seq)/len_car

def my_reshape(Seq, new_len, len_car,seuil):
    new_seq = np.zeros(new_len)
    for i in range(len(Seq)//len_car) :
        new_seq[i] = int(mean_nb(Seq[i*len_car:(i+1)*len_car], len_car)>=seuil/255)
    return new_seq
        
# Main decoding logic
def decode_ean13(Seq_red, seuil):
    num_codes = len(Seq_red) // 95
    codes = my_reshape(Seq_red,95,num_codes,seuil)
    #print("Codes : ", codes)
    codes = codes.astype(int)
    
    # Extract the parts
    left_guard = codes[0:3]; #print("left_guard : ", left_guard)
    left_part = codes[3:45]; #print("left_part : ", left_part)
    middle_guard = codes[45:50]; #print("middle_guard : ", middle_guard)
    right_part = codes[50:92]; #print("right_part : ", right_part)
    right_guard = codes[92:]; #print("right_guard : ", right_guard)

    # Validate guard bars
    if not (np.array_equal(left_guard, [1, 0, 1]) and
            np.array_equal(middle_guard, [0, 1, 0, 1, 0]) and
            np.array_equal(right_guard, [1, 0, 1])):
        print("Invalid guard bars")
        #raise ValueError("Invalid guard bars")
        #return -1

    # Decode left and right parts
    left_digits = [left_part[i:i+7] for i in range(0, len(left_part), 7)]
    right_digits = [right_part[i:i+7] for i in range(0, len(right_part), 7)]

    # Identify parities and decode
    try:
        decoded_left = [decode_block(block) for i, block in enumerate(left_digits)]
        decoded_right = [decode_block(block) for block in right_digits]
        
        # Determine first digit based on parity
        first_digit = determine_first_digit(decoded_left)
        
        # Combine digits
        full_code = [first_digit]
        for block in decoded_left:
            full_code += [block[0]]
        for block in decoded_right:
            full_code += [block[0]]
    except ValueError as e:
        print(f"Decoding error: {e}")
        #return -1

    return full_code

def validate_checksum(code):
    if (-1 in code):
        return False
    odd_sum = sum(code[i] for i in range(0, 12, 2))
    even_sum = sum(code[i] for i in range(1, 12, 2))
    total = odd_sum + 3 * even_sum
    return (10 - (total % 10)) % 10 == code[-1]

def click_point(filename):
    # Load and display the image
    img = plt.imread(filename)
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    # Initialize variables
    line = ax.plot(
        [],
        [],
        c="blue",
        linestyle="-",
        linewidth=2,
        label="Line",
    )
    MAX_CLICKS = 2
    
    # Connect the click event
    linedrawer = click.LineDrawer(line)
    # Display the image and start the GUI loop
    print("Please click on the image twice...")
    
    # Keep the GUI open and monitor clicks
    while len(linedrawer.xs) < MAX_CLICKS:
        plt.pause(0.1)  # Process GUI events in short intervals
    
    fig.canvas.mpl_disconnect(linedrawer.cid)
    
    # After clicks are registered, the program will continue here
    print(f"Captured pixel coordinates: xs = {linedrawer.xs} and ys = {linedrawer.ys}")
    
    # Show the final image with the added cross and line
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    p1 = [int(linedrawer.xs[0]), int(linedrawer.ys[0])]
    p2 = [int(linedrawer.xs[1]), int(linedrawer.ys[1])]
    
    return p1, p2
    
def decodage_complet(filename, p1, p2):
    seg = int(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))
    
    L = np.array(range(seg))
    X = p1[0] + np.round((p2[0] - p1[0]) * L / (seg - 1)).astype(int)
    Y = round((p1[1] + p2[1]) / 2)
    
    img_calc = plt.imread(filename)
    img_ycbcr = ski.color.rgb2ycbcr(img_calc[:, :, 0:3])
    img_Y = img_ycbcr[:, :, 0]
    
    Seq_cbarre = []
    
    level_y = p2[1] - p1[1] if p2[1] - p1[1] > 0 else p1[1] - p2[1] 
    level_x = p2[0] - p1[0] if p2[0] - p1[0] > 0 else p1[0] - p2[0]
    
    Seq_cbarre = select_values_between_points(img_Y, p1[0], p1[1], p2[0], p2[1])
    
    threshold_milo = cb.otsu(Seq_cbarre)
    
    #useful_signal, threshold = cb.extract_code_barre(p1, p2, img_Y)
    
    Seq_cbarre = (Seq_cbarre < threshold_milo).astype(int)
    
    plt.figure(2),
    plt.plot(Seq_cbarre)
    
    
    [deb, fin] = click.extremites(Seq_cbarre)
    
    
    # Alignement des extrémités corrigé
    U_base_tot = 95
    Seq_cbarre_red = Seq_cbarre[deb:fin+1]
    Len_red = len(Seq_cbarre_red)
    Seq_red = np.zeros((Len_red // U_base_tot) * U_base_tot)
    Len_car = len(Seq_red) / U_base_tot
    # Rapport pour rééchantillonnage
    Rap = Len_red / len(Seq_red)
    for i in range(Len_red):
        Seq_red[int(i / Rap)] = Seq_cbarre_red[i]
    
    # Normalisation et seuil pour éviter les erreurs d'échelle
    #Seq_red = (Seq_red < np.mean(Seq_red)) * 1  # Seuil dynamique
    
    plt.figure(3),
    plt.plot(Seq_red)
    
    
    # DÉCODAGE EFFECTIF
    Code_Number = decode_ean13(Seq_red, int(threshold_milo))
    print("Code détecté : ", Code_Number)
    print(" =>Validation<= ", validate_checksum(Code_Number))
    if validate_checksum(Code_Number) :
        return Code_Number
    else :
        return -1
