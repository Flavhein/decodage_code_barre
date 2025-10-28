from decodeBarre import decodage_complet
from decodeBarre import click_point
from search_position import position
import numpy as np

filename = "../livre_code_barre.jpg"

# GÉNÉRATION DES POINTS ICI

(bords1, bords2) = position(filename)

# DÉCODAGE ICI
for i in range (np.shape(bords1)[0]):
    print("bords1:",bords1[i])
    print("bords2:",bords2[i])
    Code_Barre = decodage_complet(filename, bords2[i].astype(int), bords1[i].astype(int))
    # SI -1, alors code barre invalide, sinon code barre valide !
    if (Code_Barre != -1):
        print(Code_Barre)
