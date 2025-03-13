import os
import cv2
import numpy as np
from skimage import exposure
from PIL import Image
import shutil

def white_balance_basic(image_bgr):
    """
    Exemple d'une correction de balance des blancs très simplifiée
    en utilisant la fonction cv2.xphoto.simpleWhiteBalance (si installée)
    ou bien une approche "moyenne" manuelle.
    """
    # Si vous avez la contrib OpenCV : cv2.xphoto
    # ------------------------------------------
    # wb = cv2.xphoto.createSimpleWB()
    # wb.setInputMax(255)
    # balanced_bgr = wb.balanceWhite(image_bgr)
    # return balanced_bgr
    
    # Approche simple : on sépare en B, G, R, on calcule la moyenne 
    # et on normalise (pas idéal, mais illustration).
    result = image_bgr.astype(np.float32)
    
    b, g, r = cv2.split(result)
    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)
    
    # Évite une division par zéro
    eps = 1e-5
    b = b * ( (avg_g+avg_r)/(2.0*avg_b + eps) )
    r = r * ( (avg_g+avg_b)/(2.0*avg_r + eps) )
    
    balanced = cv2.merge([b, g, r])
    balanced = np.clip(balanced, 0, 255)
    return balanced.astype(np.uint8)

def adjust_brightness_contrast(image_bgr, alpha=1.0, beta=0):
    """
    Ajuste la luminosité et le contraste d’une image BGR (OpenCV).
    alpha > 1.0 => augmente le contraste,
    beta > 0 => augmente la luminosité.
    """
    # new_image = alpha * image_bgr + beta
    new_image = cv2.convertScaleAbs(image_bgr, alpha=alpha, beta=beta)
    return new_image

def match_histogram(reference_bgr, target_bgr):
    """
    Fait une mise en correspondance d’histogramme entre l’image cible (target)
    et l’image de référence (reference) via scikit-image.

    On convertit d’abord BGR -> RGB, car scikit-image travaille plutôt en RGB.
    """
    ref_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
    tgt_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    matched_rgb = exposure.match_histograms(tgt_rgb, ref_rgb, multichannel=True)
    matched_bgr = cv2.cvtColor((matched_rgb*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return matched_bgr

def process_image(image_path, reference_bgr=None):
    """
    Effectue les différentes étapes de traitement sur une image unique :
    - Lecture
    - Balance des blancs (simplifiée)
    - Ajustement luminosité/contraste (paramètres à affiner)
    - Correspondance d’histogramme par rapport à une image de référence (si présente)
    - etc.
    """
    # Lecture de l'image via OpenCV (en BGR)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Impossible de lire l'image : {image_path}")
        return None
    
    # 1. Balance des blancs
    wb_bgr = white_balance_basic(image_bgr)
    
    # 2. Ajustement luminosité/contraste (alpha, beta à ajuster selon besoin)
    # Exemple : alpha=1.1 => contraste légèrement plus fort, beta=10 => un peu plus clair
    bc_bgr = adjust_brightness_contrast(wb_bgr, alpha=1.1, beta=10)
    
    # 3. Mise en correspondance d'histogramme si on a une image de référence
    if reference_bgr is not None:
        bc_bgr = match_histogram(reference_bgr, bc_bgr)
    
    return bc_bgr

def find_images_in_folder(folder):
    """
    Renvoie la liste des chemins d'images (jpg, jpeg, png, tiff...) d’un répertoire.
    """
    valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    images = []
    for file in os.listdir(folder):
        # Filtre selon l'extension
        if os.path.splitext(file.lower())[1] in valid_exts:
            images.append(os.path.join(folder, file))
    return images

def main(source_root, output_root, reference_image_path=None):
    """
    Parcourt tous les sous-dossiers de 'source_root'. 
    Crée un sous-dossier du même nom dans 'output_root'.
    Applique le traitement à chaque image et enregistre le résultat.
    
    :param source_root: dossier racine contenant les sous-dossiers d'images
    :param output_root: dossier racine où stocker les résultats
    :param reference_image_path: chemin d’une image qui servira de référence 
                                pour la colorimétrie (peut être None).
    """
    # Lecture éventuelle de l'image de référence
    reference_bgr = None
    if reference_image_path is not None and os.path.isfile(reference_image_path):
        reference_bgr = cv2.imread(reference_image_path)
        if reference_bgr is None:
            print(f"Impossible de lire l'image de référence : {reference_image_path}")
            reference_bgr = None
        else:
            print(f"Image de référence chargée : {reference_image_path}")
    
    # Parcours des sous-dossiers de `source_root`
    for root, dirs, files in os.walk(source_root):
        # `root` est le chemin actuel, `dirs` la liste des sous-dossiers
        # On va ignorer le niveau racine si on veut seulement traiter les sous-dossiers
        for d in dirs:
            source_folder = os.path.join(root, d)
            # Construction du chemin correspondant dans output_root
            # On fait un "mirroring" de la structure
            # Supposons que root = source_root => on veut output_folder = output_root/d
            relative_path = os.path.relpath(source_folder, source_root)
            output_folder = os.path.join(output_root, relative_path)
            
            # Créer le sous-dossier de sorties s'il n'existe pas
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
            
            # Récupération de la liste d'images
            images_paths = find_images_in_folder(source_folder)
            print(f"Trouvé {len(images_paths)} image(s) dans {source_folder}")
            
            # Traitement de chaque image
            for img_path in images_paths:
                processed = process_image(img_path, reference_bgr=reference_bgr)
                if processed is not None:
                    # Nom de fichier
                    filename = os.path.basename(img_path)
                    output_path = os.path.join(output_folder, filename)
                    # Sauvegarde
                    cv2.imwrite(output_path, processed)
                    print(f"[OK] {filename} traité et sauvegardé dans {output_folder}")
                else:
                    print(f"[ERREUR] Impossible de traiter {img_path}")

if __name__ == '__main__':
    # Exemple d’utilisation :
    # python batch_homogenize.py
    # (vous pouvez adapter pour lire les arguments via argparse)
    import sys
    
    # Chemins par défaut (à adapter)
    source = "sources"
    output = "sorties"
    reference_img = None  # "chemin/vers/une/image_reference.jpg" si besoin
    
    # Démarrer le traitement
    main(source, output, reference_img)
