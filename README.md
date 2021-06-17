# Phosgenite

Phosgenite (Photographs of Stained Glasses Identifier) est un système de reconnaissance de vitraux d'églises.

## Installation

Afin d'utiliser Phosgenite il faut installer les librairies suivantes:

- Orange 3 : ```pip install orange3-imageanalytics```
- Pandas : ```pip install pandas```
- Numpy : ```pip install numpy```
- OpenCV : ```pip install opencv-contrib-python```
- SKlearn : ```pip install scikit-learn```

## Les données

Phosgenite a été développé avec des images de vitraux provenant de l'église Saint Jean au marché (SJ). Ces images sont trouvable dans le dossier ```data/SJ``` et sont séparés en plusieurs sous dossiers:
- Les données d'entrainements stockées dans ```data/SJ/Vitraux baies```
- Les données de tests stockées dans ```data/SJ/test images```

## Utilisation

Appeler la fonction find_label du script main.py avec comme paramètre le chemin relatif de l'image à prédire.

Exemple:

```find_label("/data/SJ/test images/SJ 000 1.jpg")```

La fonction find_label appelle la fonction TSNE_KNN_model qui permet de prédire le label de l'image passée en paramètres.


