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


## Fonctionement

Afin de mieux comprendre comment fonctionne le modèle nous allons le detailler dans un diagramme de flux :

![action_objet 3](https://user-images.githubusercontent.com/72007646/133388610-2b78e388-1eda-41c1-a6b1-9f95760b44e7.PNG)

Voici le role de chaque script:

- La transformation des images en HSV est realisé par le script ```rgb2hsv.py```
- L'embedding est fait grace au script ```embedder.py```
- Le reste du traitement des données (reduction de dimensions avec TSNE et prediction du label avec un KNN) est realisé dans le script ```TSNE_KNN_model.py```




Le script ```cropper.py``` implémente un outil de rognagne automatique des images de vitraux. Ce script est expérimental et ne présente pas tout le temps des résultats satisfaisants. Ce script n'a pas été ajouté a la chaine de traitement des données. Il est possible d'appeler la fonction crop() de ce script dans le constructeur de la classe TSNE_KNN_model avant l'appel de la fonction ```self.transform_test()```

