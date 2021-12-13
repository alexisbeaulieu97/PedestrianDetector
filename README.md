# Pedestrian Detector

Ce projet est une implémentation de l'article ["Histograms of Oriented Gradients for Human Detection"](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf).


# Utilisation

Installer les modules requis:

``` shell
$ pip install -r requirements.txt
```

Vous pouvez maintenant utiliser le CLI

``` shell
$ python main.py

Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  clear
  test
  train
```

Il y a trois commandes disponibles:
* clear: effacer toutes les données entraînées
* test: tester une image pour savoir à quelle classe elle appartient
* train: entraîner le programme avec de nouvelles données

# Entraînement

Pour entraîner le programme avec de nouvelles données, il faut respecter une structure:
* Répertoire d'entraînement
    * répertoire de nom "x" contenant les données de classe "x"
        * images de classe "x"
    * répertoire de nom "y" contenant les données de classe "y"
        * images de classe "y"

Par exemple:
``` shell
mes_donnees
├───pieton
│       pieton1.jpg
│       pieton2.jpg
│
└───sans_pieton
        sans_pieton1.jpg
        sans_pieton2.jpg
```

Une fois vos fichiers répartis selon la structure suivante, vous pouvez maintenant lancer la commande "train".