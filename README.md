# Different classifiers in retinal vessel segmentation

**PROJECT:** 

This project follows the continuation of the project [Blood vessel segmentation using line operators](https://github.com/MiguelCastro3/Blood-vessel-segmentation-using-line-operators), where the objective is to automatically segment blood vessels in images fundus, using the support vector machine (SVM) which uses vectors to build its final kernel. Here, the number of points extracted at random, the type of normalization (global and individual), the type of classified (linear and rbf) and the percentage of points extracted belonging to thin vessels, thick vessels and background were varied.

**STEPS:** 

* Extraction of features and labels for random points, for vases and background and for thin vases, thick vases and background;
* Application of the kernel and respective segmentation of blood vessels;
* Variation of some variables in order to analyze how they may affect the final results.

**FILES:** 
* [Ricci_Perfetti@2007.pdf](https://github.com/MiguelCastro3/Blood-vessel-segmentation-using-line-operators/blob/master/Ricci_Perfetti%402007.pdf) - (Digital Retinal Images for Vessel Extraction) - scientific article on which the project was based.
* [DRIVE](https://github.com/MiguelCastro3/Blood-vessel-segmentation-using-line-operators/tree/master/DRIVE) - (Digital Retinal Images for Vessel Extraction) - contains image data sets (test and training) with images for segmentation, respective mask and respective manual segmentation.
* [Imagens segmentadas](https://github.com/MiguelCastro3/Blood-vessel-segmentation-using-line-operators/tree/master/Imagens%20segmentadas) - resulting images of the project [Blood vessel segmentation using line operators](https://github.com/MiguelCastro3/Blood-vessel-segmentation-using-line-operators).
* [Guardar+features+e+labels+-+pontos+aleatórios.py](https://github.com/MiguelCastro3/Different-classifiers-in-retinal-vessel-segmentation/blob/master/Guardar%2Bfeatures%2Be%2Blabels%2B-%2Bpontos%2Baleat%C3%B3rios.py) - code with the functionality of extracting features and labels of random points from the whole image, being able to choose the number of points to extract.
* [Guardar+features+e+labels+-+vasos+e+background.py](https://github.com/MiguelCastro3/Different-classifiers-in-retinal-vessel-segmentation/blob/master/Guardar%2Bfeatures%2Be%2Blabels%2B-%2Bvasos%2Be%2Bbackground.py) - code with the functionality of extracting features and point labels belonging to vases and background, being able to choose the percentage of points to be extracted from each of them.
* [Guardar+features+e+labels+-+vasos+finos%2C+vasos+grossos+e+background.py](https://github.com/MiguelCastro3/Different-classifiers-in-retinal-vessel-segmentation/blob/master/Guardar%2Bfeatures%2Be%2Blabels%2B-%2Bvasos%2Bfinos%252C%2Bvasos%2Bgrossos%2Be%2Bbackground.py) - code with the functionality of extracting features and point labels belonging to thin vessels, thick vessels and background, being able to choose the percentage of points to be extracted from each of them.
* [SVM+-+pontos+aleatórios.py](https://github.com/MiguelCastro3/Different-classifiers-in-retinal-vessel-segmentation/blob/master/SVM%2B-%2Bpontos%2Baleat%C3%B3rios.py) - code with the application of SVM to points extracted at random and respective calculation of metrics and ROC curve of the segmentations obtained.
* [SVM+-+vasos+e+background.py ](https://github.com/MiguelCastro3/Different-classifiers-in-retinal-vessel-segmentation/blob/master/SVM%2B-%2Bvasos%2Bfinos%252C%2Bvasos%2Bgrossos%2Be%2Bbackground.py) - code with the application of SVM to points belonging to vessels and background and respective calculation of the metrics and ROC curve of the segmentations obtained.
* [SVM+-+vasos+finos%2C+vasos+grossos+e+background.py](https://github.com/MiguelCastro3/Different-classifiers-in-retinal-vessel-segmentation/blob/master/SVM%2B-%2Bvasos%2Bfinos%252C%2Bvasos%2Bgrossos%2Be%2Bbackground.py) - code with the application of SVM to points belonging to thin vessels, thick vessels and background and respective calculation of metrics and ROC curve of the segmentations obtained.

**RESULTS:** 

* Different segmentations obtained by varying the number of random points extracted.

![1](https://user-images.githubusercontent.com/66881028/85033918-64e35280-b179-11ea-99c6-cdd7171f31fe.png)

| Number of points  | Accuracy (%) | AUC (%) |
| ------------- | ------------- | ------------- |
| 1000 | 94,13 | 83,12 | 
| 2000 | 94,24 | 83,19 | 
| 3000 | 94,32 | 84,14 | 
| 4000 | 94,34 | 83,45 | 
| 5000 | 94,41 | 83,40 | 
| 7500 | 94,43 | 83,50 | 
| 10000 | 94,42 | 82,82 | 

* Different segmentations obtained by varying the number of random points extracted.

![Sem Título](https://user-images.githubusercontent.com/66881028/85034786-577a9800-b17a-11ea-8d31-043a9d655b25.png)

| Image | Normalization | Kernel | Accuracy (%) | AUC (%) |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| a) | Global | Linear | 91,85 | 94,58 | 
| b) | Global | RBF | 92,38 | 81,27 |
| c) | Individual | Linear | 92,97 | 94,56 | 
| d) | Individual | RBF | 93,49 | 81,48
