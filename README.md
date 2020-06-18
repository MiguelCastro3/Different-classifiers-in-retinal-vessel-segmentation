# Different classifiers in retinal vessel segmentation

**PROJECT:** 

This project follows the continuation of the project [Blood vessel segmentation using line operators](https://github.com/MiguelCastro3/Blood-vessel-segmentation-using-line-operators), where the objective is to automatically segment blood vessels in images fundus, using the support vector machine (SVM) which uses vectors to build its final classifier. Here, the number of points extracted at random, the type of normalization (global and individual), the type of classified (linear and rbf) and the percentage of points extracted belonging to thin vessels, thick vessels and background were varied.

**STEPS:** 

* Extraction of features and labels for random points, for vases and background and for thin vases, thick vases and background;
* Application of the classifier and respective segmentation of blood vessels;
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

An example of the results obtained:
![Sem Título](https://user-images.githubusercontent.com/66881028/84937268-60fcf500-b0d3-11ea-87ee-9b9821fea0f4.png)
| Image/Metrics  | Sensitivity | Specificity | Accuracy |
| ------------- | ------------- | ------------- | ------------- |
| 40_training  | 64.45877847208224 | 98.62850631314416	| 94.79986809950357 |  


Effects obtained with the variation of the threshold and the line length:

![threshold](https://user-images.githubusercontent.com/66881028/84935216-4b3a0080-b0d0-11ea-9913-875057e28af9.png)

![comprimento de linha](https://user-images.githubusercontent.com/66881028/84935213-4aa16a00-b0d0-11ea-8271-99ad3d4d77bf.png)
