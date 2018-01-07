# CIP
**Computational Intelligence Packages (CIP) for Mathematica**

CIP is an open-source high-level function library for (non-linear) curve fitting and data smoothing (with cubic splines), clustering (k-medoids, ART-2a) and machine learning (multiple linear/polynomial regression, feedforward perceptron-type deep neural networks and support vector machines). In addition it provides several heuristics for the selection of training and test data or methods to estimate the relevance of data input components. CIP is built on top of the computing platform Mathematica to exploit its algorithmic and graphical capabilities.

The structure of CIP calculations follows an intuitive and unified Get-Fit-Show/Calculate scheme: With Get methods data are retrieved/simulated (e.g. with the CIP ExperimentalData/CalculatedData package or retrieved elsewhere) that are then submitted to a Fit method (of the CIP CurveFit, Cluster, MLR/MPR, MLP/MLP1/MLP2/MLP3 or SVM package). The result of the latter is a comprehensive info data structure (curveFitInfo, clusterInfo, mlrInfo/mprInfo, mlpInfo/mlp1Info/mlp2Info/mlp3Info or svmInfo) that can be passed to corresponding Show methods for multiple evaluation purposes like visual inspection of the goodness of fit or to Calculate methods for model related calculations. Similar operations of different packages are denoted in a similar manner to ease method changes. Method signatures contain only structural hyper-parameters where technical control parameters may be changed via options if necessary.

The CIP design goals were neither maximum speed nor minimum memory consumption but an intuitive, unified and robust access to high-level functions (not only) for educational purposes. The library packages may be used as a starting point for customized and tailored extensions.

# Packages
*Utility* - basic package that collects several general methods used by other packages like GetMeanSquaredError.

*ExperimentalData* - provides test data.

*DataTransformation* - performs internal data transformations for different purposes, e.g. all data that are passed to a machine learning method are scaled before the operation (like ScaleDataMatrix) and re-scaled afterwards (like ScaleDataMatrixReverse).

*Graphics* - tailors Mathematica's graphical functions for diagrams and graphical representations.

*CalculatedData* - complements the ExperimentalData package with methods for the generation of simulated data like normally distributed xy-error data around a function for curve fitting with GetXyErrorData.

*CurveFit* - tailors Mathematica's built in curve fitting method (NonlinearModelFit) for least-squares minimization and adds a smoothing cubic splines support.

*Cluster* - tailors Mathematica's built in FindClusters method for clustering purposes and adds an Adaptive Resonance Theory (ART-2a) support.

*MLR / MPR* - tailor Mathematica's built in Fit method for multiple linear/polynomial regression (MLR/MPR).

*MLP1* (CIP 3.0) / *Perceptron* (CIP 2.0 and previous) - provide optimization algorithms for a shallow three-layer perceptron-type neural networks (with one hidden neuron layer). They utilize Mathematica's FindMinimum (ConjugateGradient) or NMinimize (DifferentialEvolution) methods for minimization tasks. The packages also provides a backpropagation plus momentum minimization and a classical genetic algorithm based minimization.

*MLP2 / MLP3 / MLP* - provide optimization algorithms for deep four/five/arbitrary-layer perceptron-type neural networks (with two/three/arbitrary hidden neuron layers). They utilize Mathematica's FindMinimum (ConjugateGradient) or NMinimize (DifferentialEvolution) methods for minimization tasks.

*SVM* - provides constrained optimization algorithms for support vector machines (SVM). It utilizes Mathematica's FindMaximum (InteriorPoint) or NMaximize (DifferentialEvolution) methods for constrained optimization tasks.

# Citation

Achim Zielesny, Computational Intelligence Packages (CIP) for Mathematica, Version 3.0, GNWI mbH, Oer-Erkenschwick, Germany, 2018.

# Textbook
**2nd edition:**

[Achim Zielesny, From Curve Fitting to Machine Learning: An Illustrative Guide to Scientific Data Analysis and Computational Intelligence, 2nd (updated and extended) Edition 2016, Springer: Intelligent Systems Reference Library, Volume 109.](https://dx.doi.org/10.1007/978-3-319-32545-3)

The 2nd edition uses CIP 2.0 for all calculations. File *Zielesny_FromCurveFittingToMachineLearning_2ndEdition_Code.zip* contains the complete textbook examples and applications.

**1st edition:**

[Achim Zielesny, From Curve Fitting to Machine Learning: An Illustrative Guide to Scientific Data Analysis and Computational Intelligence, 2011, Springer: Intelligent Systems Reference Library, Volume 18.](http://dx.doi.org/10.1007/978-3-642-21280-2)

The 1st edition uses CIP 1.0 for all calculations. File *Zielesny_FromCurveFittingToMachineLearning_Code.zip* contains the complete textbook examples and applications.

From the reviews of the 1st edition: *'From curve fitting to machine learning' is ... a useful book. ... It contains the basic formulas of curve fitting and related subjects and throws in, what is missing in so many books, the code to reproduce the results. ... All in all this is an interesting and useful book both for novice as well as expert readers. For the novice it is a good introductory book and the expert will appreciate the many examples and working code.* (Leslie A. Piegl, Zentralblatt MATH, Zbl 1236.68004)

# Information about uploaded files
**Code**

*CIP_3.0.zip* (for Mathematica 11 or higher) - CIP 3.0 adds deep multi-layer perceptron-type neural networks, regularization, normalization and minor improvements (see *About.txt*).

*CIP_2.0.zip* (for Mathematica 9 or higher) - CIP 2.0 adds parallelized calculation support and minor improvements (see *About.txt*).

*CIP_1.2.zip* (for Mathematica 7 or higher) - CIP 1.2 adds minor improvements (see *About.txt*).

*CIP_1.1.zip* (for Mathematica 7 or higher) - CIP 1.1 adds MPR and several improvements (see *About.txt*).

*CIP_1.0.zip* (for Mathematica 7 or higher) - CIP 1.0 is the basic operational release.

**Textbook supplementary information**

*Zielesny_FromCurveFittingToMachineLearning_2ndEdition_Code.zip* - complete CIP 2.0 examples and applications of 2nd edition of textbook above.

*Zielesny_FromCurveFittingToMachineLearning_Code.zip* - complete CIP 1.0 examples and applications of 1st edition of textbook above.

**Tutorials and examples**

*CIP_1.2_DocumentCenteredDataAnalysisWorkflows.pdf* - discussion of document-centered data analysis workflows with CIP 1.2 (and corresponding Mathematica notebook *CIP_1.2_DocumentCenteredDataAnalysisWorkflowsNotebook.nb* of the discussed example).

*CIP_1.2_ScientificDataAnalysis.pdf/nb* - overview and examples of CIP 1.2 functions for scientific data analysis (PDF document and Mathematica notebook).

*CIP_1.1_EnteringNonLinearityWithMPR.nb* - entering non-linearity with MPR (Mathematica notebook that uses CIP 1.1).

*CIP_1.1_MinimalModelForWDBCDataClassification.nb* - minimal model for the WDBC Data (Mathematica notebook that uses CIP 1.1).

*CIP_1.1_QSPRwithMLR.nb* - QSPR with MLR (Mathematica notebook that uses CIP 1.1).

*CIP_1.1_QSPRwithSmallDataSet.nb* - QSPR with small Data Set (Mathematica notebook that uses CIP 1.1).

*CIP_1.1_DataCleaningOrSplitting.nb* - data cleaning or splitting (Mathematica notebook that uses CIP 1.1).

# Acknowledgements
The support of [GNWI - Gesellschaft für naturwissenschaftliche Informatik mbH](http://www.gnwi.de) is gratefully acknowledged.
