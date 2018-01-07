# CIP
Computational Intelligence Packages (CIP) for Mathematica

CIP is an open-source high-level function library for (non-linear) curve fitting and data smoothing (with cubic splines), clustering (k-medoids, ART-2a) and machine learning (multiple linear/polynomial regression, feedforward perceptron-type deep neural networks and support vector machines). In addition it provides several heuristics for the selection of training and test data or methods to estimate the relevance of data input components. CIP is built on top of the computing platform Mathematica to exploit its algorithmic and graphical capabilities.

The structure of CIP calculations follows an intuitive and unified Get-Fit-Show/Calculate scheme: With Get methods data are retrieved/simulated (e.g. with the CIP ExperimentalData/CalculatedData package or retrieved elsewhere) that are then submitted to a Fit method (of the CIP CurveFit, Cluster, MLR/MPR, MLP/MLP1/MLP2/MLP3 or SVM package). The result of the latter is a comprehensive info data structure (curveFitInfo, clusterInfo, mlrInfo/mprInfo, mlpInfo/mlp1Info/mlp2Info/mlp3Info or svmInfo) that can be passed to corresponding Show methods for multiple evaluation purposes like visual inspection of the goodness of fit or to Calculate methods for model related calculations. Similar operations of different packages are denoted in a similar manner to ease method changes. Method signatures contain only structural hyper-parameters where technical control parameters may be changed via options if necessary.

The CIP design goals were neither maximum speed nor minimum memory consumption but an intuitive, unified and robust access to high-level functions (not only) for educational purposes. The library packages may be used as a starting point for customized and tailored extensions.

# Packages
**Utility** - basic package that collects several general methods used by other packages like GetMeanSquaredError which is used by all machine learning related packages.

**ExperimentalData** - provides test data. It makes use of the packages Utility, DataTransformation and CurveFit.

**DataTransformation** - performs many internal data transformations for different purposes, e.g. all data that are passed to a machine learning method are scaled before the operation (like ScaleDataMatrix) and re-scaled afterwards (like ScaleDataMatrixReverse). The DataTransformation package comprehends all these methods in a single package. It uses the Utility package.

**Graphics** - tailors Mathematica's graphical functions for diagrams and graphical representations. It uses the Utility and DataTransformation packages.

**CalculatedData** - complements the ExperimentalData package with methods for the generation of simulated data like normally distributed xy-error data around a function for curve fitting with GetXyErrorData. It uses methods from the Utility and DataTransformation packages.

**CurveFit** - tailors Mathematica's built in curve fitting method (NonlinearModelFit) for least-squares minimization and adds a smoothing cubic splines support. It uses the Utility, Graphics, DataTransformation and CalculatedData packages.

**Cluster** - tailors Mathematica's built in FindClusters method for clustering purposes and adds an Adaptive Resonance Theory (ART-2a) support. The package uses the Utility, Graphics and DataTransformation packages.

**MLR/MPR** - tailors Mathematica's built in Fit method for multiple linear/polynomial regression (MLR/MPR). The package uses the Utility, Graphics, DataTransformation and Cluster packages.

**MLP1** (Version 3.0)/**Perceptron** (Version 2.0 and previous) - provides optimization algorithms for a shallow three-layer perceptron-type neural networks (with one hidden neuron layer). It utilizes Mathematica's FindMinimum (ConjugateGradient) or NMinimize (DifferentialEvolution) methods for minimization tasks. The package also provides a backpropagation plus momentum minimization and a classical genetic algorithm based minimization. It uses the Utility, Graphics, DataTransformation and Cluster packages.

**MLP2/MLP3/MLP** - provides optimization algorithms for deep four/five/arbitrary-layer perceptron-type neural networks (with two/three/arbitrary hidden neuron layers). It utilizes Mathematica's FindMinimum (ConjugateGradient) or NMinimize (DifferentialEvolution) methods for minimization tasks. The package also provides a backpropagation plus momentum minimization and a classical genetic algorithm based minimization. It uses the Utility, Graphics, DataTransformation and Cluster packages.

**SVM** - provides constrained optimization algorithms for support vector machines (SVM). It utilizes Mathematica's FindMaximum (InteriorPoint) or NMaximize (DifferentialEvolution) methods for constrained optimization tasks. The package uses the Utility, Graphics, DataTransformation and Cluster packages.

# Additional information
