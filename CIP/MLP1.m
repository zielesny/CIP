(*
-----------------------------------------------------------------------
Computational Intelligence Packages (CIP): Package MLP1
(Multi-Layer Perceptron with 1 Hidden-Unit Layer 
or
Three-Layer Feed-Forward Neural Network)
Version 3.1 for Mathematica 11 or higher
-----------------------------------------------------------------------

Authors: Kolja Berger (parallelization for CIP 2.0), Achim Zielesny 

GNWI - Gesellschaft fuer naturwissenschaftliche Informatik mbH, 
Dortmund, Germany

Citation:
Achim Zielesny, Computational Intelligence Packages (CIP), Version 3.1, 
GNWI mbH (http://www.gnwi.de), Dortmund, Germany, 2020.

Code partially based on:
J. A. Freeman, Simulating Neural Networks with Mathematica, 
Boston 1993, Addison-Wesley Longman Publishing Co.

Copyright 2020 Achim Zielesny

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License (LGPL) as 
published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
Lesser General Public License (LGPL) for more details.

You should have received a copy of the GNU Lesser General Public 
License along with this program. If not, see 
<http://www.gnu.org/licenses/>. 
-----------------------------------------------------------------------
*)

(* ::Section:: *)
(* Frequently used data structures *)

(*
-----------------------------------------------------------------------
Frequently used data structures
-----------------------------------------------------------------------
mlp1Info: {networks, dataSetScaleInfo, mlp1TrainingResults, normalizationInfo, activationAndScaling, optimizationMethod} 

	networks: {weights1, weights2, ...}
	weights: {hiddenWeights, outputWeights}
	hiddenWeights: Weights from input to hidden units
	outputWeights : Weights from hidden to output units
	dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs}, see GetDataSetScaleInfo
	mlp1TrainingResults: {singleTrainingResult1, singleTrainingResult2, ...}
	singleTrainingResult[[i]] corresponds to weights[[i]]
	singleTrainingResult: {trainingMeanSquaredErrorList, testMeanSquaredErrorList}
	trainingMeanSquaredErrorList: {reportPair1, reportPair2, ...}
	reportPair: {reportIteration, mean squared error of report iteration}
	testMeanSquaredErrorList: Same structure as trainingMeanSquaredErrorList
	normalizationInfo: {normalizationType, meanAndStandardDeviationList}, see CIP`DataTransformation`GetDataMatrixNormalizationInfo
	activationAndScaling: See option Mlp1OptionActivationAndScaling
	optimizationMethod: Optimization method
-----------------------------------------------------------------------
*)

(* ::Section:: *)
(* Package and dependencies *)

BeginPackage["CIP`MLP1`", {"CIP`Utility`", "CIP`Graphics`", "CIP`DataTransformation`", "CIP`Cluster`", "Combinatorica`"}]

(* ::Section:: *)
(* Off settings *)

Off[General::"spell1"]
Off[General::shdw]
Off[FindMinimum::cvmit]
Off[General::compat]

(* ::Section:: *)
(* Options *)

Options[Mlp1OptionsTraining] = 
{
	(* True: Multiple mlp1s may be used (one mlp1 for every single output component), False: One mlp1 is used only *)
    Mlp1OptionMultipleMlp1s -> True,
	
	(* Optimization method: "FindMinimum", "NMinimize", "BackpropagationPlusMomentum", "GeneticAlgorithm" *)
    Mlp1OptionOptimizationMethod -> "FindMinimum",
    
    (* Test set *)
    Mlp1OptionTestSet -> {},
    
	(* activationAndScaling: Definition of activation function and corresponding input/output scaling of data
       activationAndScaling: {activation, inputOutputTargetIntervals}
       e.g.
       {{"Sigmoid", "Sigmoid"}, {{-0.9, 0.9}, {0.1, 0.9}}}
       {{"Tanh", "Sigmoid"}, {{-0.9, 0.9}, {0.1, 0.9}}}
       {{"Tanh", "Tanh"}, {{-0.9, 0.9}, {-0.9, 0.9}}}
	   
	   activation: {<Activation function for hidden neurons>, <Activation function for output neurons>}
	   Activation function for hidden/output neurons: "Sigmoid", "Tanh"
	   
	   inputOutputTargetIntervals: {inputTargetInterval, outputTargetInterval}
	   inputTargetInterval/outputTargetInterval contains the desired minimum and maximum value for each column of inputs and outputs
	   inputTargetInterval/outputTargetInterval: {targetMin, targetMax} 
	   targetMin: Minimum value for each column 
	   targetMax: Maximum value for each column *)
	Mlp1OptionActivationAndScaling -> {{"Sigmoid", "Sigmoid"}, {{-0.9, 0.9}, {0.1, 0.9}}},
	
	(* Lambda parameter for L2 regularization: A value of 0.0 means NO L2 regularization
	   Used in methods FitMlp1WithFindMinimum and FitMlp1WithNMinimize *)
	Mlp1OptionLambdaL2Regularization -> 0.0,
	
	(* Cost function type: "SquaredError", "Cross-Entropy"
	   Used in methods FitMlp1WithFindMinimum and FitMlp1WithNMinimize *)
	Mlp1OptionCostFunctionType -> "SquaredError"
}

Options[Mlp1OptionsOptimization] = 
{
	(* Initial weights to be improved (may be empty list)
	   initialWeights: {hiddenWeights, outputWeights}
	   hiddenWeights: Weights from input to hidden units
	   outputWeights: Weights from hidden to output units *)
	Mlp1OptionInitialWeights -> {},

	(* Initial networks for multiple mlp1s training to be improved (may be empty list)
	   networks: {weights1, weights2, ...}
	   weights: {hiddenWeights, outputWeights}
	   hiddenWeights: Weights from input to hidden units
	   outputWeights: Weights from hidden to output units *)
	Mlp1OptionInitialNetworks -> {},
	
    (* Weights for genetic algorithms will be in interval 
       -Mlp1OptionWeightsValueLimit <= weight value <= +Mlp1OptionWeightsValueLimit*)
	Mlp1OptionWeightsValueLimit -> 1000.0,
	
    (* Number of digits for AccuracyGoal and PrecisionGoal (MUST be smaller than MachinePrecision) *)
    Mlp1OptionMinimizationPrecision -> 5,
    
    (* Maximum number of minimization steps *)
    Mlp1OptionMaximumIterations -> 10000,

    (* Number of iterations to improve *)
    Mlp1OptionIterationsToImprove -> 1000,
    
    (* The meanSquaredErrorLists (training protocol) will be filled every reportIteration steps.
       reportIteration <= 0 means no internal reports during training/minimization procedure. *)
    Mlp1OptionReportIteration -> 0
}

Options[Mlp1OptionsBackpropagation] = 
{
    (* Minimum learning parameter *)
    Mlp1OptionLearningParameterMin -> 0.1,
    
    (* Maximum learning parameter *)
    Mlp1OptionLearningParameterMax -> 0.1,
    
    (* Momentum parameter *)
    Mlp1OptionMomentumParameter -> 0.5
}

Options[Mlp1OptionsGeneticAlgorithm] =
{
    (* Size of population *)
    Mlp1OptionPopulationSize -> 50,
    
    (* Crossover probability : 0 to 1.0 *)
    Mlp1OptionCrossoverProbability -> 0.9,
    
    (* Mutation probability : 0 to 1.0 *)
    Mlp1OptionMutationProbability -> 0.9
}

(* ::Section:: *)
(* Declarations *)

BumpFunction::usage = 
	"BumpFunction[x, interval]"

BumpSum::usage = 
	"BumpSum[x, intervals]"

CalculateMlp1Value2D::usage = 
	"CalculateMlp1Value2D[argumentValue, indexOfInput, indexOfFunctionValueOutput, input, mlp1Info]"

CalculateMlp1Value3D::usage = 
	"CalculateMlp1Value3D[argumentValue1, argumentValue2, indexOfInput1, indexOfInput2, indexOfFunctionValueOutput, input, mlp1Info]"

CalculateMlp1ClassNumber::usage = 
	"CalculateMlp1ClassNumber[input, mlp1Info]"

CalculateMlp1ClassNumbers::usage = 
	"CalculateMlp1ClassNumbers[inputs, mlp1Info]"

CalculateMlp1DataSetRmse::usage = 
	"CalculateMlp1DataSetRmse[dataSet, mlp1Info]"

CalculateMlp1Output::usage = 
	"CalculateMlp1Output[input, mlp1Info]"

CalculateMlp1Outputs::usage = 
	"CalculateMlp1Outputs[inputs, mlp1Info]"

FitMlp1::usage = 
	"FitMlp1[dataSet, numberOfHiddenNeurons, options]"

FitMlp1Series::usage = 
	"FitMlp1Series[dataSet, numberOfHiddenNeuronsList, options]"

GetBestMlp1ClassOptimization::usage = 
	"GetBestMlp1ClassOptimization[mlp1TrainOptimization, options]"

GetBestMlp1RegressOptimization::usage = 
	"GetBestMlp1RegressOptimization[mlp1TrainOptimization, options]"

GetNumberOfHiddenNeurons::usage = 
	"GetNumberOfHiddenNeurons[mlp1Info]"

GetMlp1InputInclusionClass::usage = 
	"GetMlp1InputInclusionClass[trainingAndTestSet, numberOfHiddenNeurons, options]"

GetMlp1InputInclusionRegress::usage = 
	"GetMlp1InputInclusionRegress[trainingAndTestSet, numberOfHiddenNeurons, options]"

GetMlp1InputRelevanceClass::usage = 
	"GetMlp1InputRelevanceClass[trainingAndTestSet, numberOfHiddenNeurons, options]"

GetMlp1ClassRelevantComponents::usage = 
    "GetMlp1ClassRelevantComponents[mlp1InputComponentRelevanceListForClassification, numberOfComponents]"

GetMlp1InputRelevanceRegress::usage = 
	"GetMlp1InputRelevanceRegress[trainingAndTestSet, numberOfHiddenNeurons, options]"

GetMlp1RegressRelevantComponents::usage = 
    "GetMlp1RegressRelevantComponents[mlp1InputComponentRelevanceListForRegression, numberOfComponents]"

GetMlp1RegressionResult::usage = 
	"GetMlp1RegressionResult[namedProperty, dataSet, mlp1Info, options]"

GetMlp1SeriesClassificationResult::usage = 
	"GetMlp1SeriesClassificationResult[trainingAndTestSet, mlp1InfoList]"

GetMlp1SeriesRmse::usage = 
	"GetMlp1SeriesRmse[trainingAndTestSet, mlp1InfoList]"

GetMlp1Structure::usage = 
	"GetMlp1Structure[mlp1Info]"

GetMlp1TrainOptimization::usage = 
	"GetMlp1TrainOptimization[dataSet, numberOfHiddenNeurons, trainingFraction, numberOfTrainingSetOptimizationSteps, options]"

GetMlp1Weights::usage = 
	"GetMlp1Weights[mlp1Info, indexOfNetwork]"

ScanClassTrainingWithMlp1::usage = 
	"ScanClassTrainingWithMlp1[dataSet, numberOfHiddenNeurons, trainingFractionList, options]"

ScanRegressTrainingWithMlp1::usage = 
	"ScanRegressTrainingWithMlp1[dataSet, numberOfHiddenNeurons, trainingFractionList, options]"

ShowMlp1Output2D::usage = 
	"ShowMlp1Output2D[indexOfInput, indexOfFunctionValueOutput, input, arguments, mlp1Info]"

ShowMlp1Output3D::usage = 
	"ShowMlp1Output3D[indexOfInput1, indexOfInput2, indexOfFunctionValueOutput, input, mlp1Info, options]"

ShowMlp1ClassificationResult::usage = 
	"ShowMlp1ClassificationResult[namedPropertyList, trainingAndTestSet, mlp1Info]"

ShowMlp1SingleClassification::usage = 
	"ShowMlp1SingleClassification[namedPropertyList, classificationDataSet, mlp1Info]"

ShowMlp1ClassificationScan::usage = 
	"ShowMlp1ClassificationScan[mlp1ClassificationScan, options]"

ShowMlp1InputRelevanceClass::usage = 
	"ShowMlp1InputRelevanceClass[mlp1InputComponentRelevanceListForClassification, options]"
	
ShowMlp1InputRelevanceRegress::usage = 
	"ShowMlp1InputRelevanceRegress[mlp1InputComponentRelevanceListForRegression, options]"

ShowMlp1RegressionResult::usage = 
	"ShowMlp1RegressionResult[namedPropertyList, trainingAndTestSet, mlp1Info]"

ShowMlp1SingleRegression::usage = 
	"ShowMlp1SingleRegression[namedPropertyList, dataSet, mlp1Info]"

ShowMlp1RegressionScan::usage = 
	"ShowMlp1RegressionScan[mlp1RegressionScan, options]"

ShowMlp1SeriesClassificationResult::usage = 
	"ShowMlp1SeriesClassificationResult[mlp1SeriesClassificationResult, options]"

ShowMlp1SeriesRmse::usage = 
	"ShowMlp1SeriesRmse[mlp1SeriesRmse, options]"

ShowMlp1Training::usage = 
	"ShowMlp1Training[mlp1Info]"

ShowMlp1TrainOptimization::usage = 
	"ShowMlp1TrainOptimization[mlp1TrainOptimization, options]"

SigmoidFunction::usage = 
	"SigmoidFunction[x]"

(* ::Section:: *)
(* Functions *)
	Begin["`Private`"]

BumpFunction[

	(* Bump function.

	   Returns:
	   Bump function value *)

	(* Argument *)
	x_?NumberQ,
	
	(* Bump interval: {minValue, maxValue} *)
	interval_/;VectorQ[interval, NumberQ]
	
	] := SigmoidFunction[x - interval[[1]]] - SigmoidFunction[x - interval[[2]]];

BumpSum[

	(* Bump sum function.

	   Returns:
	   Bump sum value *)

	(* Argument *)
	x_?NumberQ,
	
	(* Bump intervals: {interval1, interval2, ...} 
	   interval: {minValue, maxValue} *)
	intervals_/;MatrixQ[intervals, NumberQ]

	] := 
	
	Module[
    
		{
			i
		},
		
		Return[
			Sum[BumpFunction[x, intervals[[i]]], {i, Length[intervals]}]
		]	
	];

CalculateMlp1Value2D[

	(* Calculates 2D output for specified argument and input for specified 3-Layer-True-Unit mlp1.
	   This special method assumes an input and an output with one component only.

	   Returns:
	   Value of specified output neuron for argument *)

    (* Argument value for neuron with index indexOfInput *)
    argumentValue_?NumberQ,
    
  	(* See "Frequently used data structures" *)
    mlp1Info_
    
	] :=
  
	Module[
    
		{
			indexOfInput,
			indexOfFunctionValueOutput,
			input
		},

		indexOfInput = 1;
		indexOfFunctionValueOutput = 1;
		input = {0.0};
		Return[
			CalculateMlp1Value2D[argumentValue, indexOfInput, indexOfFunctionValueOutput, input, mlp1Info]
		]
	];

CalculateMlp1Value2D[

	(* Calculates 2D output for specified argument and input for specified 3-Layer-True-Unit mlp1.

	   Returns:
	   Value of specified output neuron for argument and input *)

    (* Argument value for neuron with index indexOfInput *)
    argumentValue_?NumberQ,
    
    (* Index of input neuron that receives argumentValue *)
    indexOfInput_?IntegerQ,

    (* Index of output neuron that returns function value *)
    indexOfFunctionValueOutput_?IntegerQ,
    
    (* Mlp1 input in original units: 
       inputsInOriginalUnits = {inputValue1, inputValue2, ...} 
       Value of input neuron with specified index (indexOfInput) is replaced by argumentValue *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlp1Info_
    
	] :=
  
	Module[
    
		{
			currentInput,
			output
		},
		
		currentInput = ReplacePart[input, {indexOfInput -> argumentValue}];
		output = CalculateMlp1Output[currentInput, mlp1Info];
		Return[output[[indexOfFunctionValueOutput]]];
	];

CalculateMlp1Value3D[

	(* Calculates 3D output for specified arguments for specified 3-Layer-True-Unit mlp1. 
	   This specific methods assumes a 3-Layer-True-Unit mlp1 with 2 input neurons and 1 output neuron.

	   Returns:
	   Value of the single output neuron for arguments *)


    (* Argument value for neuron with index indexOfInput1 *)
    argumentValue1_?NumberQ,
    
    (* Argument value for neuron with index indexOfInput2 *)
    argumentValue2_?NumberQ,
    
  	(* See "Frequently used data structures" *)
    mlp1Info_
    
	] :=
  
	Module[
    
		{
			indexOfInput1,
			indexOfInput2,
			indexOfOutput,
			input
		},
		
		indexOfInput1 = 1;
		indexOfInput2 = 2;
		indexOfOutput = 1;
		input = {0.0, 0.0};
		Return[
			CalculateMlp1Value3D[argumentValue1, argumentValue2, indexOfInput1, indexOfInput2, indexOfOutput, input, mlp1Info]
		]
	];

CalculateMlp1Value3D[

	(* Calculates 3D output for specified arguments and input for specified 3-Layer-True-Unit mlp1.

	   Returns:
	   Value of specified output neuron for arguments and input *)


    (* Argument value for neuron with index indexOfInput1 *)
    argumentValue1_?NumberQ,
    
    (* Argument value for neuron with index indexOfInput2 *)
    argumentValue2_?NumberQ,
    
    (* Index of input neuron that receives argumentValue1 *)
    indexOfInput1_?IntegerQ,

    (* Index of input neuron that receives argumentValue2 *)
    indexOfInput2_?IntegerQ,

    (* Index of output neuron that returns function value *)
    indexOfFunctionValueOutput_?IntegerQ,
    
    (* Mlp1 input in original units: 
       inputsInOriginalUnits = {inputValue1, inputValue2, ...} 
       Value of input neurons with specified indices (indexOfInput1, indexOfInput2) are replaced by argument values *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlp1Info_
    
	] :=
  
	Module[
    
		{
			currentInput,
			output
		},
		
		currentInput = ReplacePart[input, {indexOfInput1 -> argumentValue1, indexOfInput2 -> argumentValue2}];
		output = CalculateMlp1Output[currentInput, mlp1Info];
		Return[output[[indexOfFunctionValueOutput]]];
	];

CalculateMlp1ClassNumber[

	(* Returns class number for specified input for 3-Layer-True-Unit classification mlp1 with specified weights.

	   Returns:
	   Class number of input *)

    
    (* input in original units: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    input_/;VectorQ[input, NumberQ],
        
  	(* See "Frequently used data structures" *)
    mlp1Info_
    
	] :=
  
	Module[
    
		{
			combinedOutputs,
			hiddenWeights,
			i,
			networks,
			scaledInputs,
			outputs,
			outputWeights,
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			weights
		},
    
    	networks = mlp1Info[[1]];
    	dataSetScaleInfo = mlp1Info[[2]];
    	normalizationInfo = mlp1Info[[4]];
    	activationAndScaling = mlp1Info[[5]];

		If[Length[networks] == 1,
	
			(* --------------------------------------------------------------------------------
			   One network
			   -------------------------------------------------------------------------------- *)		

			weights = networks[[1]];
    		hiddenWeights = weights[[1]];
			outputWeights = weights[[2]];
			(* Transform original input *)
			scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo];
			outputs = GetInternalMlp1Outputs[scaledInputs, hiddenWeights, outputWeights, activationAndScaling];
			Return[CIP`Utility`GetPositionOfMaximumValue[outputs[[1]]]],
			
			(* --------------------------------------------------------------------------------
			   Multiple networks (with ONE output value each)
			   -------------------------------------------------------------------------------- *)		

			combinedOutputs =
				Table[
					weights = networks[[i]];
		    		hiddenWeights = weights[[1]];
					outputWeights = weights[[2]];
					(* Transform original input *)
					scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo];
					outputs = GetInternalMlp1Outputs[scaledInputs, hiddenWeights, outputWeights, activationAndScaling];
					outputs[[1, 1]],
					
					{i, Length[networks]}
				];
			Return[CIP`Utility`GetPositionOfMaximumValue[combinedOutputs]]
		]
	];

CalculateMlp1ClassNumbers[

	(* Returns class numbers for specified inputs for 3-Layer-True-Unit classification mlp1 with specified weights.

	   Returns:
	   {class number of input1, class number of input2, ...} *)

    
    (* {inputsInOriginalUnit1, inputsInOriginalUnit2, ...}
        inputsInOriginalUnit: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    inputs_/;MatrixQ[inputs, NumberQ],
        
  	(* See "Frequently used data structures" *)
    mlp1Info_
    
	] :=
  
	Module[
    
		{
			combinedOutputs,
			correspondingOutput,
			hiddenWeights,
			i,
			networks,
			scaledInputs,
			outputs,
			outputWeights,
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			weights
		},

    	networks = mlp1Info[[1]];
    	dataSetScaleInfo = mlp1Info[[2]];
    	normalizationInfo = mlp1Info[[4]];
    	activationAndScaling = mlp1Info[[5]];

		If[Length[networks] == 1,
	
			(* --------------------------------------------------------------------------------
			   One network
			   -------------------------------------------------------------------------------- *)		

			weights = networks[[1]];
    		hiddenWeights = weights[[1]];
			outputWeights = weights[[2]];
			(* Transform original inputs *)
			scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
			outputs = GetInternalMlp1Outputs[scaledInputs, hiddenWeights, outputWeights, activationAndScaling];
			Return[
				Table[CIP`Utility`GetPositionOfMaximumValue[outputs[[i]]], {i, Length[outputs]}]
			],
			
			(* --------------------------------------------------------------------------------
			   Multiple networks (with ONE output value each)
			   -------------------------------------------------------------------------------- *)		

			combinedOutputs =
				Table[
					weights = networks[[i]];
		    		hiddenWeights = weights[[1]];
					outputWeights = weights[[2]];
					(* Transform original input *)
					scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
					outputs = GetInternalMlp1Outputs[scaledInputs, hiddenWeights, outputWeights, activationAndScaling];
					Flatten[outputs],
					
					{i, Length[networks]}
				];
			Return[
				Table[
					correspondingOutput = combinedOutputs[[All, i]];
					CIP`Utility`GetPositionOfMaximumValue[correspondingOutput],
				
					{i, Length[First[combinedOutputs]]}
				]
			]
		]
	];

CalculateMlp1CorrectClassificationInPercent[

	(* Returns correct classification in percent for classification data set.

	   Returns: 
	   Correct classification in percent for classification data set *)


	(* classificationDataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} 
	   outputValue: 0/1
	   Data set must be a classification data set, i.e. the output components must 0/1 code a class,
	   i.e. class 4 of 5 must be coded {0, 0, 0, 1, 0} *)
    classificationDataSet_,

  	(* See "Frequently used data structures" *)
    mlp1Info_
    
	] :=
  
	Module[
    
		{
			pureFunction
		},

		pureFunction = Function[inputs, CalculateMlp1ClassNumbers[inputs, mlp1Info]];
		Return[CIP`Utility`GetCorrectClassificationInPercent[classificationDataSet, pureFunction]]
	];

CalculateMlp1DataSetRmse[

	(* Returns RMSE of data set.

	   Returns: 
	   RMSE of data set *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

  	(* See "Frequently used data structures" *)
    mlp1Info_
    
	] :=
  
	Module[
    
		{
			pureFunction,
			rmse
		},

		pureFunction = Function[inputs, CalculateMlp1Outputs[inputs, mlp1Info]];
		rmse = Sqrt[CIP`Utility`GetMeanSquaredError[dataSet, pureFunction]];
		Return[rmse]
	];

CalculateMlp1Output[

	(* Calculates output for specified input for specified 3-Layer-True-Unit mlp1.

	   Returns:
	   output: {transformedValueOfOutput1, transformedValueOfOutput2, ...} *)

    
    (* Input in original units: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlp1Info_
    
	] :=
  
	Module[
    
		{
			combinedOutputs,
			dataMatrixScaleInfo,
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			hiddenWeights,
			i,
			networks,
			outputsInOriginalUnits,
			scaledOutputs,
			outputWeights,
			scaledInputs,
			weights
		},
    
    	networks = mlp1Info[[1]];
    	dataSetScaleInfo = mlp1Info[[2]];
    	normalizationInfo = mlp1Info[[4]];
    	activationAndScaling = mlp1Info[[5]];

		If[Length[networks] == 1,
	
			(* --------------------------------------------------------------------------------
			   One network (with multiple output values)
			   -------------------------------------------------------------------------------- *)		

			weights = networks[[1]];
    		hiddenWeights = weights[[1]];
			outputWeights = weights[[2]];
			(* Transform original input *)
			scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo];
			scaledOutputs = GetInternalMlp1Outputs[scaledInputs, hiddenWeights, outputWeights, activationAndScaling];
			(* Transform outputs to original units *)
			outputsInOriginalUnits = CIP`DataTransformation`ScaleDataMatrixReverse[scaledOutputs, dataSetScaleInfo[[2]]];
			Return[First[outputsInOriginalUnits]],
			
			(* --------------------------------------------------------------------------------
			   Multiple networks (with ONE output value each)
			   -------------------------------------------------------------------------------- *)		

			combinedOutputs =
				Table[
					weights = networks[[i]];
		    		hiddenWeights = weights[[1]];
					outputWeights = weights[[2]];
					(* Transform original input *)
					scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo];
					scaledOutputs = GetInternalMlp1Outputs[scaledInputs, hiddenWeights, outputWeights, activationAndScaling];

					(* Transform outputs to original units:
					   dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs} 
					   dataMatrixScaleInfo: {minMaxList, targetInterval} *)					
					dataMatrixScaleInfo = {{dataSetScaleInfo[[2, 1, i]]}, dataSetScaleInfo[[2, 2]]};
					outputsInOriginalUnits = CIP`DataTransformation`ScaleDataMatrixReverse[scaledOutputs, dataMatrixScaleInfo];
					outputsInOriginalUnits[[1, 1]],
					
					{i, Length[networks]}
				];
			Return[combinedOutputs]
		]
	];

CalculateMlp1Outputs[

	(* Calculates outputs for specified inputs for specified 3-Layer-True-Unit mlp1.

	   Returns:
	   outputs: {output1, output2, ...} 
	   output: {transformedValueOfOutput1, transformedValueOfOutput1, ...} *)

    
    (* {inputsInOriginalUnit1, inputsInOriginalUnit2, ...}
        inputsInOriginalUnit: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    inputs_/;MatrixQ[inputs, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlp1Info_
    
	] :=
  
	Module[
    
		{
			combinedOutputs,
			dataMatrixScaleInfo,
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			hiddenWeights,
			i,
			networks,
			outputsInOriginalUnits,
			scaledOutputs,
			outputWeights,
			scaledInputs,
			weights
		},
		
    	networks = mlp1Info[[1]];
    	dataSetScaleInfo = mlp1Info[[2]];
    	normalizationInfo = mlp1Info[[4]];
    	activationAndScaling = mlp1Info[[5]];

		If[Length[networks] == 1,
	
			(* --------------------------------------------------------------------------------
			   One network (with multiple output values)
			   -------------------------------------------------------------------------------- *)		

			weights = networks[[1]];
    		hiddenWeights = weights[[1]];
			outputWeights = weights[[2]];
			(* Transform original inputs *)
			scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
			scaledOutputs = GetInternalMlp1Outputs[scaledInputs, hiddenWeights, outputWeights, activationAndScaling];
			(* Transform outputs to original units *)
			outputsInOriginalUnits = CIP`DataTransformation`ScaleDataMatrixReverse[scaledOutputs, dataSetScaleInfo[[2]]];
			Return[outputsInOriginalUnits],
			
			(* --------------------------------------------------------------------------------
			   Multiple networks (with ONE output value each)
			   -------------------------------------------------------------------------------- *)		

			combinedOutputs =
				Table[
					weights = networks[[i]];
		    		hiddenWeights = weights[[1]];
					outputWeights = weights[[2]];
					(* Transform original input *)
					scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
					scaledOutputs = GetInternalMlp1Outputs[scaledInputs, hiddenWeights, outputWeights, activationAndScaling];

					(* Transform outputs to original units:
					   dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs} 
					   dataMatrixScaleInfo: {minMaxList, targetInterval} *)					
					dataMatrixScaleInfo = {{dataSetScaleInfo[[2, 1, i]]}, dataSetScaleInfo[[2, 2]]};
					outputsInOriginalUnits = CIP`DataTransformation`ScaleDataMatrixReverse[scaledOutputs, dataMatrixScaleInfo];
					Flatten[outputsInOriginalUnits],
					
					{i, Length[networks]}
				];
			Return[Transpose[combinedOutputs]]
		]
	];

CrossoverChromosomes[

	(* Returns 2 chromosomes after random crossover between chromosomes and random mutation of each chromosome.
	   NOTE: Uses random number. A possible seed must be set in superior methods.

	   Returns:
	  {chromosome1, chromosome2}
	   chromosome: {hiddenWeights, outputWeights} *)

    
    (* Chromosome has form: {hiddenWeights, outputWeights} *)
    chromosome1_,
    
    (* Chromosome has form: {hiddenWeights, outputWeights} *)
    chromosome2_,
    
    crossoverProbability_?NumberQ,
    
    mutationProbability_?NumberQ,
    
    (* Mutated value will be in interval -mutatedValueBound <= mutated value <= +mutatedValueBound *)
    mutatedValueBound_?NumberQ
    
	] := 
	
	Module[
    
		{
			i,
			randomNumber,
			randomPosition1,
			randomPosition2,
			pos1,
			pos2,
			xOverStart,
			xOverEnd,
			newChromosome1,
			newChromosome2
		},
    
		randomNumber = RandomReal[];
    
		If[randomNumber <= crossoverProbability,
      
			(* ----------------------------------------------------------------------------- *)
			(* Perform crossover and mutation: First choose layer : Hidden or output weights *)
			(* ----------------------------------------------------------------------------- *)
			randomPosition1 = RandomInteger[{1, 2}];
			(* Second choose neuron : Hidden or output neuron *)
			randomPosition2 = RandomInteger[{1, Length[chromosome1[[randomPosition1]] ]}];
			(* Determine positions for crossover start and end *)
			pos1 = RandomInteger[{1, Length[chromosome1[[randomPosition1, randomPosition2]] ]}];
			pos2 = pos1;
			While[pos2 == pos1, 
				pos2 = RandomInteger[{1, Length[chromosome1[[randomPosition1, randomPosition2]] ]}]
			];
			xOverStart = Min[pos1, pos2];
			xOverEnd = Max[pos1, pos2];
			(* Perform crossover *)
			newChromosome1 = chromosome1;
			Do[
				newChromosome1 = ReplacePart[newChromosome1, {randomPosition1, randomPosition2, i} -> chromosome2[[randomPosition1, randomPosition2, i]]],
        
				{i, xOverStart, xOverEnd}
			];
			newChromosome2 = chromosome2;
			Do[
				newChromosome2 = ReplacePart[newChromosome2, {randomPosition1, randomPosition2, i} -> chromosome1[[randomPosition1, randomPosition2, i]]],
        
				{i, xOverStart, xOverEnd}
			];
			Return[
				{
					MutateChromosome[newChromosome1, mutationProbability, mutatedValueBound],
          			MutateChromosome[newChromosome2, mutationProbability, mutatedValueBound]
				}
			],
      
			(* ----------------------------------------------------------------------------- *)
			(* No crossover, only mutation *)
			(* ----------------------------------------------------------------------------- *)
			Return[
				{
					MutateChromosome[chromosome1, mutationProbability, mutatedValueBound],
					MutateChromosome[chromosome2, mutationProbability, mutatedValueBound]
				}
			]
		]
	];

FitMultipleMlp1SC[

	(* Training of multiple (1 mlp1 per output component of data set) 3-Layer True-Unit Mlp1.
	
	   Returns:
	   mlp1Info (see "Frequently used data structures") *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,
	
	(* Options *)
	opts___

	] :=
  
	Module[
    
		{
			crossoverProbability,
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			normalizationType,
			i,
			initialNetworks,
			initialWeights,
			learningParameterMin,
			learningParameterMax,
			maximumNumberOfIterations,
			minimizationPrecision,
			momentumParameter,
			multipleTestSet,
			multipleTrainingSet,
			mutationProbability,
			networks,
			numberOfIterationsToImprove,
			mlp1Info,
			mlp1TrainingResults,
			populationSize,
			randomValueInitialization,
			reportIteration,
			testSet,
			optimizationMethod,
			trainingSet,
			weightsValueLimit,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		reportIteration = Mlp1OptionReportIteration/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		dataSetScaleInfo = CIP`DataTransformation`GetDataSetScaleInfoForTrainingAndTestSet[trainingAndTestSet, activationAndScaling[[2, 1]], activationAndScaling[[2, 2]]];
		normalizationInfo = CIP`DataTransformation`GetDataSetNormalizationInfoForTrainingAndTestSet[trainingAndTestSet, normalizationType, dataSetScaleInfo];
		trainingSet = trainingAndTestSet[[1]];
		testSet = trainingAndTestSet[[2]];
		multipleTrainingSet = CIP`DataTransformation`TransformDataSetToMultipleDataSet[trainingSet];
		If[Length[testSet] > 0,
			
			multipleTestSet = CIP`DataTransformation`TransformDataSetToMultipleDataSet[testSet],
			
			multipleTestSet = Table[{}, {i, Length[multipleTrainingSet]}]
		];

		(* ----------------------------------------------------------------------------------------------------
		   Training
		   ---------------------------------------------------------------------------------------------------- *)
		networks = {};
		mlp1TrainingResults = {};
		Do[
			(* If initial networks are available overwrite initialWeights *)
			If[Length[initialNetworks] > 0 && Length[initialNetworks] == Length[multipleTrainingSet],
				initialWeights = initialNetworks[[i]];
			];
			mlp1Info = 
				FitSingleMlp1[
					{multipleTrainingSet[[i]], multipleTestSet[[i]]},
					numberOfHiddenNeurons,
	    			Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp1OptionReportIteration -> reportIteration,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp1OptionActivationAndScaling -> activationAndScaling,
	    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp1OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				];
			AppendTo[networks, mlp1Info[[1, 1]]];
			AppendTo[mlp1TrainingResults, mlp1Info[[3, 1]]],
			
			{i, Length[multipleTrainingSet]}
		];

		(* ----------------------------------------------------------------------------------------------------
		   Return mlp1Info
		   ---------------------------------------------------------------------------------------------------- *)
		Return[{networks, dataSetScaleInfo, mlp1TrainingResults, normalizationInfo, activationAndScaling}]		
	];
	
FitMultipleMlp1PC[

	(* Training of multiple (1 mlp1 per output component of data set) 3-Layer True-Unit Mlp1.
	
	   Returns:
	   mlp1Info (see "Frequently used data structures") *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,
	
	(* Options *)
	opts___

	] :=
  
	Module[
    
		{
			crossoverProbability,
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			normalizationType,
			i,
			initialNetworks,
			initialWeights,
			learningParameterMin,
			learningParameterMax,
			maximumNumberOfIterations,
			minimizationPrecision,
			momentumParameter,
			multipleTestSet,
			multipleTrainingSet,
			mutationProbability,
			networks,
			numberOfIterationsToImprove,
			mlp1TrainingResults,
			populationSize,
			randomValueInitialization,
			reportIteration,
			testSet,
			optimizationMethod,
			trainingSet,
			weightsValueLimit,
			mlp1InfoList,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		reportIteration = Mlp1OptionReportIteration/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		dataSetScaleInfo = CIP`DataTransformation`GetDataSetScaleInfoForTrainingAndTestSet[trainingAndTestSet, activationAndScaling[[2, 1]], activationAndScaling[[2, 2]]];
		normalizationInfo = CIP`DataTransformation`GetDataSetNormalizationInfoForTrainingAndTestSet[trainingAndTestSet, normalizationType, dataSetScaleInfo];
		trainingSet = trainingAndTestSet[[1]];
		testSet = trainingAndTestSet[[2]];
		multipleTrainingSet = CIP`DataTransformation`TransformDataSetToMultipleDataSet[trainingSet];
		If[Length[testSet] > 0,
			
			multipleTestSet = CIP`DataTransformation`TransformDataSetToMultipleDataSet[testSet],
			
			multipleTestSet = Table[{}, {i, Length[multipleTrainingSet]}]
		];
		
		ParallelNeeds[{"CIP`Mlp1`", "CIP`DataTransformation`", "CIP`Utility`", "Combinatorica`"}];
		DistributeDefinitions[initialNetworks, multipleTrainingSet, multipleTestSet, optimizationMethod, initialWeights, 
			weightsValueLimit, minimizationPrecision, maximumNumberOfIterations, numberOfIterationsToImprove, 
			reportIteration, learningParameterMin, learningParameterMax, momentumParameter, populationSize, 
			crossoverProbability, mutationProbability, randomValueInitialization, activationAndScaling, normalizationType, 
			lambdaL2Regularization, costFunctionType];

		(* ----------------------------------------------------------------------------------------------------
		   Training
		   ---------------------------------------------------------------------------------------------------- *)
		mlp1InfoList = ParallelTable[
			(* If initial networks are available overwrite initialWeights *)
			If[Length[initialNetworks] > 0 && Length[initialNetworks] == Length[multipleTrainingSet],
				initialWeights = initialNetworks[[i]]
			];
			
			FitSingleMlp1[
				{multipleTrainingSet[[i]], multipleTestSet[[i]]},
				numberOfHiddenNeurons,
	    		Mlp1OptionOptimizationMethod -> optimizationMethod,
				Mlp1OptionInitialWeights -> initialWeights,
				Mlp1OptionWeightsValueLimit -> weightsValueLimit,
				Mlp1OptionMinimizationPrecision -> minimizationPrecision,
				Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
				Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
	 			Mlp1OptionReportIteration -> reportIteration,
				Mlp1OptionLearningParameterMin -> learningParameterMin,
				Mlp1OptionLearningParameterMax -> learningParameterMax,
				Mlp1OptionMomentumParameter -> momentumParameter,
				Mlp1OptionPopulationSize -> populationSize,
				Mlp1OptionCrossoverProbability -> crossoverProbability,
				Mlp1OptionMutationProbability -> mutationProbability,
	    		UtilityOptionRandomInitializationMode -> randomValueInitialization,
	   			Mlp1OptionActivationAndScaling -> activationAndScaling,
	   			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
	   			Mlp1OptionCostFunctionType -> costFunctionType,
	   			DataTransformationOptionNormalizationType -> normalizationType
			],
			{i, Length[multipleTrainingSet]}
		];
		networks = Table[mlp1InfoList[[i, 1, 1]], {i, Length[multipleTrainingSet]}];
		mlp1TrainingResults = Table[mlp1InfoList[[i, 3, 1]], {i, Length[multipleTrainingSet]}];
		(* ----------------------------------------------------------------------------------------------------
		   Return mlp1Info
		   ---------------------------------------------------------------------------------------------------- *)
		Return[{networks, dataSetScaleInfo, mlp1TrainingResults, normalizationInfo, activationAndScaling}]		
	];

FitMlp1[

	(* Training of single or multiple 3-Layer True-Unit Mlp1(s).

	   Returns:
	   mlp1Info (see "Frequently used data structures") *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			crossoverProbability,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			learningParameterMin,
			learningParameterMax,
			maximumNumberOfIterations,
			minimizationPrecision,
			momentumParameter,
			multipleMlp1s,
			mutationProbability,
			numberOfIterationsToImprove,
			populationSize,
			randomValueInitialization,
			reportIteration,
			activationAndScaling,
			normalizationType,
			testSet,
			trainingAndTestSet,
			optimizationMethod,
			parallelization,
			lambdaL2Regularization,
			costFunctionType
		},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
	    testSet = Mlp1OptionTestSet/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		reportIteration = Mlp1OptionReportIteration/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		(* ----------------------------------------------------------------------------------------------------
		   Switch training method
		   ---------------------------------------------------------------------------------------------------- *)
		trainingAndTestSet = {dataSet, testSet};
		
		If[multipleMlp1s,
			
			Switch[parallelization,
			
				(* ------------------------------------------------------------------------------- *)
				"ParallelCalculation",
				Return[
					FitMultipleMlp1PC[
						trainingAndTestSet,
						numberOfHiddenNeurons,
		    			Mlp1OptionOptimizationMethod -> optimizationMethod,
						Mlp1OptionInitialWeights -> initialWeights,
						Mlp1OptionInitialNetworks -> initialNetworks,
						Mlp1OptionWeightsValueLimit -> weightsValueLimit,
						Mlp1OptionMinimizationPrecision -> minimizationPrecision,
						Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
			 			Mlp1OptionReportIteration -> reportIteration,
						Mlp1OptionLearningParameterMin -> learningParameterMin,
						Mlp1OptionLearningParameterMax -> learningParameterMax,
						Mlp1OptionMomentumParameter -> momentumParameter,
						Mlp1OptionPopulationSize -> populationSize,
						Mlp1OptionCrossoverProbability -> crossoverProbability,
						Mlp1OptionMutationProbability -> mutationProbability,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			Mlp1OptionActivationAndScaling -> activationAndScaling,
		    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
		    			Mlp1OptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					]
				],

				(* ------------------------------------------------------------------------------- *)
				"SequentialCalculation",
				Return[
					FitMultipleMlp1SC[
						trainingAndTestSet,
						numberOfHiddenNeurons,
		    			Mlp1OptionOptimizationMethod -> optimizationMethod,
						Mlp1OptionInitialWeights -> initialWeights,
						Mlp1OptionInitialNetworks -> initialNetworks,
						Mlp1OptionWeightsValueLimit -> weightsValueLimit,
						Mlp1OptionMinimizationPrecision -> minimizationPrecision,
						Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
			 			Mlp1OptionReportIteration -> reportIteration,
						Mlp1OptionLearningParameterMin -> learningParameterMin,
						Mlp1OptionLearningParameterMax -> learningParameterMax,
						Mlp1OptionMomentumParameter -> momentumParameter,
						Mlp1OptionPopulationSize -> populationSize,
						Mlp1OptionCrossoverProbability -> crossoverProbability,
						Mlp1OptionMutationProbability -> mutationProbability,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			Mlp1OptionActivationAndScaling -> activationAndScaling,
		    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
		    			Mlp1OptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					]
				]
			],
			
			Return[
				FitSingleMlp1[
					trainingAndTestSet,
					numberOfHiddenNeurons,
	    			Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp1OptionReportIteration -> reportIteration,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp1OptionActivationAndScaling -> activationAndScaling,
	    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp1OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				]
			]
		]
	];

FitMlp1Series[

	(* Trains of a series of single or multiple 3-Layer True-Unit Mlp1(s).

	   Returns:
	   mlp1InfoList: {mlp1Info1, mlp1Info2, ...}
	   mlp1Info[[i]] corresponds to numberOfHiddenNeuronsList[[i]]
	   (see "Frequently used data structures") *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* List with numbers of hidden neurons *)
	numberOfHiddenNeuronsList_/;VectorQ[numberOfHiddenNeuronsList, IntegerQ],

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			crossoverProbability,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			learningParameterMin,
			learningParameterMax,
			maximumNumberOfIterations,
			minimizationPrecision,
			momentumParameter,
			multipleMlp1s,
			mutationProbability,
			numberOfIterationsToImprove,
			populationSize,
			randomValueInitialization,
			reportIteration,
			activationAndScaling,
			normalizationType,
			testSet,
			optimizationMethod,
			parallelization,
			lambdaL2Regularization,
			costFunctionType
		},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
	    testSet = Mlp1OptionTestSet/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		reportIteration = Mlp1OptionReportIteration/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
	    (* Parallelization options *)
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				FitMlp1SeriesPC[
					dataSet,
					numberOfHiddenNeuronsList,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
				    Mlp1OptionOptimizationMethod -> optimizationMethod,
				    Mlp1OptionTestSet -> testSet,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
					Mlp1OptionReportIteration -> reportIteration,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
				    Mlp1OptionActivationAndScaling -> activationAndScaling,
				    Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
				    Mlp1OptionCostFunctionType -> costFunctionType,
				    DataTransformationOptionNormalizationType -> normalizationType
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				FitMlp1SeriesSC[
					dataSet,
					numberOfHiddenNeuronsList,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
				    Mlp1OptionOptimizationMethod -> optimizationMethod,
				    Mlp1OptionTestSet -> testSet,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
					Mlp1OptionReportIteration -> reportIteration,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
				    Mlp1OptionActivationAndScaling -> activationAndScaling,
				    Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
				    Mlp1OptionCostFunctionType -> costFunctionType,
				    DataTransformationOptionNormalizationType -> normalizationType
				]
			]
		]
	];

FitMlp1SeriesSC[

	(* Trains of a series of single or multiple 3-Layer True-Unit Mlp1(s).

	   Returns:
	   mlp1InfoList: {mlp1Info1, mlp1Info2, ...}
	   mlp1Info[[i]] corresponds to numberOfHiddenNeuronsList[[i]]
	   (see "Frequently used data structures") *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* List with numbers of hidden neurons *)
	numberOfHiddenNeuronsList_/;VectorQ[numberOfHiddenNeuronsList, IntegerQ],

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			i,
			crossoverProbability,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			learningParameterMin,
			learningParameterMax,
			maximumNumberOfIterations,
			minimizationPrecision,
			momentumParameter,
			multipleMlp1s,
			mutationProbability,
			numberOfIterationsToImprove,
			populationSize,
			randomValueInitialization,
			reportIteration,
			activationAndScaling,
			normalizationType,
			testSet,
			optimizationMethod,
			lambdaL2Regularization,
			costFunctionType
		},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
	    testSet = Mlp1OptionTestSet/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		reportIteration = Mlp1OptionReportIteration/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		Return[
			Table[
				FitMlp1[
					dataSet,
					numberOfHiddenNeuronsList[[i]],
					Mlp1OptionTestSet -> testSet,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
	    			Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp1OptionReportIteration -> reportIteration,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp1OptionActivationAndScaling -> activationAndScaling,
	    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp1OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				],
				
				{i, Length[numberOfHiddenNeuronsList]}
			]			
		]
	];
	
FitMlp1SeriesPC[

	(* Trains of a series of single or multiple 3-Layer True-Unit Mlp1(s).

	   Returns:
	   mlp1InfoList: {mlp1Info1, mlp1Info2, ...}
	   mlp1Info[[i]] corresponds to numberOfHiddenNeuronsList[[i]]
	   (see "Frequently used data structures") *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* List with numbers of hidden neurons *)
	numberOfHiddenNeuronsList_/;VectorQ[numberOfHiddenNeuronsList, IntegerQ],

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			i,
			crossoverProbability,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			learningParameterMin,
			learningParameterMax,
			maximumNumberOfIterations,
			minimizationPrecision,
			momentumParameter,
			multipleMlp1s,
			mutationProbability,
			numberOfIterationsToImprove,
			populationSize,
			randomValueInitialization,
			reportIteration,
			activationAndScaling,
			normalizationType,
			testSet,
			optimizationMethod,
			lambdaL2Regularization,
			costFunctionType
		},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
	    testSet = Mlp1OptionTestSet/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		reportIteration = Mlp1OptionReportIteration/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		
		ParallelNeeds[{"CIP`Mlp1`", "CIP`DataTransformation`", "CIP`Utility`", "Combinatorica`"}];
		DistributeDefinitions[testSet, multipleMlp1s, optimizationMethod, initialWeights, initialNetworks, 
			weightsValueLimit, minimizationPrecision, maximumNumberOfIterations, numberOfIterationsToImprove, 
			reportIteration, learningParameterMin, learningParameterMax, momentumParameter, populationSize, 
			crossoverProbability, mutationProbability, randomValueInitialization, activationAndScaling, normalizationType, 
			lambdaL2Regularization, costFunctionType];

		Return[
			ParallelTable[
				FitMlp1[
					dataSet,
					numberOfHiddenNeuronsList[[i]],
					Mlp1OptionTestSet -> testSet,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
   	 				Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
	 				Mlp1OptionReportIteration -> reportIteration,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
    				UtilityOptionRandomInitializationMode -> randomValueInitialization,
    				Mlp1OptionActivationAndScaling -> activationAndScaling,
    				Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
    				Mlp1OptionCostFunctionType -> costFunctionType,
    				DataTransformationOptionNormalizationType -> normalizationType
				],
			
				{i, Length[numberOfHiddenNeuronsList]}
			]			
		]
	];

FitMlp1WithBP[

	(* Training of 3-Layer True-Unit Mlp1 with standard backpropagation plus momentum.

	   Returns:
	   mlp1Info (see "Frequently used data structures") *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			dataSetScaleInfo,
			activationAndScaling,
			normalizationType,
			normalizationInfo,
			bestEpoch,
			bestMeanSquaredErrorOfTrainingSet,
			bestWeights,
			desiredOutputs,
			hiddenDelta,
			hiddenLastDelta,
			hiddenOutputs,
			hiddenWeights,
			i,
			initialWeights,
			internalMlp1OptionReportIteration,
			isReported,
			k,
			lastMlp1OptionReportIteration,
			learningParameter,
			learningParameterMax,
			learningParameterMin,
			maximumNumberOfIterations,
			meanSquaredError,
			momentumParameter,
			numberOfInputs,
			numberOfIterationsToImprove,
			numberOfOutputs,
			numberOfTestPairs,
			numberOfTrainingPairs,
			outputDelta,
			outputErrors,
			outputLastDelta,
			outputs,
			outputWeights,
			randomValueInitialization,
			reportIteration,
			scaledTestSet,
			scaledTrainingAndTestSet,
			scaledTrainingSet,
			testMeanSquaredErrorList,
			trainingInputs,
			trainingInputOutputPair,
			trainingList,
			trainingMeanSquaredErrorList,
			trueUnitHiddenOutputs,
			hiddenWeightsValueLimit,
			outputWeightsValueLimit
		},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		reportIteration = Mlp1OptionReportIteration/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		(* Set seed for random numbers if necessary *)
		If[randomValueInitialization == "Seed", SeedRandom[1], SeedRandom[]];

		(* Check activation function and outputs scaling *)
		If[activationAndScaling[[1, 2]] == "Sigmoid",
			If[activationAndScaling[[2, 2, 1]] < 0.0 || activationAndScaling[[2, 2, 2]] > 1.0,
				activationAndScaling[[2, 2]] = {0.1, 0.9}
			]			
		];

		dataSetScaleInfo = CIP`DataTransformation`GetDataSetScaleInfoForTrainingAndTestSet[trainingAndTestSet, activationAndScaling[[2, 1]], activationAndScaling[[2, 2]]];
		normalizationInfo = CIP`DataTransformation`GetDataSetNormalizationInfoForTrainingAndTestSet[trainingAndTestSet, normalizationType, dataSetScaleInfo];

    	(* Set training and test set *)
    	(* Set training and test set *)
    	scaledTrainingAndTestSet = CIP`DataTransformation`ScaleAndNormalizeTrainingAndTestSet[trainingAndTestSet, dataSetScaleInfo, normalizationInfo];
    	scaledTrainingSet = scaledTrainingAndTestSet[[1]];
    	scaledTestSet = scaledTrainingAndTestSet[[2]];
    
	    (* Initialization *)
	    If[reportIteration > maximumNumberOfIterations, reportIteration = maximumNumberOfIterations];
	    internalMlp1OptionReportIteration = reportIteration;
	    numberOfInputs = First[Dimensions[scaledTrainingSet[[1, 1]] ]];
	    numberOfOutputs = First[Dimensions[scaledTrainingSet[[1, 2]] ]];
	    bestWeights = {};
	    bestMeanSquaredErrorOfTrainingSet = Infinity;

		(* Y. Bengio, Practical Recommendations for Gradient-Based Training of Deep Architectures, https://arxiv.org/abs/1206.5533v2
		   
		   Wight initialization for sigmoid activation neurons:
		   hiddenWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons)];
		   outputWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfHiddenNeurons + 1 + numberOfOutputs)];
		   
		   Wight initialization for tanh activation neurons:
		   hiddenWeightsValueLimit = Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons)];
		   outputWeightsValueLimit = Sqrt[6.0/(numberOfHiddenNeurons + 1 + numberOfOutputs)];
		*)
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			hiddenWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons)],
			
			"Tanh",
			hiddenWeightsValueLimit = Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons)]
		];
		Switch[activationAndScaling[[1, 2]],
			
			"Sigmoid",
			outputWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfHiddenNeurons + 1 + numberOfOutputs)],
			
			"Tanh",
			outputWeightsValueLimit = Sqrt[6.0/(numberOfHiddenNeurons + 1 + numberOfOutputs)]
		];
    
		(* Initialization of hidden and output weights *)
	    If[Length[initialWeights] > 0,
      
			(* Use specified weights as initial weights *)
			hiddenWeights = initialWeights[[1]];
			outputWeights = initialWeights[[2]],
      
			(* Use random weights as initial weights *)
			(* 'True unit' : 'numberOfInputs + 1' and 'numberOfHiddenNeurons + 1' *)
			hiddenWeights = Table[RandomReal[{-hiddenWeightsValueLimit, hiddenWeightsValueLimit}, numberOfInputs + 1], {numberOfHiddenNeurons}];
			outputWeights = Table[RandomReal[{-outputWeightsValueLimit, outputWeightsValueLimit}, numberOfHiddenNeurons + 1], {numberOfOutputs}]
		];
    
		(* 'True unit' : 'numberOfInputs + 1' and 'numberOfHiddenNeurons + 1' *)
		hiddenLastDelta = Table[Table[0.0, {numberOfInputs + 1}], {numberOfHiddenNeurons}];
		outputLastDelta = Table[Table[0.0, {numberOfHiddenNeurons + 1}], {numberOfOutputs}];
    
	    (* Get length of training and test set *)
	    numberOfTrainingPairs = Length[scaledTrainingSet];
	    numberOfTestPairs = Length[scaledTestSet];
    
		(* Initialize training protocol *)
		trainingMeanSquaredErrorList = {{0, GetInternalMeanSquaredErrorOfMlp1[scaledTrainingSet, hiddenWeights, outputWeights, activationAndScaling]}};
		If[numberOfTestPairs > 0,
		
			(* Test set exists *)
			testMeanSquaredErrorList = {{0, GetInternalMeanSquaredErrorOfMlp1[scaledTestSet, hiddenWeights, outputWeights, activationAndScaling]}},
		
			(* No test set*)
			testMeanSquaredErrorList = {}
		];
    
		(* ----------------------------------------------------------------------------------------------------
		   Training
		   ---------------------------------------------------------------------------------------------------- *)
    	(* Main training loop over all epochs *)
		Do[
			trainingList = Combinatorica`RandomPermutation[numberOfTrainingPairs];
			If[learningParameterMax == learningParameterMin,
        
				learningParameter = learningParameterMin,
        
				learningParameter = learningParameterMax - i*(learningParameterMax - learningParameterMin)/maximumNumberOfIterations
			];
    		(* One epoch loop over all training pairs *)
			Do[
        
				(* Select training input/output pair *)
		        trainingInputOutputPair = scaledTrainingSet[[ trainingList[[k]] ]];
		        (* Add 'true unit' to training inputs *)
		        trainingInputs = Append[trainingInputOutputPair[[1]], 1.0];
		        desiredOutputs = trainingInputOutputPair[[2]];

		        (* Forward pass *)
				Switch[activationAndScaling[[1, 1]],
					
					"Sigmoid",
					hiddenOutputs = SigmoidFunction[hiddenWeights.trainingInputs],
					
					"Tanh",
					hiddenOutputs = Tanh[hiddenWeights.trainingInputs]
				];

		        (* Add 'true unit' to hidden outputs *)
		        trueUnitHiddenOutputs = Append[hiddenOutputs, 1.0];

				Switch[activationAndScaling[[1, 2]],
					
					"Sigmoid",
					outputs = SigmoidFunction[outputWeights.trueUnitHiddenOutputs],
					
					"Tanh",
					outputs = Tanh[outputWeights.trueUnitHiddenOutputs]
				];
        
		        (* Determine errors and deltas for weight update *)
		        outputErrors = desiredOutputs - outputs;
		        outputDelta = outputErrors*(outputs*(1 - outputs));
		        hiddenDelta = (trueUnitHiddenOutputs*(1 - trueUnitHiddenOutputs))*Transpose[outputWeights].outputDelta;
        
		        (* Update weights *)
		        outputLastDelta = learningParameter*Outer[Times, outputDelta, trueUnitHiddenOutputs] + momentumParameter*outputLastDelta;
		        outputWeights += outputLastDelta;
		        hiddenLastDelta = learningParameter*Drop[Outer[Times, hiddenDelta, trainingInputs], -1] + momentumParameter*hiddenLastDelta;
		        hiddenWeights += hiddenLastDelta,
        
				{k, numberOfTrainingPairs}
        
			]; (* End 'Do' {k, numberOfTrainingPairs} *)
      
			(* Calculate mean squared error of training set of epoch *)
			meanSquaredError = GetInternalMeanSquaredErrorOfMlp1[scaledTrainingSet, hiddenWeights, outputWeights, activationAndScaling];
      
			(* Compare to best mean squared error value and save *)
			If[meanSquaredError < bestMeanSquaredErrorOfTrainingSet,
				bestEpoch = i;
				bestMeanSquaredErrorOfTrainingSet = meanSquaredError;
				bestWeights = {hiddenWeights, outputWeights}
			];
      
			(* Report training *)
			If[internalMlp1OptionReportIteration == i,
		
				(* Report taining of this epoch *)		
				AppendTo[trainingMeanSquaredErrorList, {i, bestMeanSquaredErrorOfTrainingSet}];
				If[numberOfTestPairs > 0,
					AppendTo[testMeanSquaredErrorList, {i, GetInternalMeanSquaredErrorOfMlp1[scaledTestSet, bestWeights[[1]], bestWeights[[2]], activationAndScaling]}]
				];
				internalMlp1OptionReportIteration += reportIteration;
				isReported = True,
				
				(* Do NOT report taining of this epoch *)		
				isReported = False
			];
      
  			(* Check termination condition because of numberOfIterationsToImprove *)
			If[i > (bestEpoch + numberOfIterationsToImprove),
				lastMlp1OptionReportIteration = i;
				Break[]
			];

			(* Set lastMlp1OptionReportIteration at end of loop *)
			If[i == maximumNumberOfIterations,
				lastMlp1OptionReportIteration = i
			],
      
			{i, maximumNumberOfIterations}
		]; (* End 'Do' {i, maximumNumberOfIterations} *)
		
		(* ----------------------------------------------------------------------------------------------------
		   Set results
		   ---------------------------------------------------------------------------------------------------- *)
		If[!isReported,
			AppendTo[trainingMeanSquaredErrorList, {lastMlp1OptionReportIteration, bestMeanSquaredErrorOfTrainingSet}];
			If[numberOfTestPairs > 0,
				AppendTo[testMeanSquaredErrorList, {lastMlp1OptionReportIteration, GetInternalMeanSquaredErrorOfMlp1[scaledTestSet, bestWeights[[1]], bestWeights[[2]], activationAndScaling]}]
			];
		];
		
		(* Return mlp1Info *)
		Return[
			{
				{bestWeights},
				dataSetScaleInfo,
				{{trainingMeanSquaredErrorList, testMeanSquaredErrorList}},
				normalizationInfo,
				activationAndScaling,
				"BackpropagationPlusMomentum"
			}
		]		
	];

FitMlp1WithFindMinimum[

	(* Training of mlp1 with FindMinimum and "ConjugateGradient" method.
	
	   Returns:
	   mlp1Info (see "Frequently used data structures") *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,

	(* Options *)
	opts___

	] :=
  
	Module[
    
		{
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			normalizationType,
			hiddenWeights,
			hiddenWeightsVariables,
			i,
			initialWeights,
			inputs,
			j,
			k,
			lastTrainingStep,
			maximumNumberOfIterations,
			costFunction,
			minimizationPrecision,
			minimizationStep,
			minimumInfo,
			numberOfInputs,
			numberOfIOPairs,
			numberOfOutputs,
			outputs,
			outputWeights,
			outputWeightsVariables,
			mlp1Outputs,
			randomValueInitialization,
			reportIteration,
			reportIterationCounter,
			scaledTrainingAndTestSet,
			scaledTrainingSet,
			scaledTestSet,
			startVariables,
			stepNumber,
			steps,
			testMeanSquaredErrorList,
			trainingMeanSquaredErrorList,
			wHiddenToOutput,
			wInputToHidden,
			weightsRules,
			weightsVariables,
			weights,
			hiddenWeightsValueLimit,
			outputWeightsValueLimit,
			intermediateResult,
			weightsVariablesWithoutTrueUnitBias,
			lambdaL2Regularization,
			costFunctionType
		},


		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
	    minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
	    maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
	    reportIteration = Mlp1OptionReportIteration/.{opts}/.Options[Mlp1OptionsOptimization];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		(* Set seed for random numbers if necessary *)
		If[randomValueInitialization == "Seed", SeedRandom[1], SeedRandom[]];

		(* Check costFunction and outputs scaling *)
		If[costFunctionType == "Cross-Entropy",
			If[activationAndScaling[[2, 2, 1]] < 0.0 || activationAndScaling[[2, 2, 2]] > 1.0,
				activationAndScaling[[2, 2]] = {0.1, 0.9}
			]			
		];

		(* Check activation function and outputs scaling *)
		If[activationAndScaling[[1, 2]] == "Sigmoid",
			If[activationAndScaling[[2, 2, 1]] < 0.0 || activationAndScaling[[2, 2, 2]] > 1.0,
				activationAndScaling[[2, 2]] = {0.1, 0.9}
			]			
		];

		dataSetScaleInfo = CIP`DataTransformation`GetDataSetScaleInfoForTrainingAndTestSet[trainingAndTestSet, activationAndScaling[[2, 1]], activationAndScaling[[2, 2]]];
		normalizationInfo = CIP`DataTransformation`GetDataSetNormalizationInfoForTrainingAndTestSet[trainingAndTestSet, normalizationType, dataSetScaleInfo];
    
    	(* Set training and test set and related variables *)
    	scaledTrainingAndTestSet = CIP`DataTransformation`ScaleAndNormalizeTrainingAndTestSet[trainingAndTestSet, dataSetScaleInfo, normalizationInfo];
    	scaledTrainingSet = scaledTrainingAndTestSet[[1]];
    	numberOfIOPairs = Length[scaledTrainingSet];
    	inputs = CIP`Utility`GetInputsOfDataSet[scaledTrainingSet];
    	outputs = CIP`Utility`GetOutputsOfDataSet[scaledTrainingSet];
    	scaledTestSet = scaledTrainingAndTestSet[[2]];
    
	    (* Network structure *)
	    numberOfInputs = First[Dimensions[scaledTrainingSet[[1, 1]]]];
	    numberOfOutputs = First[Dimensions[scaledTrainingSet[[1, 2]]]];

		(* Y. Bengio, Practical Recommendations for Gradient-Based Training of Deep Architectures, https://arxiv.org/abs/1206.5533v2
		   
		   Wight initialization for sigmoid activation neurons:
		   hiddenWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons)];
		   outputWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfHiddenNeurons + 1 + numberOfOutputs)];
		   
		   Wight initialization for tanh activation neurons:
		   hiddenWeightsValueLimit = Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons)];
		   outputWeightsValueLimit = Sqrt[6.0/(numberOfHiddenNeurons + 1 + numberOfOutputs)];
		*)
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			hiddenWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons)],
			
			"Tanh",
			hiddenWeightsValueLimit = Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons)]
		];
		Switch[activationAndScaling[[1, 2]],
			
			"Sigmoid",
			outputWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfHiddenNeurons + 1 + numberOfOutputs)],
			
			"Tanh",
			outputWeightsValueLimit = Sqrt[6.0/(numberOfHiddenNeurons + 1 + numberOfOutputs)]
		];

	    (* Initialize hidden and output weights *)
	    If[Length[initialWeights] > 0,
      
			(* Use specified weights as initial weights *)
			hiddenWeights = initialWeights[[1]];
			outputWeights = initialWeights[[2]],
      
			(* Use random weights as initial weights *)
			(* 'True unit' : 'numberOfInputs + 1' and 'numberOfHiddenNeurons + 1' *)
			hiddenWeights = Table[RandomReal[{-hiddenWeightsValueLimit, hiddenWeightsValueLimit}, numberOfInputs + 1], {numberOfHiddenNeurons}];
			outputWeights = Table[RandomReal[{-outputWeightsValueLimit, outputWeightsValueLimit}, numberOfHiddenNeurons + 1], {numberOfOutputs}]
		];

		(* Initialize training protocol *)
		trainingMeanSquaredErrorList = {{0, GetInternalMeanSquaredErrorOfMlp1[scaledTrainingSet, hiddenWeights, outputWeights, activationAndScaling]}};
		If[Length[scaledTestSet] > 0,
		
			(* Test set exists *)
			testMeanSquaredErrorList = {{0, GetInternalMeanSquaredErrorOfMlp1[scaledTestSet, hiddenWeights, outputWeights, activationAndScaling]}},
		
			(* No test set*)
			testMeanSquaredErrorList = {}
		];

		(* ----------------------------------------------------------------------------------------------------
		   Definition of start variables
		   ---------------------------------------------------------------------------------------------------- *)
		startVariables = GetWeightsStartVariables[numberOfInputs, numberOfHiddenNeurons, numberOfOutputs, wInputToHidden, wHiddenToOutput, hiddenWeights, outputWeights];

		(* ----------------------------------------------------------------------------------------------------
		   Mean squared error function to minimize
		   ---------------------------------------------------------------------------------------------------- *)
	    (* Map: Add 'true unit' *)
	    weightsVariables = GetWeightsVariables[numberOfInputs, numberOfHiddenNeurons, numberOfOutputs, wInputToHidden, wHiddenToOutput];
	    hiddenWeightsVariables = weightsVariables[[1]];
	    outputWeightsVariables = weightsVariables[[2]];

		weightsVariablesWithoutTrueUnitBias = GetWeightsVariablesWithoutTrueUnitBias[numberOfInputs, numberOfHiddenNeurons, numberOfOutputs, wInputToHidden, wHiddenToOutput];
	    
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			intermediateResult = SigmoidFunction[Map[Append[#, 1] &, inputs].Transpose[hiddenWeightsVariables]],
			
			"Tanh",
			intermediateResult = Tanh[Map[Append[#, 1] &, inputs].Transpose[hiddenWeightsVariables]]
		];

		Switch[activationAndScaling[[1, 2]],
			
			"Sigmoid",
			mlp1Outputs = SigmoidFunction[Map[Append[#, 1] &, intermediateResult].Transpose[outputWeightsVariables]],
			
			"Tanh",
			Switch[costFunctionType,
			
				"SquaredError",				
				mlp1Outputs = Tanh[Map[Append[#, 1] &, intermediateResult].Transpose[outputWeightsVariables]],
				
				(* Cross-entropy cost function arguments MUST be in interval {0, 1} *)
				"Cross-Entropy",
				mlp1Outputs = 0.5 * (Tanh[Map[Append[#, 1] &, intermediateResult].Transpose[outputWeightsVariables]] + 1.0)
			]
		];
	    
	    Switch[costFunctionType,
	    	
	    	"SquaredError",
		    If[lambdaL2Regularization == 0.0,
	
				(* NO L2 regularization *)
				costFunction =
					Sum[
						Sum[
							(outputs[[i, k]] - mlp1Outputs[[i, k]])^2,
							
							{k, numberOfOutputs}
						],
							    
						{i, numberOfIOPairs}	
					]/numberOfIOPairs,
		    	
				(* L2 regularization *)
				costFunction =
					Sum[
						Sum[
							(outputs[[i, k]] - mlp1Outputs[[i, k]])^2,
							
							{k, numberOfOutputs}
						],
							    
						{i, numberOfIOPairs}	
					]/numberOfIOPairs
					+
					0.5 * lambdaL2Regularization/numberOfIOPairs * 
					Sum[
						weightsVariablesWithoutTrueUnitBias[[j]] * weightsVariablesWithoutTrueUnitBias[[j]],
						
						{j, Length[weightsVariablesWithoutTrueUnitBias]}
					]
		    ],
	    	
	    	"Cross-Entropy",
		    If[lambdaL2Regularization == 0.0,
	
				(* NO L2 regularization *)
				costFunction =
					Sum[
						Sum[
							outputs[[i, k]] * Log[mlp1Outputs[[i, k]]] + (1.0 - outputs[[i, k]]) * Log[1.0 - mlp1Outputs[[i, k]]],
							
							{k, numberOfOutputs}
						],
							    
						{i, numberOfIOPairs}	
					]/numberOfIOPairs,
		    	
				(* L2 regularization *)
				costFunction =
					Sum[
						Sum[
							outputs[[i, k]] * Log[mlp1Outputs[[i, k]]] + (1.0 - outputs[[i, k]]) * Log[1.0 - mlp1Outputs[[i, k]]],
							
							{k, numberOfOutputs}
						],
							    
						{i, numberOfIOPairs}	
					]/numberOfIOPairs
					+
					0.5 * lambdaL2Regularization/numberOfIOPairs * 
					Sum[
						weightsVariablesWithoutTrueUnitBias[[j]] * weightsVariablesWithoutTrueUnitBias[[j]],
						
						{j, Length[weightsVariablesWithoutTrueUnitBias]}
					]
		    ]
	    ];

		(* ----------------------------------------------------------------------------------------------------
		   Find minimum
		   ---------------------------------------------------------------------------------------------------- *)
		steps = 0;
		reportIterationCounter = 0;
		minimumInfo = 
			Reap[FindMinimum[
				costFunction, 
				startVariables, 
				Method -> "ConjugateGradient", 
				MaxIterations -> maximumNumberOfIterations, 
				WorkingPrecision -> MachinePrecision,
				AccuracyGoal -> minimizationPrecision, 
				PrecisionGoal -> minimizationPrecision, 
				StepMonitor :> 
					(
						steps++;
						reportIterationCounter++;
						If[reportIterationCounter == reportIteration, 
							reportIterationCounter = 0;
							Sow[{steps, weightsVariables}]
						]
					)
			]];
		weightsRules = minimumInfo[[1, 2]];

		(* ----------------------------------------------------------------------------------------------------
		   Set training protocol if necessary
		   ---------------------------------------------------------------------------------------------------- *)
		If[Length[minimumInfo[[2]]] > 0,
			Do[
				minimizationStep = minimumInfo[[2, 1, i]];
				stepNumber = minimizationStep[[1]];
				weights = minimizationStep[[2]];
				AppendTo[trainingMeanSquaredErrorList, {stepNumber, GetInternalMeanSquaredErrorOfMlp1[scaledTrainingSet, weights[[1]], weights[[2]], activationAndScaling]}];
				If[Length[scaledTestSet] > 0,
					(* Test set exists *)
					AppendTo[testMeanSquaredErrorList, {stepNumber, GetInternalMeanSquaredErrorOfMlp1[scaledTestSet, weights[[1]], weights[[2]], activationAndScaling]}]
				],
				
				{i, Length[minimumInfo[[2, 1]]]}
			]
		];
			
		(* ----------------------------------------------------------------------------------------------------
		   Set results
		   ---------------------------------------------------------------------------------------------------- *)
		weights = GetWeightsVariables[numberOfInputs, numberOfHiddenNeurons, numberOfOutputs, wInputToHidden, wHiddenToOutput]/.weightsRules;

		(* End of training protocol *)
		lastTrainingStep = Last[trainingMeanSquaredErrorList];
		If[lastTrainingStep[[1]] < steps,
			AppendTo[trainingMeanSquaredErrorList, {steps, GetInternalMeanSquaredErrorOfMlp1[scaledTrainingSet, weights[[1]], weights[[2]], activationAndScaling]}];
			If[Length[scaledTestSet] > 0,
				(* Test set exists *)
				AppendTo[testMeanSquaredErrorList, {steps, GetInternalMeanSquaredErrorOfMlp1[scaledTestSet, weights[[1]], weights[[2]], activationAndScaling]}]
			]
		];
		
		(* Return mlp1Info *)
		Return[
			{
				{weights},
				dataSetScaleInfo,
				{{trainingMeanSquaredErrorList, testMeanSquaredErrorList}},
				normalizationInfo,
				activationAndScaling,
				"FindMinimum"
			}
		]		
	];

FitMlp1WithNMinimize[

	(* Training of mlp1 with NMinimize and "DifferentialEvolution".
	
	   Returns:
	   mlp1Info (see "Frequently used data structures") *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,

	(* Options *)
	opts___

	] :=
  
	Module[
    
		{
			constraints,
			dataSetScaleInfo,
			normalizationInfo,
			flatWeightsVariables,
			activationAndScaling,
			normalizationType,
			hiddenWeightsVariables,
			i,
			inputs,
			j,
			k,
			lastTrainingStep,
			maximumNumberOfIterations,
			costFunction,
			minimizationPrecision,
			minimizationStep,
			minimumInfo,
			numberOfInputs,
			numberOfIOPairs,
			numberOfOutputs,
			outputs,
			outputWeightsVariables,
			mlp1Outputs,
			randomValueInitialization,
			reportIteration,
			reportIterationCounter,
			scaledTrainingAndTestSet,
			scaledTrainingSet,
			scaledTestSet,
			stepNumber,
			steps,
			testMeanSquaredErrorList,
			trainingMeanSquaredErrorList,
			wHiddenToOutput,
			wInputToHidden,
			weightsRules,
			weightsValueLimit,
			weightsVariables,
			weights,
			weightsVariablesWithoutTrueUnitBias,
			lambdaL2Regularization,
			costFunctionType
		},


		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
	    minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
	    maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
	    reportIteration = Mlp1OptionReportIteration/.{opts}/.Options[Mlp1OptionsOptimization];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		(* Set seed for random numbers if necessary *)
		If[randomValueInitialization == "Seed", SeedRandom[1], SeedRandom[]];

		(* Check costFunction and outputs scaling *)
		If[costFunctionType == "Cross-Entropy",
			If[activationAndScaling[[2, 2, 1]] < 0.0 || activationAndScaling[[2, 2, 2]] > 1.0,
				activationAndScaling[[2, 2]] = {0.1, 0.9}
			]			
		];

		(* Check activation function and outputs scaling *)
		If[activationAndScaling[[1, 2]] == "Sigmoid",
			If[activationAndScaling[[2, 2, 1]] < 0.0 || activationAndScaling[[2, 2, 2]] > 1.0,
				activationAndScaling[[2, 2]] = {0.1, 0.9}
			]			
		];

		dataSetScaleInfo = CIP`DataTransformation`GetDataSetScaleInfoForTrainingAndTestSet[trainingAndTestSet, activationAndScaling[[2, 1]], activationAndScaling[[2, 2]]];
		normalizationInfo = CIP`DataTransformation`GetDataSetNormalizationInfoForTrainingAndTestSet[trainingAndTestSet, normalizationType, dataSetScaleInfo];
    
    	(* Set training and test set and related variables *)
    	scaledTrainingAndTestSet = CIP`DataTransformation`ScaleAndNormalizeTrainingAndTestSet[trainingAndTestSet, dataSetScaleInfo, normalizationInfo];
    	scaledTrainingSet = scaledTrainingAndTestSet[[1]];
    	numberOfIOPairs = Length[scaledTrainingSet];
    	inputs = CIP`Utility`GetInputsOfDataSet[scaledTrainingSet];
    	outputs = CIP`Utility`GetOutputsOfDataSet[scaledTrainingSet];
    	scaledTestSet = scaledTrainingAndTestSet[[2]];
    
	    (* Network structure *)
	    numberOfInputs = First[Dimensions[scaledTrainingSet[[1, 1]]]];
	    numberOfOutputs = First[Dimensions[scaledTrainingSet[[1, 2]]]];

		(* Initialize training protocol *)
		trainingMeanSquaredErrorList = {};
		testMeanSquaredErrorList = {};

		(* ----------------------------------------------------------------------------------------------------
		   Mean squared error function to minimize
		   ---------------------------------------------------------------------------------------------------- *)
	    (* Map: Add 'true unit' *)
	    weightsVariables = GetWeightsVariables[numberOfInputs, numberOfHiddenNeurons, numberOfOutputs, wInputToHidden, wHiddenToOutput];
	    hiddenWeightsVariables = weightsVariables[[1]];
	    outputWeightsVariables = weightsVariables[[2]];

		weightsVariablesWithoutTrueUnitBias = GetWeightsVariablesWithoutTrueUnitBias[numberOfInputs, numberOfHiddenNeurons, numberOfOutputs, wInputToHidden, wHiddenToOutput];

		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			intermediateResult = SigmoidFunction[Map[Append[#, 1] &, inputs].Transpose[hiddenWeightsVariables]],
			
			"Tanh",
			intermediateResult = Tanh[Map[Append[#, 1] &, inputs].Transpose[hiddenWeightsVariables]]
		];

		Switch[activationAndScaling[[1, 2]],
			
			"Sigmoid",
			mlp1Outputs = SigmoidFunction[Map[Append[#, 1] &, intermediateResult].Transpose[outputWeightsVariables]],
			
			"Tanh",
			Switch[costFunctionType,
			
				"SquaredError",				
				mlp1Outputs = Tanh[Map[Append[#, 1] &, intermediateResult].Transpose[outputWeightsVariables]],
				
				(* Cross-entropy cost function arguments MUST be in interval {0, 1} *)
				"Cross-Entropy",
				mlp1Outputs = 0.5 * (Tanh[Map[Append[#, 1] &, intermediateResult].Transpose[outputWeightsVariables]] + 1.0)
			]
		];
	    
	    Switch[costFunctionType,
	    	
	    	"SquaredError",
		    If[lambdaL2Regularization == 0.0,
	
				(* NO L2 regularization *)
				costFunction =
					Sum[
						Sum[
							(outputs[[i, k]] - mlp1Outputs[[i, k]])^2,
							
							{k, numberOfOutputs}
						],
							    
						{i, numberOfIOPairs}	
					]/numberOfIOPairs,
		    	
				(* L2 regularization *)
				costFunction =
					Sum[
						Sum[
							(outputs[[i, k]] - mlp1Outputs[[i, k]])^2,
							
							{k, numberOfOutputs}
						],
							    
						{i, numberOfIOPairs}	
					]/numberOfIOPairs
					+
					0.5 * lambdaL2Regularization/numberOfIOPairs * 
					Sum[
						weightsVariablesWithoutTrueUnitBias[[j]] * weightsVariablesWithoutTrueUnitBias[[j]],
						
						{j, Length[weightsVariablesWithoutTrueUnitBias]}
					]
		    ],
	    	
	    	"Cross-Entropy",
		    If[lambdaL2Regularization == 0.0,
	
				(* NO L2 regularization *)
				costFunction =
					Sum[
						Sum[
							outputs[[i, k]] * Log[mlp1Outputs[[i, k]]] + (1.0 - outputs[[i, k]]) * Log[1.0 - mlp1Outputs[[i, k]]],
							
							{k, numberOfOutputs}
						],
							    
						{i, numberOfIOPairs}	
					]/numberOfIOPairs,
		    	
				(* L2 regularization *)
				costFunction =
					Sum[
						Sum[
							outputs[[i, k]] * Log[mlp1Outputs[[i, k]]] + (1.0 - outputs[[i, k]]) * Log[1.0 - mlp1Outputs[[i, k]]],
							
							{k, numberOfOutputs}
						],
							    
						{i, numberOfIOPairs}	
					]/numberOfIOPairs
					+
					0.5 * lambdaL2Regularization/numberOfIOPairs * 
					Sum[
						weightsVariablesWithoutTrueUnitBias[[j]] * weightsVariablesWithoutTrueUnitBias[[j]],
						
						{j, Length[weightsVariablesWithoutTrueUnitBias]}
					]
		    ]
	    ];

		(* ----------------------------------------------------------------------------------------------------
		   Set constraints for weights
		   ---------------------------------------------------------------------------------------------------- *)
		flatWeightsVariables = Flatten[weightsVariables];
		constraints = 
			Apply[And, 
				Table[
					-weightsValueLimit <= flatWeightsVariables[[i]] <= weightsValueLimit, 
					
					{i, Length[flatWeightsVariables]}
				]
			];

		(* ----------------------------------------------------------------------------------------------------
		   Find minimum
		   ---------------------------------------------------------------------------------------------------- *)
		steps = 0;
		reportIterationCounter = 0;
		minimumInfo = 
			Reap[NMinimize[
				{costFunction, constraints}, 
				flatWeightsVariables, 
				Method -> "DifferentialEvolution", 
				MaxIterations -> maximumNumberOfIterations, 
				WorkingPrecision -> MachinePrecision,
				AccuracyGoal -> minimizationPrecision, 
				PrecisionGoal -> minimizationPrecision, 
				StepMonitor :> 
					(
						steps++;
						reportIterationCounter++;
						If[reportIterationCounter == reportIteration, 
							reportIterationCounter = 0;
							Sow[{steps, weightsVariables}]
						]
					)
			]];
		weightsRules = minimumInfo[[1, 2]];

		(* ----------------------------------------------------------------------------------------------------
		   Set training protocol if necessary
		   ---------------------------------------------------------------------------------------------------- *)
		If[Length[minimumInfo[[2]]] > 0,
			Do[
				minimizationStep = minimumInfo[[2, 1, i]];
				stepNumber = minimizationStep[[1]];
				weights = minimizationStep[[2]];
				AppendTo[trainingMeanSquaredErrorList, {stepNumber, GetInternalMeanSquaredErrorOfMlp1[scaledTrainingSet, weights[[1]], weights[[2]], activationAndScaling]}];
				If[Length[scaledTestSet] > 0,
					(* Test set exists *)
					AppendTo[testMeanSquaredErrorList, {stepNumber, GetInternalMeanSquaredErrorOfMlp1[scaledTestSet, weights[[1]], weights[[2]], activationAndScaling]}]
				],
				
				{i, Length[minimumInfo[[2, 1]]]}
			]
		];
			
		(* ----------------------------------------------------------------------------------------------------
		   Set results
		   ---------------------------------------------------------------------------------------------------- *)
		weights = GetWeightsVariables[numberOfInputs, numberOfHiddenNeurons, numberOfOutputs, wInputToHidden, wHiddenToOutput]/.weightsRules;

		(* End of training protocol *)
		lastTrainingStep = Last[trainingMeanSquaredErrorList];
		If[lastTrainingStep[[1]] < steps,
			AppendTo[trainingMeanSquaredErrorList, {steps, GetInternalMeanSquaredErrorOfMlp1[scaledTrainingSet, weights[[1]], weights[[2]], activationAndScaling]}];
			If[Length[scaledTestSet] > 0,
				(* Test set exists *)
				AppendTo[testMeanSquaredErrorList, {steps, GetInternalMeanSquaredErrorOfMlp1[scaledTestSet, weights[[1]], weights[[2]], activationAndScaling]}]
			]
		];
		
		(* Return mlp1Info *)
		Return[
			{
				{weights},
				dataSetScaleInfo,
				{{trainingMeanSquaredErrorList, testMeanSquaredErrorList}},
				normalizationInfo,
				activationAndScaling,
				"NMinimize"
			}
		]		
	];

FitMlp1WithGA[

	(* Training of 3-Layer True-Unit mlp1 with Genetic Algorithm.

	   Returns:
	   mlp1Info (see "Frequently used data structures") *)

	
	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			normalizationType,
			bestGeneration,
			bestMeanSquaredErrorOfTrainingSet,
			crossoverProbability,
			fitnessList,
			fittestChromosome,
			i,
			initialWeights,
			weightsValueLimit,
			internalMlp1OptionReportIteration,
			isReported,
			k,
			lastMlp1OptionReportIteration,
			maximumNumberOfIterations,
			meanSquaredError,
			mutationProbability,
			numberOfInputs,
			numberOfIterationsToImprove,
			numberOfOutputs,
			numberOfTestPairs,
			population,
			populationSize,
			randomValueInitialization,
			reportIteration,
			scaledFitnessSum,
			scaledTestSet,
			scaledTrainingAndTestSet,
			scaledTrainingSet,
			testMeanSquaredErrorList,
			trainingMeanSquaredErrorList,
			hiddenWeightsValueLimit,
			outputWeightsValueLimit
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		reportIteration = Mlp1OptionReportIteration/.{opts}/.Options[Mlp1OptionsOptimization];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		(* Set seed for random numbers if necessary *)
		If[randomValueInitialization == "Seed", SeedRandom[1], SeedRandom[]];

		(* Check activation function and outputs scaling *)
		If[activationAndScaling[[1, 2]] == "Sigmoid",
			If[activationAndScaling[[2, 2, 1]] < 0.0 || activationAndScaling[[2, 2, 2]] > 1.0,
				activationAndScaling[[2, 2]] = {0.1, 0.9}
			]			
		];

		dataSetScaleInfo = CIP`DataTransformation`GetDataSetScaleInfoForTrainingAndTestSet[trainingAndTestSet, activationAndScaling[[2, 1]], activationAndScaling[[2, 2]]];
		normalizationInfo = CIP`DataTransformation`GetDataSetNormalizationInfoForTrainingAndTestSet[trainingAndTestSet, normalizationType, dataSetScaleInfo];
    
    	(* Set training and test set *)
    	scaledTrainingAndTestSet = CIP`DataTransformation`ScaleAndNormalizeTrainingAndTestSet[trainingAndTestSet, dataSetScaleInfo, normalizationInfo];
    	scaledTrainingSet = scaledTrainingAndTestSet[[1]];
    	scaledTestSet = scaledTrainingAndTestSet[[2]];
    
	    (* Initialization *)
	    If[reportIteration > maximumNumberOfIterations, reportIteration = maximumNumberOfIterations];
	    internalMlp1OptionReportIteration = reportIteration;
	    numberOfInputs = First[Dimensions[scaledTrainingSet[[1, 1]] ]];
	    numberOfOutputs = First[Dimensions[scaledTrainingSet[[1, 2]] ]];
		trainingMeanSquaredErrorList = {};
		testMeanSquaredErrorList = {};
	    bestMeanSquaredErrorOfTrainingSet = Infinity;

		(* Y. Bengio, Practical Recommendations for Gradient-Based Training of Deep Architectures, https://arxiv.org/abs/1206.5533v2
		   
		   Wight initialization for sigmoid activation neurons:
		   hiddenWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons)];
		   outputWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfHiddenNeurons + 1 + numberOfOutputs)];
		   
		   Wight initialization for tanh activation neurons:
		   hiddenWeightsValueLimit = Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons)];
		   outputWeightsValueLimit = Sqrt[6.0/(numberOfHiddenNeurons + 1 + numberOfOutputs)];
		*)
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			hiddenWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons)],
			
			"Tanh",
			hiddenWeightsValueLimit = Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons)]
		];
		Switch[activationAndScaling[[1, 2]],
			
			"Sigmoid",
			outputWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfHiddenNeurons + 1 + numberOfOutputs)],
			
			"Tanh",
			outputWeightsValueLimit = Sqrt[6.0/(numberOfHiddenNeurons + 1 + numberOfOutputs)]
		];

		(* Create initial population of form {{{hiddenWeights}, {outputWeights}}, ...} *)
		population = 
			Table[
				{
					Table[RandomReal[{-hiddenWeightsValueLimit, hiddenWeightsValueLimit}, numberOfInputs + 1], {numberOfHiddenNeurons}],
					Table[RandomReal[{-outputWeightsValueLimit, outputWeightsValueLimit}, numberOfHiddenNeurons + 1], {numberOfOutputs}]
				},
        
				{populationSize}
			];
   
		(* Check if there are weights to be improved *)
		If[Length[initialWeights] > 0,
			(* Add weights to be improved to population *)
			AppendTo[population, initialWeights]
		];
    
		(* Get length of test set *)
		numberOfTestPairs = Length[scaledTestSet];
    
		(* ----------------------------------------------------------------------------------------------------
		   Training
		   ---------------------------------------------------------------------------------------------------- *)
		Do[
      
			(* Evaluate fitness of population *)
			fitnessList = 
				Table[
					1.0/GetInternalMeanSquaredErrorOfMlp1[scaledTrainingSet, population[[k, 1]], population[[k, 2]], activationAndScaling],
          
					{k, Length[population]}
				];
			scaledFitnessSum = CIP`Utility`GetScaledFitnessSumList[fitnessList];
      
			(* Determine fittest chromosome of form {{hiddenWeights}, {outputWeights}} *)
			fittestChromosome = population[[ First[Flatten[Position[fitnessList, Max[fitnessList], {1}, 1]]] ]];
      
			(* Calculate mean squared error of training set of generation *)
			meanSquaredError = GetInternalMeanSquaredErrorOfMlp1[scaledTrainingSet, fittestChromosome[[1]], fittestChromosome[[2]], activationAndScaling];
      
			(* Compare to best mean squared error value and save *)
			If[meanSquaredError < bestMeanSquaredErrorOfTrainingSet,
				bestGeneration = i;
				bestMeanSquaredErrorOfTrainingSet = meanSquaredError
			];
      
			(* Report training *)
			If[i == 1 || internalMlp1OptionReportIteration == i,
				
				(* Report taining of this generation *)		
				AppendTo[trainingMeanSquaredErrorList, {i, bestMeanSquaredErrorOfTrainingSet}];
				If[numberOfTestPairs > 0,
					AppendTo[testMeanSquaredErrorList, {i, GetInternalMeanSquaredErrorOfMlp1[scaledTestSet, fittestChromosome[[1]], fittestChromosome[[2]], activationAndScaling]}]
				];
				If[internalMlp1OptionReportIteration == i, 
					internalMlp1OptionReportIteration += reportIteration
				];
				isReported = True,
				
				(* Do NOT report taining of this generation *)		
				isReported = False
			];
      
			(* Check termination condition because of numberOfIterationsToImprove or end of loop *)
			If[i > (bestGeneration + numberOfIterationsToImprove) || i == maximumNumberOfIterations,
				lastMlp1OptionReportIteration = i;
				Break[]
			];

			(* Create new population: Roulette-wheel selection and crossover + mutation *)
			population = 
				Flatten[
					Table[
						CrossoverChromosomes[
							SelectChromosome[population, scaledFitnessSum], 
							SelectChromosome[population, scaledFitnessSum], 
							crossoverProbability, 
							mutationProbability, 
							weightsValueLimit
						],
            
						{CIP`Utility`GetNextHigherEvenIntegerNumber[populationSize]/2}
					], 1
				];
      
			(* Elitism : Add fittest chromosome of last epoch/generation *)
			AppendTo[population, fittestChromosome],
      
			{i, maximumNumberOfIterations}
      
		]; (* End 'Do' {i, maximumNumberOfIterations} *)

		(* ----------------------------------------------------------------------------------------------------
		   Results
		   ---------------------------------------------------------------------------------------------------- *)
		(* Report training if necessary *)
		If[!isReported,
			AppendTo[trainingMeanSquaredErrorList, {lastMlp1OptionReportIteration, bestMeanSquaredErrorOfTrainingSet}];
			If[numberOfTestPairs > 0,
				AppendTo[testMeanSquaredErrorList, {lastMlp1OptionReportIteration, GetInternalMeanSquaredErrorOfMlp1[scaledTestSet, fittestChromosome[[1]], fittestChromosome[[2]], activationAndScaling]}]
			];
		];
    
		(* Return mlp1Info *)
		Return[
			{
				{fittestChromosome},
				dataSetScaleInfo,
				{{trainingMeanSquaredErrorList, testMeanSquaredErrorList}},
				normalizationInfo,
				activationAndScaling,
				"GeneticAlgorithm"
			}
		]
	];

FitSingleMlp1[

	(* Training of single 3-Layer True-Unit Mlp1.

	   Returns:
	   mlp1Info (see "Frequently used data structures") *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			crossoverProbability,
			initialWeights,
			weightsValueLimit,
			learningParameterMin,
			learningParameterMax,
			maximumNumberOfIterations,
			minimizationPrecision,
			momentumParameter,
			mutationProbability,
			numberOfIterationsToImprove,
			populationSize,
			randomValueInitialization,
			reportIteration,
			activationAndScaling,
			normalizationType,
			optimizationMethod,
			lambdaL2Regularization,
			costFunctionType
		},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		reportIteration = Mlp1OptionReportIteration/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		
		(* ----------------------------------------------------------------------------------------------------
		   Switch training method
		   ---------------------------------------------------------------------------------------------------- *)
		Switch[optimizationMethod,
			
			"FindMinimum",
			Return[
				FitMlp1WithFindMinimum[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					Mlp1OptionInitialWeights -> initialWeights,
	    			Mlp1OptionMinimizationPrecision -> minimizationPrecision,
	    			Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionReportIteration -> reportIteration,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp1OptionActivationAndScaling -> activationAndScaling,
	    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp1OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				]
			],
			
			"NMinimize",
			Return[
				FitMlp1WithNMinimize[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
	    			Mlp1OptionMinimizationPrecision -> minimizationPrecision,
	    			Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionReportIteration -> reportIteration,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp1OptionActivationAndScaling -> activationAndScaling,
	    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp1OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				]
			],
			
			"BackpropagationPlusMomentum",
			Return[
				FitMlp1WithBP[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
					Mlp1OptionReportIteration -> reportIteration,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp1OptionActivationAndScaling -> activationAndScaling,
	    			DataTransformationOptionNormalizationType -> normalizationType
				]
			],
			
			"GeneticAlgorithm",
			Return[
				FitMlp1WithGA[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
					Mlp1OptionReportIteration -> reportIteration,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp1OptionActivationAndScaling -> activationAndScaling,
	    			DataTransformationOptionNormalizationType -> normalizationType
				]
			]
		]
	];

GetBestMlp1ClassOptimization[

	(* Returns best training set optimization result of mlp1 for classification.

	   Returns: 
	   Best index for classification *)


	(* mlp1TrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlp1InfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlp1InfoList: List with mlp1Info
	   mlp1InfoList[[i]] refers to optimization step i *)
	mlp1TrainOptimization_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			bestOptimization,
			parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Utility options *)
	    bestOptimization = UtilityOptionBestOptimization/.{opts}/.Options[UtilityOptionsOptimization];
	    (* Parallelization options *)
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				GetBestMlp1ClassOptimizationPC[
					mlp1TrainOptimization,
					UtilityOptionBestOptimization -> bestOptimization
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetBestMlp1ClassOptimizationSC[
					mlp1TrainOptimization,
					UtilityOptionBestOptimization -> bestOptimization
				]
			]
		]
	];

GetBestMlp1ClassOptimizationSC[

	(* Returns best training set optimization result of mlp1 for classification.

	   Returns: 
	   Best index for classification *)


	(* mlp1TrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlp1InfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlp1InfoList: List with mlp1Info
	   mlp1InfoList[[i]] refers to optimization step i *)
	mlp1TrainOptimization_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			k,
			trainingAndTestSetList,
			mlp1InfoList,
			maximumCorrectClassificationInPercent,
			mlp1Info,
			correctClassificationInPercent,
			bestIndex,
			testSet,
			trainingSet,
			trainingSetCorrectClassificationInPercent,
			testSetCorrectClassificationInPercent,
			bestOptimization,
			bestTestSetCorrectClassificationInPercent,
			minimumDeviation,
			deviation 
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Utility options *)
	    bestOptimization = UtilityOptionBestOptimization/.{opts}/.Options[UtilityOptionsOptimization];

		Switch[bestOptimization,

			(* ------------------------------------------------------------------------------- *)
			"BestTestResult",			
			trainingAndTestSetList = mlp1TrainOptimization[[3]];
			mlp1InfoList = mlp1TrainOptimization[[4]];
			maximumCorrectClassificationInPercent = -1.0;
			Do[
				testSet = trainingAndTestSetList[[k, 2]];
				mlp1Info = mlp1InfoList[[k]];
				correctClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[testSet, mlp1Info];
				If[correctClassificationInPercent > maximumCorrectClassificationInPercent,
					maximumCorrectClassificationInPercent = correctClassificationInPercent;
					bestIndex = k
				],
				
				{k, Length[mlp1InfoList]}
			],
			
			(* ------------------------------------------------------------------------------- *)			
			"MinimumDeviation",
			trainingAndTestSetList = mlp1TrainOptimization[[3]];
			mlp1InfoList = mlp1TrainOptimization[[4]];
			minimumDeviation = Infinity;
			Do[
				trainingSet = trainingAndTestSetList[[k, 1]];
				testSet = trainingAndTestSetList[[k, 2]];
				mlp1Info = mlp1InfoList[[k]];
				trainingSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[trainingSet, mlp1Info];
				testSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[testSet, mlp1Info];
				deviation = Abs[testSetCorrectClassificationInPercent - trainingSetCorrectClassificationInPercent];
				If[deviation < minimumDeviation || (deviation == minimumDeviation && testSetCorrectClassificationInPercent < bestTestSetCorrectClassificationInPercent),
					minimumDeviation = deviation;
					bestTestSetCorrectClassificationInPercent = testSetCorrectClassificationInPercent;
					bestIndex = k
				],
				
				{k, Length[mlp1InfoList]}
			]
		];

		Return[bestIndex]
	];

GetBestMlp1ClassOptimizationPC[

	(* Returns best training set optimization result of mlp1 for classification.

	   Returns: 
	   Best index for classification *)


	(* mlp1TrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlp1InfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlp1InfoList: List with mlp1Info
	   mlp1InfoList[[i]] refers to optimization step i *)
	mlp1TrainOptimization_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			k,
			trainingAndTestSetList,
			mlp1InfoList,
			maximumCorrectClassificationInPercent,
			mlp1Info,
			bestIndex,
			testSet,
			trainingSet,
			trainingSetCorrectClassificationInPercent,
			testSetCorrectClassificationInPercent,
			bestOptimization,
			bestTestSetCorrectClassificationInPercent,
			minimumDeviation,
			deviation,
			correctClassificationInPercentList,
			listOfTestSetCorrectClassificationInPercentAndDeviation
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Utility options *)
	    bestOptimization = UtilityOptionBestOptimization/.{opts}/.Options[UtilityOptionsOptimization];

		Switch[bestOptimization,

			(* ------------------------------------------------------------------------------- *)
			"BestTestResult",			
			trainingAndTestSetList = mlp1TrainOptimization[[3]];
			mlp1InfoList = mlp1TrainOptimization[[4]];
			maximumCorrectClassificationInPercent = -1.0;
			
			ParallelNeeds[{"CIP`Mlp1`", "CIP`DataTransformation`", "CIP`Utility`"}];
			DistributeDefinitions[trainingAndTestSetList, mlp1InfoList];
			
			correctClassificationInPercentList = ParallelTable[
				testSet = trainingAndTestSetList[[k, 2]];
				mlp1Info = mlp1InfoList[[k]];
				
				CalculateMlp1CorrectClassificationInPercent[testSet, mlp1Info],
				
				{k, Length[mlp1InfoList]}
			];
			
			Do[
				If[correctClassificationInPercentList[[k]] > maximumCorrectClassificationInPercent,
					maximumCorrectClassificationInPercent = correctClassificationInPercentList[[k]];
					bestIndex = k
				],
				
				{k, Length[mlp1InfoList]}
			],
			
			(* ------------------------------------------------------------------------------- *)			
			"MinimumDeviation",
			trainingAndTestSetList = mlp1TrainOptimization[[3]];
			mlp1InfoList = mlp1TrainOptimization[[4]];
			minimumDeviation = Infinity;
			
			ParallelNeeds[{"CIP`Mlp1`", "CIP`DataTransformation`", "CIP`Utility`"}];
			DistributeDefinitions[trainingAndTestSetList, mlp1InfoList];
			
			listOfTestSetCorrectClassificationInPercentAndDeviation = ParallelTable[
				trainingSet = trainingAndTestSetList[[k, 1]];
				testSet = trainingAndTestSetList[[k, 2]];
				mlp1Info = mlp1InfoList[[k]];
				trainingSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[trainingSet, mlp1Info];
				testSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[testSet, mlp1Info];
				deviation = Abs[testSetCorrectClassificationInPercent - trainingSetCorrectClassificationInPercent];
				
				{
					testSetCorrectClassificationInPercent,
					deviation
				},
				
				{k, Length[mlp1InfoList]}
			];
			
			Do[
				If[listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]] < minimumDeviation || (listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]] == minimumDeviation && listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 1]] < bestTestSetCorrectClassificationInPercent),
					minimumDeviation = listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]];
					bestTestSetCorrectClassificationInPercent = listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 1]];
					bestIndex = k
				],
				
				{k, Length[mlp1InfoList]}
			]
		];

		Return[bestIndex]
	];

GetBestMlp1RegressOptimization[

	(* Returns best optimization result of mlp1 for regression.

	   Returns: 
	   Best index for regression *)


	(* mlp1TrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlp1InfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlp1InfoList: List with mlp1Info
	   mlp1InfoList[[i]] refers to optimization step i *)
	mlp1TrainOptimization_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			bestOptimization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Utility options *)
	    bestOptimization = UtilityOptionBestOptimization/.{opts}/.Options[UtilityOptionsOptimization];

		Return[
			CIP`Utility`GetBestRegressOptimization[
				mlp1TrainOptimization, 
				UtilityOptionBestOptimization -> bestOptimization
			]
		]
	];

GetInternalMeanSquaredErrorOfMlp1[

	(* Calculates mean squared error of specified data set for 3-Layer-True-Unit Mlp1 with specified weights

	   Returns:
	   Mean squared error of data set *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output: {outputComponent1, outputComponent2, ...})
	   NOTE: Each component must be in [0, 1] *)
    dataSet_,
    
    (* Weights from input to hidden units *)
    hiddenWeights_/;MatrixQ[hiddenWeights, NumberQ],
    
    (* Weights from hidden to output units *)
    outputWeights_/;MatrixQ[outputWeights, NumberQ],
    
    (* Activation and scaling, see Mlp1OptionActivationAndScaling *)
    activationAndScaling_
    
	] :=
  
	Module[
    
		{
			errors,
			hidden,
			inputs,
			machineOutputs,
			outputs
		},
    
		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];
		outputs = CIP`Utility`GetOutputsOfDataSet[dataSet];

	    (* Add 'true unit' to inputs *)
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			hidden = SigmoidFunction[Map[Append[#, 1.0] &, inputs].Transpose[hiddenWeights]],
			
			"Tanh",
			hidden = Tanh[Map[Append[#, 1.0] &, inputs].Transpose[hiddenWeights]]
		];

	    (* Add 'true unit' to hidden *)
		Switch[activationAndScaling[[1, 2]],
			
			"Sigmoid",
			machineOutputs = SigmoidFunction[Map[Append[#, 1.0] &, hidden].Transpose[outputWeights]],
			
			"Tanh",
			machineOutputs = Tanh[Map[Append[#, 1.0] &, hidden].Transpose[outputWeights]]
		];

	    errors = outputs - machineOutputs;
        Return[Apply[Plus, errors^2, {0,1}]/Length[dataSet]]
	];

GetInternalMlp1Output[

	(* Calculates internal output for specified input of 3-Layer-True-Unit mlp1 with specified weights.

	   Returns:
	   output: {valueOfOutput1, valueOfOutput2, ...} *)

    
    (* input: {valueForInput1, valueForInput1, ...} *)
    input_/;VectorQ[input, NumberQ],
    
    (* Weights from input to hidden units *)
    hiddenWeights_/;MatrixQ[hiddenWeights, NumberQ],
    
    (* Weights from hidden to output units *)
    outputWeights_/;MatrixQ[outputWeights, NumberQ],
    
    (* Activation and scaling, see Mlp1OptionActivationAndScaling *)
    activationAndScaling_
    
	] :=
  
	Module[
    
		{
	      hidden,
	      internalInputs,
	      trueUnitHidden,
	      outputs
		},
    
		(* Add 'true unit' to inputs *)
		internalInputs = Append[input, 1.0];
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			hidden = SigmoidFunction[internalInputs.Transpose[hiddenWeights]],
			
			"Tanh",
			hidden = Tanh[internalInputs.Transpose[hiddenWeights]]
		];

	    (* Add 'true unit' to hidden *)
		trueUnitHidden = Append[hidden, 1.0];
		Switch[activationAndScaling[[1, 2]],
			
			"Sigmoid",
			outputs = SigmoidFunction[trueUnitHidden.Transpose[outputWeights]],
			
			"Tanh",
			outputs = Tanh[trueUnitHidden.Transpose[outputWeights]]
		];
		
		Return[outputs];
    ];

GetInternalMlp1Outputs[

	(* Calculates internal outputs for specified inputs for 3-Layer-True-Unit mlp1 with specified weights.

	   Returns:
	   outputs: {output1, output2, ...} 
	   output: {valueOfOutput1, valueOfOutput2, ...} *)

    
    (* inputs: {input1, input2, ...} 
       input: {valueForInput1, valueForInput1, ...} *)
    inputs_/;MatrixQ[inputs, NumberQ],
    
    (* Weights from input to hidden units *)
    hiddenWeights_/;MatrixQ[hiddenWeights, NumberQ],
    
    (* Weights from hidden to output units *)
    outputWeights_/;MatrixQ[outputWeights, NumberQ],
    
    (* Activation and scaling, see Mlp1OptionActivationAndScaling *)
    activationAndScaling_
    
	] :=
  
	Module[
    
		{
	      hidden,
	      internalInputs,
	      trueUnitHidden,
	      outputs
		},
    
		(* Add 'true unit' to inputs *)
		internalInputs = Map[Append[#, 1.0] &, inputs];
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			hidden = SigmoidFunction[internalInputs.Transpose[hiddenWeights]],
			
			"Tanh",
			hidden = Tanh[internalInputs.Transpose[hiddenWeights]]
		];
	    
	    (* Add 'true unit' to hidden *)
		trueUnitHidden = Map[Append[#, 1.0] &, hidden];
		Switch[activationAndScaling[[1, 2]],
			
			"Sigmoid",
			outputs = SigmoidFunction[trueUnitHidden.Transpose[outputWeights]],
			
			"Tanh",
			outputs = Tanh[trueUnitHidden.Transpose[outputWeights]]
		];
		
		Return[outputs];
    ];

GetNumberOfHiddenNeurons[

	(* Returns number of hidden neurons for specified mlp1Info.

	   Returns:
	   Number of hidden neurons *)

    
  	(* See "Frequently used data structures" *)
    mlp1Info_
    
	] :=
  
	Module[
    
		{},
		
		Return[
			GetMlp1Structure[mlp1Info][[2]]
		]
	];

GetMlp1InputInclusionClass[

	(* Analyzes relevance of input components by successive get-one-in for classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlp1InputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mlp1Info}
	   trainingSetResult: {numberOfIncludedInputs, (best) classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, (best) classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,	
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlp1s,
			optimizationMethod,
			initialWeights,
			initialNetworks,
			weightsValueLimit,
			minimizationPrecision,
			maximumNumberOfIterations,
			numberOfIterationsToImprove,
			learningParameterMin,
			learningParameterMax,
			momentumParameter,
			populationSize,
			crossoverProbability,
			mutationProbability,
			randomValueInitialization,
			activationAndScaling,
			normalizationType,
			isIntermediateOutput,
			numberOfInclusionsPerStepList,
			isRegression,
			inclusionStartList,
			parallelization,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp1 options *)   
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
		(* Utility options *)   		
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfInclusionsPerStepList = UtilityOptionInclusionsPerStep/.{opts}/.Options[UtilityOptionsInclusion];
	    inclusionStartList = UtilityOptionInclusionStartList/.{opts}/.Options[UtilityOptionsInclusion];
	    parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
	    (* DataTransformation options *)   
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		
		isRegression = False;
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				GetMlp1InputInclusionCalculationPC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
    				Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
	 				Mlp1OptionReportIteration -> 0,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
    				Mlp1OptionActivationAndScaling -> activationAndScaling,
    				Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
    				Mlp1OptionCostFunctionType -> costFunctionType,
    				DataTransformationOptionNormalizationType -> normalizationType,
    				UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetMlp1InputInclusionCalculationSC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
    				Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
	 				Mlp1OptionReportIteration -> 0,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
    				Mlp1OptionActivationAndScaling -> activationAndScaling,
    				Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
    				Mlp1OptionCostFunctionType -> costFunctionType,
    				DataTransformationOptionNormalizationType -> normalizationType,
    				UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			]
		]
	];

GetMlp1InputInclusionRegress[

	(* Analyzes relevance of input components by successive get-one-in for regression.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlp1InputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mlp1Info}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,	

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlp1s,
			optimizationMethod,
			initialWeights,
			initialNetworks,
			weightsValueLimit,
			minimizationPrecision,
			maximumNumberOfIterations,
			numberOfIterationsToImprove,
			learningParameterMin,
			learningParameterMax,
			momentumParameter,
			populationSize,
			crossoverProbability,
			mutationProbability,
			randomValueInitialization,
			activationAndScaling,
			normalizationType,
			isIntermediateOutput,
			numberOfInclusionsPerStepList,
			isRegression,
			inclusionStartList,
			parallelization,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp1 options *)   
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
		(* Utility options *)   		
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfInclusionsPerStepList = UtilityOptionInclusionsPerStep/.{opts}/.Options[UtilityOptionsInclusion];
	    inclusionStartList = UtilityOptionInclusionStartList/.{opts}/.Options[UtilityOptionsInclusion];
	    parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
	    (* DataTransformation options *)   
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		
		isRegression = True;
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				GetMlp1InputInclusionCalculationPC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
    				Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
	 				Mlp1OptionReportIteration -> 0,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
    				Mlp1OptionActivationAndScaling -> activationAndScaling,
    				Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
    				Mlp1OptionCostFunctionType -> costFunctionType,
    				DataTransformationOptionNormalizationType -> normalizationType,
    				UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetMlp1InputInclusionCalculationSC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
    				Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
	 				Mlp1OptionReportIteration -> 0,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
    				Mlp1OptionActivationAndScaling -> activationAndScaling,
    				Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
    				Mlp1OptionCostFunctionType -> costFunctionType,
    				DataTransformationOptionNormalizationType -> normalizationType,
    				UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			]
		]
	];

GetMlp1InputInclusionCalculationSC[

	(* Analyzes relevance of input components by successive get-one-in for regression and classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlp1InputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mlp1Info}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,	
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlp1s,
			optimizationMethod,
			initialWeights,
			initialNetworks,
			weightsValueLimit,
			minimizationPrecision,
			maximumNumberOfIterations,
			numberOfIterationsToImprove,
			learningParameterMin,
			learningParameterMax,
			momentumParameter,
			populationSize,
			crossoverProbability,
			mutationProbability,
			randomValueInitialization,
			activationAndScaling,
			normalizationType,
			currentIncludedInputComponentList,
			i,
			k,
			numberOfInputs,
			numberOfIncludedInputs,
			mlp1InputComponentRelevanceList,
	        mlp1Info,
			includedInputComponentList,
			relevance,
			testSet,
			trainingSet,
			testSetRmse,
			trainingSetRmse,
			isIntermediateOutput,
			rmseList,
			sortedRmseList,
			currentTrainingSetRmse,
			currentTestSetRmse,
			currentNumberOfInclusions,
			numberOfInclusionsPerStepList,
			currentTestSetCorrectClassificationInPercent,
			currentTrainingSetCorrectClassificationInPercent,
			inclusionStartList,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp1 options *)   
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
		(* Utility options *)   		
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfInclusionsPerStepList = UtilityOptionInclusionsPerStep/.{opts}/.Options[UtilityOptionsInclusion];
	    inclusionStartList = UtilityOptionInclusionStartList/.{opts}/.Options[UtilityOptionsInclusion];
	    (* DataTransformation options *)   
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		numberOfInputs = First[Dimensions[trainingAndTestSet[[1, 1, 1]] ]];
		If[Length[numberOfInclusionsPerStepList] == 0,
			numberOfInclusionsPerStepList = Table[1, {numberOfInputs}];
		];				   
		includedInputComponentList = inclusionStartList;
		numberOfIncludedInputs = Length[includedInputComponentList];
		mlp1InputComponentRelevanceList = {};
    
		(* ----------------------------------------------------------------------------------------------------
		   Main loop over numberOfInclusionsPerStepList
		   ---------------------------------------------------------------------------------------------------- *)
		Do[
			(* Loop over all input units *)
			currentNumberOfInclusions = numberOfInclusionsPerStepList[[k]];
			rmseList = {};
			Do[
				If[Length[Position[includedInputComponentList, i]] == 0,
					currentIncludedInputComponentList = Append[includedInputComponentList, i];
					trainingSet = trainingAndTestSet[[1]];
					testSet = trainingAndTestSet[[2]];
					trainingSet = CIP`DataTransformation`IncludeInputComponentsOfDataSet[trainingSet, currentIncludedInputComponentList];
    				If[Length[testSet] > 0, 
						testSet = CIP`DataTransformation`IncludeInputComponentsOfDataSet[testSet, currentIncludedInputComponentList]
					];
					mlp1Info = 
						FitMlp1[
							trainingSet,
							numberOfHiddenNeurons,
							Mlp1OptionMultipleMlp1s -> multipleMlp1s,
			    			Mlp1OptionOptimizationMethod -> optimizationMethod,
							Mlp1OptionInitialWeights -> initialWeights,
							Mlp1OptionInitialNetworks -> initialNetworks,
							Mlp1OptionWeightsValueLimit -> weightsValueLimit,
							Mlp1OptionMinimizationPrecision -> minimizationPrecision,
							Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
							Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
				 			Mlp1OptionReportIteration -> 0,
							Mlp1OptionLearningParameterMin -> learningParameterMin,
							Mlp1OptionLearningParameterMax -> learningParameterMax,
							Mlp1OptionMomentumParameter -> momentumParameter,
							Mlp1OptionPopulationSize -> populationSize,
							Mlp1OptionCrossoverProbability -> crossoverProbability,
							Mlp1OptionMutationProbability -> mutationProbability,
			    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
			    			Mlp1OptionActivationAndScaling -> activationAndScaling,
			    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
			    			Mlp1OptionCostFunctionType -> costFunctionType,
			    			DataTransformationOptionNormalizationType -> normalizationType
						];
					If[Length[testSet] > 0,
            
						testSetRmse = CalculateMlp1DataSetRmse[testSet, mlp1Info];
						AppendTo[rmseList,{testSetRmse, i}],
          
						trainingSetRmse = CalculateMlp1DataSetRmse[trainingSet, mlp1Info];
						AppendTo[rmseList,{trainingSetRmse, i}]
					]
				],
        
				{i, numberOfInputs}
			];
			sortedRmseList = Sort[rmseList];
			currentIncludedInputComponentList = Flatten[AppendTo[includedInputComponentList, Take[sortedRmseList[[All, 2]], currentNumberOfInclusions]]];
			numberOfIncludedInputs = Length[currentIncludedInputComponentList];
			trainingSet = trainingAndTestSet[[1]];
			testSet = trainingAndTestSet[[2]];
			trainingSet = CIP`DataTransformation`IncludeInputComponentsOfDataSet[trainingSet, currentIncludedInputComponentList];
			If[Length[testSet] > 0, 
				testSet = CIP`DataTransformation`IncludeInputComponentsOfDataSet[testSet, currentIncludedInputComponentList]
			];
			mlp1Info = 
				FitMlp1[
					trainingSet,
					numberOfHiddenNeurons,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
	    			Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp1OptionReportIteration -> 0,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp1OptionActivationAndScaling -> activationAndScaling,
	    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp1OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMlp1DataSetRmse[trainingSet, mlp1Info];
					currentTestSetRmse = CalculateMlp1DataSetRmse[testSet, mlp1Info];
					If[isIntermediateOutput,
						Print["numberOfIncludedInputs            = ", numberOfIncludedInputs];
						Print["currentIncludedInputComponentList = ", currentIncludedInputComponentList];
						Print["currentTrainingSetRmse            = ", currentTrainingSetRmse];
						Print["currentTestSetRmse                = ", currentTestSetRmse]
					];
					relevance = 
						{
							{N[numberOfIncludedInputs], currentTrainingSetRmse}, 
							{N[numberOfIncludedInputs], currentTestSetRmse}, 
							currentIncludedInputComponentList, 
							mlp1Info
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMlp1DataSetRmse[trainingSet, mlp1Info];
					If[isIntermediateOutput,
						Print["numberOfIncludedInputs            = ", numberOfIncludedInputs];
						Print["currentIncludedInputComponentList = ", currentIncludedInputComponentList];
						Print["currentTrainingSetRmse            = ", currentTrainingSetRmse]
					];
					relevance = 
						{
							{N[numberOfIncludedInputs], currentTrainingSetRmse}, 
							{}, 
							currentIncludedInputComponentList, 
							mlp1Info
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[trainingSet, mlp1Info];
					currentTestSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[testSet, mlp1Info];
					If[isIntermediateOutput,
						Print["numberOfIncludedInputs                           = ", numberOfIncludedInputs];
						Print["currentIncludedInputComponentList                = ", currentIncludedInputComponentList];
						Print["currentTrainingSetCorrectClassificationInPercent = ", currentTrainingSetCorrectClassificationInPercent];
						Print["currentTestSetCorrectClassificationInPercent     = ", currentTestSetCorrectClassificationInPercent]
					];
					relevance = 
						{
							{N[numberOfIncludedInputs], currentTrainingSetCorrectClassificationInPercent}, 
							{N[numberOfIncludedInputs], currentTestSetCorrectClassificationInPercent}, 
							currentIncludedInputComponentList, 
							mlp1Info
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[trainingSet, mlp1Info];
					If[isIntermediateOutput,
						Print["numberOfIncludedInputs                           = ", numberOfIncludedInputs];
						Print["currentIncludedInputComponentList                = ", currentIncludedInputComponentList];
						Print["currentTrainingSetCorrectClassificationInPercent = ", currentTrainingSetCorrectClassificationInPercent]
					];
					relevance = 
						{
							{N[numberOfIncludedInputs], currentTrainingSetCorrectClassificationInPercent}, 
							{}, 
							currentIncludedInputComponentList, 
							mlp1Info
						}
				]
			];	

			AppendTo[mlp1InputComponentRelevanceList, relevance];
			includedInputComponentList = currentIncludedInputComponentList,
			
			{k, Length[numberOfInclusionsPerStepList]}
		];
		
		Return[mlp1InputComponentRelevanceList]
	];

GetMlp1InputInclusionCalculationPC[

	(* Analyzes relevance of input components by successive get-one-in for regression and classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlp1InputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mlp1Info}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,	
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlp1s,
			optimizationMethod,
			initialWeights,
			initialNetworks,
			weightsValueLimit,
			minimizationPrecision,
			maximumNumberOfIterations,
			numberOfIterationsToImprove,
			learningParameterMin,
			learningParameterMax,
			momentumParameter,
			populationSize,
			crossoverProbability,
			mutationProbability,
			randomValueInitialization,
			activationAndScaling,
			normalizationType,
			currentIncludedInputComponentList,
			i,
			k,
			numberOfInputs,
			numberOfIncludedInputs,
			mlp1InputComponentRelevanceList,
	        mlp1Info,
			includedInputComponentList,
			relevance,
			testSet,
			trainingSet,
			testSetRmse,
			trainingSetRmse,
			isIntermediateOutput,
			rmseList,
			sortedRmseList,
			currentTrainingSetRmse,
			currentTestSetRmse,
			currentNumberOfInclusions,
			numberOfInclusionsPerStepList,
			currentTestSetCorrectClassificationInPercent,
			currentTrainingSetCorrectClassificationInPercent,
			inclusionStartList,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp1 options *)   
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
		(* Utility options *)   		
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfInclusionsPerStepList = UtilityOptionInclusionsPerStep/.{opts}/.Options[UtilityOptionsInclusion];
	    inclusionStartList = UtilityOptionInclusionStartList/.{opts}/.Options[UtilityOptionsInclusion];
	    (* DataTransformation options *)   
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		
		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		numberOfInputs = First[Dimensions[trainingAndTestSet[[1, 1, 1]] ]];
		If[Length[numberOfInclusionsPerStepList] == 0,
			numberOfInclusionsPerStepList = Table[1, {numberOfInputs}];
		];				   
		includedInputComponentList = inclusionStartList;
		numberOfIncludedInputs = Length[includedInputComponentList];
		mlp1InputComponentRelevanceList = {};
    	
    	ParallelNeeds[{"CIP`Mlp1`", "CIP`DataTransformation`", "CIP`Utility`", "Combinatorica`"}];
		DistributeDefinitions[multipleMlp1s, optimizationMethod, initialWeights,
			initialNetworks, weightsValueLimit, minimizationPrecision, maximumNumberOfIterations, numberOfIterationsToImprove,
			learningParameterMin, learningParameterMax, momentumParameter, populationSize, crossoverProbability,
			mutationProbability, randomValueInitialization, activationAndScaling, normalizationType, lambdaL2Regularization, 
			costFunctionType];
			
		(* ----------------------------------------------------------------------------------------------------
		   Loop over numberOfInclusionsPerStepList
		   ---------------------------------------------------------------------------------------------------- *)
		
		Do[
			(* List over all input units *)
			currentNumberOfInclusions = numberOfInclusionsPerStepList[[k]];
			
			rmseList = With[{temporaryIncludedInputComponentList = includedInputComponentList},
				ParallelTable[
					If[Length[Position[temporaryIncludedInputComponentList, i]] == 0,
						currentIncludedInputComponentList = Append[temporaryIncludedInputComponentList, i];
						trainingSet = trainingAndTestSet[[1]];
						testSet = trainingAndTestSet[[2]];
						trainingSet = CIP`DataTransformation`IncludeInputComponentsOfDataSet[trainingSet, currentIncludedInputComponentList];
    					If[Length[testSet] > 0, 
							testSet = CIP`DataTransformation`IncludeInputComponentsOfDataSet[testSet, currentIncludedInputComponentList]
						];
						
						mlp1Info = 
							FitMlp1[
								trainingSet,
								numberOfHiddenNeurons,
								Mlp1OptionMultipleMlp1s -> multipleMlp1s,
				    			Mlp1OptionOptimizationMethod -> optimizationMethod,
								Mlp1OptionInitialWeights -> initialWeights,
								Mlp1OptionInitialNetworks -> initialNetworks,
								Mlp1OptionWeightsValueLimit -> weightsValueLimit,
								Mlp1OptionMinimizationPrecision -> minimizationPrecision,
								Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
								Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
				 				Mlp1OptionReportIteration -> 0,
								Mlp1OptionLearningParameterMin -> learningParameterMin,
								Mlp1OptionLearningParameterMax -> learningParameterMax,
								Mlp1OptionMomentumParameter -> momentumParameter,
								Mlp1OptionPopulationSize -> populationSize,
								Mlp1OptionCrossoverProbability -> crossoverProbability,
								Mlp1OptionMutationProbability -> mutationProbability,
				    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
			    				Mlp1OptionActivationAndScaling -> activationAndScaling,
			    				Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
			    				Mlp1OptionCostFunctionType -> costFunctionType,
			    				DataTransformationOptionNormalizationType -> normalizationType
							];
						
						If[Length[testSet] > 0,
            
							testSetRmse = CalculateMlp1DataSetRmse[testSet, mlp1Info];
							{testSetRmse, i},
          	
							trainingSetRmse = CalculateMlp1DataSetRmse[trainingSet, mlp1Info];
							{trainingSetRmse, i}
						]
					],
        
					{i, numberOfInputs}
					
				]
			];
			
			(* The Else-Case creates "Null" in the rmseList therefore they have to be deleted *)
			rmseList = DeleteCases[rmseList, Null];
			
			sortedRmseList = Sort[rmseList];
			currentIncludedInputComponentList = Flatten[AppendTo[includedInputComponentList, Take[sortedRmseList[[All, 2]], currentNumberOfInclusions]]];
			numberOfIncludedInputs = Length[currentIncludedInputComponentList];
			trainingSet = trainingAndTestSet[[1]];
			testSet = trainingAndTestSet[[2]];
			trainingSet = CIP`DataTransformation`IncludeInputComponentsOfDataSet[trainingSet, currentIncludedInputComponentList];
			If[Length[testSet] > 0, 
				testSet = CIP`DataTransformation`IncludeInputComponentsOfDataSet[testSet, currentIncludedInputComponentList]
			];
			mlp1Info = 
				FitMlp1[
					trainingSet,
					numberOfHiddenNeurons,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
	    			Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp1OptionReportIteration -> 0,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp1OptionActivationAndScaling -> activationAndScaling,
	    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp1OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionsParallelization -> "ParallelCalculation"
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMlp1DataSetRmse[trainingSet, mlp1Info];
					currentTestSetRmse = CalculateMlp1DataSetRmse[testSet, mlp1Info];
					If[isIntermediateOutput,
						Print["numberOfIncludedInputs            = ", numberOfIncludedInputs];
						Print["currentIncludedInputComponentList = ", currentIncludedInputComponentList];
						Print["currentTrainingSetRmse            = ", currentTrainingSetRmse];
						Print["currentTestSetRmse                = ", currentTestSetRmse]
					];
					relevance = 
						{
							{N[numberOfIncludedInputs], currentTrainingSetRmse}, 
							{N[numberOfIncludedInputs], currentTestSetRmse}, 
							currentIncludedInputComponentList, 
							mlp1Info
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMlp1DataSetRmse[trainingSet, mlp1Info];
					If[isIntermediateOutput,
						Print["numberOfIncludedInputs            = ", numberOfIncludedInputs];
						Print["currentIncludedInputComponentList = ", currentIncludedInputComponentList];
						Print["currentTrainingSetRmse            = ", currentTrainingSetRmse]
					];
					relevance = 
						{
							{N[numberOfIncludedInputs], currentTrainingSetRmse}, 
							{}, 
							currentIncludedInputComponentList, 
							mlp1Info
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[trainingSet, mlp1Info];
					currentTestSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[testSet, mlp1Info];
					If[isIntermediateOutput,
						Print["numberOfIncludedInputs                           = ", numberOfIncludedInputs];
						Print["currentIncludedInputComponentList                = ", currentIncludedInputComponentList];
						Print["currentTrainingSetCorrectClassificationInPercent = ", currentTrainingSetCorrectClassificationInPercent];
						Print["currentTestSetCorrectClassificationInPercent     = ", currentTestSetCorrectClassificationInPercent]
					];
					relevance = 
						{
							{N[numberOfIncludedInputs], currentTrainingSetCorrectClassificationInPercent}, 
							{N[numberOfIncludedInputs], currentTestSetCorrectClassificationInPercent}, 
							currentIncludedInputComponentList, 
							mlp1Info
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[trainingSet, mlp1Info];
					If[isIntermediateOutput,
						Print["numberOfIncludedInputs                           = ", numberOfIncludedInputs];
						Print["currentIncludedInputComponentList                = ", currentIncludedInputComponentList];
						Print["currentTrainingSetCorrectClassificationInPercent = ", currentTrainingSetCorrectClassificationInPercent]
					];
					relevance = 
						{
							{N[numberOfIncludedInputs], currentTrainingSetCorrectClassificationInPercent}, 
							{}, 
							currentIncludedInputComponentList, 
							mlp1Info
						}
				]
			];	

			AppendTo[mlp1InputComponentRelevanceList, relevance];
			includedInputComponentList = currentIncludedInputComponentList,
			
			{k, Length[numberOfInclusionsPerStepList]}
		];
		
		Return[mlp1InputComponentRelevanceList]
	];

GetMlp1InputRelevanceClass[

	(* Analyzes relevance of input components by successive leave-one-out for classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlp1InputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlp1Info}
	   trainingSetResult: {numberOfRemovedInputs, (best) classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, (best) classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,	
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlp1s,
			optimizationMethod,
			initialWeights,
			initialNetworks,
			weightsValueLimit,
			minimizationPrecision,
			maximumNumberOfIterations,
			numberOfIterationsToImprove,
			learningParameterMin,
			learningParameterMax,
			momentumParameter,
			populationSize,
			crossoverProbability,
			mutationProbability,
			randomValueInitialization,
			activationAndScaling,
			normalizationType,
			isIntermediateOutput,
			numberOfExclusionsPerStepList,
			isRegression,
			parallelization,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp1 options *)   
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
		(* Utility options *)   		
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfExclusionsPerStepList = UtilityOptionExclusionsPerStep/.{opts}/.Options[UtilityOptionsExclusion];
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
	    (* DataTransformation options *)   
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		
		isRegression = False;
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				GetMlp1InputRelevanceCalculationPC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
	    			Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp1OptionReportIteration -> 0,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
	    			Mlp1OptionActivationAndScaling -> activationAndScaling,
	    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp1OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetMlp1InputRelevanceCalculationSC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
	    			Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp1OptionReportIteration -> 0,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
	    			Mlp1OptionActivationAndScaling -> activationAndScaling,
	    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp1OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			]
		]
	];

GetMlp1ClassRelevantComponents[

	(* Returns most-to-least-relevance sorted components from mlp1InputComponentRelevanceListForClassification.

	   Returns: Returns most-to-least-relevance sorted components *)


	(* mlp1InputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlp1Info}
	   trainingSetResult: {numberOfRemovedInputs, (best) correct classification in percent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best correct classification in percent of test set} *)
	mlp1InputComponentRelevanceListForClassification_,
	
	(* Number of components to return *)
	numberOfComponents_?IntegerQ
    
	] :=
  
	Module[
    
		{},

		Return[
			CIP`Graphics`GetClassRelevantComponents[mlp1InputComponentRelevanceListForClassification, numberOfComponents]
		]
	];

GetMlp1InputRelevanceRegress[

	(* Analyzes relevance of input components by successive leave-one-out for regression.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlp1InputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlp1Info}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlp1s,
			optimizationMethod,
			initialWeights,
			initialNetworks,
			weightsValueLimit,
			minimizationPrecision,
			maximumNumberOfIterations,
			numberOfIterationsToImprove,
			learningParameterMin,
			learningParameterMax,
			momentumParameter,
			populationSize,
			crossoverProbability,
			mutationProbability,
			randomValueInitialization,
			activationAndScaling,
			normalizationType,
			isIntermediateOutput,
			numberOfExclusionsPerStepList,
			isRegression,
			parallelization,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp1 options *)   
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
		(* Utility options *)   		
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfExclusionsPerStepList = UtilityOptionExclusionsPerStep/.{opts}/.Options[UtilityOptionsExclusion];
	    (* DataTransformation options *)   
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		isRegression = True;
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				GetMlp1InputRelevanceCalculationPC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
	    			Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp1OptionReportIteration -> 0,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
	    			Mlp1OptionActivationAndScaling -> activationAndScaling,
	    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp1OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetMlp1InputRelevanceCalculationSC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
	    			Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp1OptionReportIteration -> 0,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
	    			Mlp1OptionActivationAndScaling -> activationAndScaling,
	    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp1OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			]
		]
	];

GetMlp1RegressRelevantComponents[

	(* Returns most-to-least-relevance sorted components from mlp1InputComponentRelevanceListForRegression.

	   Returns: Returns most-to-least-relevance sorted components *)


	(* mlp1InputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlp1Info}
	   trainingSetResult: {numberOfRemovedInputs, (best) RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)
	mlp1InputComponentRelevanceListForRegression_,
	
	(* Number of components to return *)
	numberOfComponents_?IntegerQ
    
	] :=
  
	Module[
    
		{},

		Return[
			CIP`Graphics`GetRegressRelevantComponents[mlp1InputComponentRelevanceListForRegression, numberOfComponents]
		]
	];

GetMlp1InputRelevanceCalculationSC[

	(* Analyzes relevance of input components by successive leave-one-out for regression and classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlp1InputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlp1Info}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlp1s,
			optimizationMethod,
			initialWeights,
			initialNetworks,
			weightsValueLimit,
			minimizationPrecision,
			maximumNumberOfIterations,
			numberOfIterationsToImprove,
			learningParameterMin,
			learningParameterMax,
			momentumParameter,
			populationSize,
			crossoverProbability,
			mutationProbability,
			randomValueInitialization,
			activationAndScaling,
			normalizationType,
			currentRemovedInputComponentList,
			i,
			k,
			numberOfInputs,
			numberOfRemovedInputs,
			mlp1InputComponentRelevanceList,
	        mlp1Info,
			removedInputComponentList,
			relevance,
			testSet,
			trainingSet,
			initialTestSetRmse,
			initialTrainingSetRmse,
			testSetRmse,
			trainingSetRmse,
			isIntermediateOutput,
			rmseList,
			sortedRmseList,
			currentTrainingSetRmse,
			currentTestSetRmse,
			currentNumberOfExclusions,
			numberOfExclusionsPerStepList,
			initialTestSetCorrectClassificationInPercent,
			initialTrainingSetCorrectClassificationInPercent,
			currentTestSetCorrectClassificationInPercent,
			currentTrainingSetCorrectClassificationInPercent,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp1 options *)   
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
		(* Utility options *)   		
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfExclusionsPerStepList = UtilityOptionExclusionsPerStep/.{opts}/.Options[UtilityOptionsExclusion];
	    (* DataTransformation options *)   
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		numberOfInputs = First[Dimensions[trainingAndTestSet[[1, 1, 1]] ]];
		If[Length[numberOfExclusionsPerStepList] == 0,
			numberOfExclusionsPerStepList = Table[1, {numberOfInputs - 1}];
		];				   
		removedInputComponentList = {};
		mlp1InputComponentRelevanceList = {};
    
		(* Result for no removal *)
		trainingSet = trainingAndTestSet[[1]];
		testSet = trainingAndTestSet[[2]];
		mlp1Info = 
			FitMlp1[
				trainingSet,
				numberOfHiddenNeurons,
				Mlp1OptionMultipleMlp1s -> multipleMlp1s,
    			Mlp1OptionOptimizationMethod -> optimizationMethod,
				Mlp1OptionInitialWeights -> initialWeights,
				Mlp1OptionInitialNetworks -> initialNetworks,
				Mlp1OptionWeightsValueLimit -> weightsValueLimit,
				Mlp1OptionMinimizationPrecision -> minimizationPrecision,
				Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
				Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
	 			Mlp1OptionReportIteration -> 0,
				Mlp1OptionLearningParameterMin -> learningParameterMin,
				Mlp1OptionLearningParameterMax -> learningParameterMax,
				Mlp1OptionMomentumParameter -> momentumParameter,
				Mlp1OptionPopulationSize -> populationSize,
				Mlp1OptionCrossoverProbability -> crossoverProbability,
				Mlp1OptionMutationProbability -> mutationProbability,
    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
    			Mlp1OptionActivationAndScaling -> activationAndScaling,
    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
    			Mlp1OptionCostFunctionType -> costFunctionType,
    			DataTransformationOptionNormalizationType -> normalizationType
			];
		
		initialTrainingSetRmse = CalculateMlp1DataSetRmse[trainingSet, mlp1Info];
		If[isRegression,
			
			(* Regression*)
			If[Length[testSet] > 0,
				
				(* Regression WITH test set*)
				initialTestSetRmse = CalculateMlp1DataSetRmse[testSet, mlp1Info];
				If[isIntermediateOutput,
					Print["initialTrainingSetRmse = ", initialTrainingSetRmse];
					Print["initialTestSetRmse     = ", initialTestSetRmse]
				];
				relevance = 
					{
						{0.0, initialTrainingSetRmse},
						{0.0, initialTestSetRmse},
						{}, 
						mlp1Info
					},
	          
				(* Regression WITHOUT test set*)
				If[isIntermediateOutput,
					Print["initialTrainingSetRmse = ", initialTrainingSetRmse]
				];
				relevance = 
					{
						{0.0, initialTrainingSetRmse},
						{},
						{}, 
						mlp1Info
					}
			],
			
			(* Classification *)
			initialTrainingSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[trainingSet, mlp1Info];
			If[Length[testSet] > 0,
				
				(* Classification WITH test set*)
				initialTestSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[testSet, mlp1Info];
				If[isIntermediateOutput,
					Print["initialTrainingSetCorrectClassificationInPercent = ", initialTrainingSetCorrectClassificationInPercent];
					Print["initialTestSetCorrectClassificationInPercent     = ", initialTestSetCorrectClassificationInPercent]
				];
				relevance = 
					{
						{0.0, initialTrainingSetCorrectClassificationInPercent},
						{0.0, initialTestSetCorrectClassificationInPercent},
						{}, 
						mlp1Info
					},
	          
				(* Classification WITHOUT test set*)
				If[isIntermediateOutput,
					Print["initialTrainingSetCorrectClassificationInPercent = ", initialTrainingSetCorrectClassificationInPercent]
				];
				relevance = 
					{
						{0.0, initialTrainingSetCorrectClassificationInPercent},
						{},
						{}, 
						mlp1Info
					}
			]
		];	
		
		AppendTo[mlp1InputComponentRelevanceList, relevance];
    
		(* ----------------------------------------------------------------------------------------------------
		   Main loop over numberOfExclusionsPerStepList
		   ---------------------------------------------------------------------------------------------------- *)
		numberOfRemovedInputs = 0;
		Do[
			(* Loop over all input units *)
			currentNumberOfExclusions = numberOfExclusionsPerStepList[[k]];
			rmseList = {};
			Do[
				If[Length[Position[removedInputComponentList, i]] == 0,
					currentRemovedInputComponentList = Append[removedInputComponentList, i];
					trainingSet = trainingAndTestSet[[1]];
					testSet = trainingAndTestSet[[2]];
					trainingSet = CIP`DataTransformation`RemoveInputComponentsOfDataSet[trainingSet, currentRemovedInputComponentList];
    				If[Length[testSet] > 0, 
						testSet = CIP`DataTransformation`RemoveInputComponentsOfDataSet[testSet, currentRemovedInputComponentList]
					];
					mlp1Info = 
						FitMlp1[
							trainingSet,
							numberOfHiddenNeurons,
							Mlp1OptionMultipleMlp1s -> multipleMlp1s,
			    			Mlp1OptionOptimizationMethod -> optimizationMethod,
							Mlp1OptionInitialWeights -> initialWeights,
							Mlp1OptionInitialNetworks -> initialNetworks,
							Mlp1OptionWeightsValueLimit -> weightsValueLimit,
							Mlp1OptionMinimizationPrecision -> minimizationPrecision,
							Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
							Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
				 			Mlp1OptionReportIteration -> 0,
							Mlp1OptionLearningParameterMin -> learningParameterMin,
							Mlp1OptionLearningParameterMax -> learningParameterMax,
							Mlp1OptionMomentumParameter -> momentumParameter,
							Mlp1OptionPopulationSize -> populationSize,
							Mlp1OptionCrossoverProbability -> crossoverProbability,
							Mlp1OptionMutationProbability -> mutationProbability,
			    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
			    			Mlp1OptionActivationAndScaling -> activationAndScaling,
			    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
			    			Mlp1OptionCostFunctionType -> costFunctionType,
			    			DataTransformationOptionNormalizationType -> normalizationType
						];
					If[Length[testSet] > 0,
            
						testSetRmse = CalculateMlp1DataSetRmse[testSet, mlp1Info];
						AppendTo[rmseList,{testSetRmse, i}],
          
						trainingSetRmse = CalculateMlp1DataSetRmse[trainingSet, mlp1Info];
						AppendTo[rmseList,{trainingSetRmse, i}]
					]
				],
        
				{i, numberOfInputs}
			];
			sortedRmseList = Sort[rmseList];
			currentRemovedInputComponentList = Flatten[AppendTo[removedInputComponentList, Take[sortedRmseList[[All, 2]], currentNumberOfExclusions]]];
			numberOfRemovedInputs = Length[currentRemovedInputComponentList];
			trainingSet = trainingAndTestSet[[1]];
			testSet = trainingAndTestSet[[2]];
			trainingSet = CIP`DataTransformation`RemoveInputComponentsOfDataSet[trainingSet, currentRemovedInputComponentList];
			If[Length[testSet] > 0, 
				testSet = CIP`DataTransformation`RemoveInputComponentsOfDataSet[testSet, currentRemovedInputComponentList]
			];
			mlp1Info = 
				FitMlp1[
					trainingSet,
					numberOfHiddenNeurons,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
	    			Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp1OptionReportIteration -> 0,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp1OptionActivationAndScaling -> activationAndScaling,
	    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp1OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMlp1DataSetRmse[trainingSet, mlp1Info];
					currentTestSetRmse = CalculateMlp1DataSetRmse[testSet, mlp1Info];
					If[isIntermediateOutput,
						Print["numberOfRemovedInputs            = ", numberOfRemovedInputs];
						Print["currentRemovedInputComponentList = ", currentRemovedInputComponentList];
						Print["currentTrainingSetRmse           = ", currentTrainingSetRmse];
						Print["Delta(current - initial)         = ", currentTrainingSetRmse - initialTrainingSetRmse];
						Print["currentTestSetRmse               = ", currentTestSetRmse];
						Print["Delta(current - initial)         = ", currentTestSetRmse - initialTestSetRmse]
					];
					relevance = 
						{
							{N[numberOfRemovedInputs], currentTrainingSetRmse}, 
							{N[numberOfRemovedInputs], currentTestSetRmse}, 
							currentRemovedInputComponentList, 
							mlp1Info
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMlp1DataSetRmse[trainingSet, mlp1Info];
					If[isIntermediateOutput,
						Print["numberOfRemovedInputs            = ", numberOfRemovedInputs];
						Print["currentRemovedInputComponentList = ", currentRemovedInputComponentList];
						Print["currentTrainingSetRmse           = ", currentTrainingSetRmse];
						Print["Delta(current - initial)         = ", currentTrainingSetRmse - initialTrainingSetRmse]
					];
					relevance = 
						{
							{N[numberOfRemovedInputs], currentTrainingSetRmse}, 
							{}, 
							currentRemovedInputComponentList, 
							mlp1Info
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[trainingSet, mlp1Info];
					currentTestSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[testSet, mlp1Info];
					If[isIntermediateOutput,
						Print["numberOfRemovedInputs                            = ", numberOfRemovedInputs];
						Print["currentRemovedInputComponentList                 = ", currentRemovedInputComponentList];
						Print["currentTrainingSetCorrectClassificationInPercent = ", currentTrainingSetCorrectClassificationInPercent];
						Print["Delta(initial - current)                         = ", initialTrainingSetCorrectClassificationInPercent - currentTrainingSetCorrectClassificationInPercent];
						Print["currentTestSetCorrectClassificationInPercent     = ", currentTestSetCorrectClassificationInPercent];
						Print["Delta(initial - current)                         = ", initialTestSetCorrectClassificationInPercent - currentTestSetCorrectClassificationInPercent]
					];
					relevance = 
						{
							{N[numberOfRemovedInputs], currentTrainingSetCorrectClassificationInPercent}, 
							{N[numberOfRemovedInputs], currentTestSetCorrectClassificationInPercent}, 
							currentRemovedInputComponentList, 
							mlp1Info
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[trainingSet, mlp1Info];
					If[isIntermediateOutput,
						Print["numberOfRemovedInputs                            = ", numberOfRemovedInputs];
						Print["currentRemovedInputComponentList                 = ", currentRemovedInputComponentList];
						Print["currentTrainingSetCorrectClassificationInPercent = ", currentTrainingSetCorrectClassificationInPercent];
						Print["Delta(initial - current)                         = ", initialTrainingSetCorrectClassificationInPercent - currentTrainingSetCorrectClassificationInPercent]
					];
					relevance = 
						{
							{N[numberOfRemovedInputs], currentTrainingSetCorrectClassificationInPercent}, 
							{}, 
							currentRemovedInputComponentList, 
							mlp1Info
						}
				]
			];	

			AppendTo[mlp1InputComponentRelevanceList, relevance];
			removedInputComponentList = currentRemovedInputComponentList,
			
			{k, Length[numberOfExclusionsPerStepList]}
		];
		
		Return[mlp1InputComponentRelevanceList]
	];

GetMlp1InputRelevanceCalculationPC[

	(* Analyzes relevance of input components by successive leave-one-out for regression and classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlp1InputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlp1Info}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlp1s,
			optimizationMethod,
			initialWeights,
			initialNetworks,
			weightsValueLimit,
			minimizationPrecision,
			maximumNumberOfIterations,
			numberOfIterationsToImprove,
			learningParameterMin,
			learningParameterMax,
			momentumParameter,
			populationSize,
			crossoverProbability,
			mutationProbability,
			randomValueInitialization,
			activationAndScaling,
			normalizationType,
			currentRemovedInputComponentList,
			i,
			k,
			numberOfInputs,
			numberOfRemovedInputs,
			mlp1InputComponentRelevanceList,
	        mlp1Info,
			removedInputComponentList,
			relevance,
			testSet,
			trainingSet,
			initialTestSetRmse,
			initialTrainingSetRmse,
			testSetRmse,
			trainingSetRmse,
			isIntermediateOutput,
			rmseList,
			sortedRmseList,
			currentTrainingSetRmse,
			currentTestSetRmse,
			currentNumberOfExclusions,
			numberOfExclusionsPerStepList,
			initialTestSetCorrectClassificationInPercent,
			initialTrainingSetCorrectClassificationInPercent,
			currentTestSetCorrectClassificationInPercent,
			currentTrainingSetCorrectClassificationInPercent,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp1 options *)   
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
		(* Utility options *)   		
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfExclusionsPerStepList = UtilityOptionExclusionsPerStep/.{opts}/.Options[UtilityOptionsExclusion];
	    (* DataTransformation options *)   
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		
		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		numberOfInputs = First[Dimensions[trainingAndTestSet[[1, 1, 1]] ]];
		If[Length[numberOfExclusionsPerStepList] == 0,
			numberOfExclusionsPerStepList = Table[1, {numberOfInputs - 1}];
		];				   
		removedInputComponentList = {};
		mlp1InputComponentRelevanceList = {};
    
		(* Result for no removal *)
		trainingSet = trainingAndTestSet[[1]];
		testSet = trainingAndTestSet[[2]];
		mlp1Info = 
			FitMlp1[
				trainingSet,
				numberOfHiddenNeurons,
				Mlp1OptionMultipleMlp1s -> multipleMlp1s,
    			Mlp1OptionOptimizationMethod -> optimizationMethod,
				Mlp1OptionInitialWeights -> initialWeights,
				Mlp1OptionInitialNetworks -> initialNetworks,
				Mlp1OptionWeightsValueLimit -> weightsValueLimit,
				Mlp1OptionMinimizationPrecision -> minimizationPrecision,
				Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
				Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
	 			Mlp1OptionReportIteration -> 0,
				Mlp1OptionLearningParameterMin -> learningParameterMin,
				Mlp1OptionLearningParameterMax -> learningParameterMax,
				Mlp1OptionMomentumParameter -> momentumParameter,
				Mlp1OptionPopulationSize -> populationSize,
				Mlp1OptionCrossoverProbability -> crossoverProbability,
				Mlp1OptionMutationProbability -> mutationProbability,
    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
    			Mlp1OptionActivationAndScaling -> activationAndScaling,
    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
    			Mlp1OptionCostFunctionType -> costFunctionType,
    			DataTransformationOptionNormalizationType -> normalizationType,
    			UtilityOptionsParallelization -> "ParallelCalculation"
			];
			
		initialTrainingSetRmse = CalculateMlp1DataSetRmse[trainingSet, mlp1Info];
		If[isRegression,
			
			(* Regression*)
			If[Length[testSet] > 0,
				
				(* Regression WITH test set*)
				initialTestSetRmse = CalculateMlp1DataSetRmse[testSet, mlp1Info];
				If[isIntermediateOutput,
					Print["initialTrainingSetRmse = ", initialTrainingSetRmse];
					Print["initialTestSetRmse     = ", initialTestSetRmse]
				];
				relevance = 
					{
						{0.0, initialTrainingSetRmse},
						{0.0, initialTestSetRmse},
						{}, 
						mlp1Info
					},
	          
				(* Regression WITHOUT test set*)
				If[isIntermediateOutput,
					Print["initialTrainingSetRmse = ", initialTrainingSetRmse]
				];
				relevance = 
					{
						{0.0, initialTrainingSetRmse},
						{},
						{}, 
						mlp1Info
					}
			],
			
			(* Classification *)
			initialTrainingSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[trainingSet, mlp1Info];
			If[Length[testSet] > 0,
				
				(* Classification WITH test set*)
				initialTestSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[testSet, mlp1Info];
				If[isIntermediateOutput,
					Print["initialTrainingSetCorrectClassificationInPercent = ", initialTrainingSetCorrectClassificationInPercent];
					Print["initialTestSetCorrectClassificationInPercent     = ", initialTestSetCorrectClassificationInPercent]
				];
				relevance = 
					{
						{0.0, initialTrainingSetCorrectClassificationInPercent},
						{0.0, initialTestSetCorrectClassificationInPercent},
						{}, 
						mlp1Info
					},
	          
				(* Classification WITHOUT test set*)
				If[isIntermediateOutput,
					Print["initialTrainingSetCorrectClassificationInPercent = ", initialTrainingSetCorrectClassificationInPercent]
				];
				relevance = 
					{
						{0.0, initialTrainingSetCorrectClassificationInPercent},
						{},
						{}, 
						mlp1Info
					}
			]
		];	
		
		AppendTo[mlp1InputComponentRelevanceList, relevance];
    
		ParallelNeeds[{"CIP`Mlp1`", "CIP`DataTransformation`", "CIP`Utility`", "Combinatorica`"}];
		DistributeDefinitions[multipleMlp1s, optimizationMethod, initialWeights,
			initialNetworks, weightsValueLimit, minimizationPrecision, maximumNumberOfIterations, numberOfIterationsToImprove,
			learningParameterMin, learningParameterMax, momentumParameter, populationSize, crossoverProbability,
			mutationProbability, randomValueInitialization, activationAndScaling, normalizationType];
			    
		(* ----------------------------------------------------------------------------------------------------
		   Loop over numberOfExclusionsPerStepList
		   ---------------------------------------------------------------------------------------------------- *)
		numberOfRemovedInputs = 0;
		Do[
			(* List over all input units *)
			currentNumberOfExclusions = numberOfExclusionsPerStepList[[k]];
			
			rmseList = With[{temporaryRemovedInputComponentList = removedInputComponentList},
				ParallelTable[
					If[Length[Position[temporaryRemovedInputComponentList, i]] == 0,
						currentRemovedInputComponentList = Append[temporaryRemovedInputComponentList, i];
						trainingSet = trainingAndTestSet[[1]];
						testSet = trainingAndTestSet[[2]];
						trainingSet = CIP`DataTransformation`RemoveInputComponentsOfDataSet[trainingSet, currentRemovedInputComponentList];
	    				If[Length[testSet] > 0, 
							testSet = CIP`DataTransformation`RemoveInputComponentsOfDataSet[testSet, currentRemovedInputComponentList]
						];
						
						mlp1Info = 
							FitMlp1[
								trainingSet,
								numberOfHiddenNeurons,
								Mlp1OptionMultipleMlp1s -> multipleMlp1s,
				    			Mlp1OptionOptimizationMethod -> optimizationMethod,
								Mlp1OptionInitialWeights -> initialWeights,
								Mlp1OptionInitialNetworks -> initialNetworks,
								Mlp1OptionWeightsValueLimit -> weightsValueLimit,
								Mlp1OptionMinimizationPrecision -> minimizationPrecision,
								Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
								Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
					 			Mlp1OptionReportIteration -> 0,
								Mlp1OptionLearningParameterMin -> learningParameterMin,
								Mlp1OptionLearningParameterMax -> learningParameterMax,
								Mlp1OptionMomentumParameter -> momentumParameter,
								Mlp1OptionPopulationSize -> populationSize,
								Mlp1OptionCrossoverProbability -> crossoverProbability,
								Mlp1OptionMutationProbability -> mutationProbability,
				    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
				    			Mlp1OptionActivationAndScaling -> activationAndScaling,
				    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
				    			Mlp1OptionCostFunctionType -> costFunctionType,
				    			DataTransformationOptionNormalizationType -> normalizationType
							];
								
						If[Length[testSet] > 0,
	            
							testSetRmse = CalculateMlp1DataSetRmse[testSet, mlp1Info];
							{testSetRmse, i},
	          
							trainingSetRmse = CalculateMlp1DataSetRmse[trainingSet, mlp1Info];
							{trainingSetRmse, i}
						]
					],
	        
					{i, numberOfInputs}
				]
			];
				
			(* The Else-Case creates "Null" in the rmseList therefore they have to be deleted *)
			rmseList = DeleteCases[rmseList, Null];
			
			sortedRmseList = Sort[rmseList];
			currentRemovedInputComponentList = Flatten[AppendTo[removedInputComponentList, Take[sortedRmseList[[All, 2]], currentNumberOfExclusions]]];
			numberOfRemovedInputs = Length[currentRemovedInputComponentList];
			trainingSet = trainingAndTestSet[[1]];
			testSet = trainingAndTestSet[[2]];
			trainingSet = CIP`DataTransformation`RemoveInputComponentsOfDataSet[trainingSet, currentRemovedInputComponentList];
			If[Length[testSet] > 0, 
				testSet = CIP`DataTransformation`RemoveInputComponentsOfDataSet[testSet, currentRemovedInputComponentList]
			];
			mlp1Info = 
				FitMlp1[
					trainingSet,
					numberOfHiddenNeurons,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
	    			Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp1OptionReportIteration -> 0,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp1OptionActivationAndScaling -> activationAndScaling,
	    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp1OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionsParallelization -> "ParallelCalculation"
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMlp1DataSetRmse[trainingSet, mlp1Info];
					currentTestSetRmse = CalculateMlp1DataSetRmse[testSet, mlp1Info];
					If[isIntermediateOutput,
						Print["numberOfRemovedInputs            = ", numberOfRemovedInputs];
						Print["currentRemovedInputComponentList = ", currentRemovedInputComponentList];
						Print["currentTrainingSetRmse           = ", currentTrainingSetRmse];
						Print["Delta(current - initial)         = ", currentTrainingSetRmse - initialTrainingSetRmse];
						Print["currentTestSetRmse               = ", currentTestSetRmse];
						Print["Delta(current - initial)         = ", currentTestSetRmse - initialTestSetRmse]
					];
					relevance = 
						{
							{N[numberOfRemovedInputs], currentTrainingSetRmse}, 
							{N[numberOfRemovedInputs], currentTestSetRmse}, 
							currentRemovedInputComponentList, 
							mlp1Info
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMlp1DataSetRmse[trainingSet, mlp1Info];
					If[isIntermediateOutput,
						Print["numberOfRemovedInputs            = ", numberOfRemovedInputs];
						Print["currentRemovedInputComponentList = ", currentRemovedInputComponentList];
						Print["currentTrainingSetRmse           = ", currentTrainingSetRmse];
						Print["Delta(current - initial)         = ", currentTrainingSetRmse - initialTrainingSetRmse]
					];
					relevance = 
						{
							{N[numberOfRemovedInputs], currentTrainingSetRmse}, 
							{}, 
							currentRemovedInputComponentList, 
							mlp1Info
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[trainingSet, mlp1Info];
					currentTestSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[testSet, mlp1Info];
					If[isIntermediateOutput,
						Print["numberOfRemovedInputs                            = ", numberOfRemovedInputs];
						Print["currentRemovedInputComponentList                 = ", currentRemovedInputComponentList];
						Print["currentTrainingSetCorrectClassificationInPercent = ", currentTrainingSetCorrectClassificationInPercent];
						Print["Delta(initial - current)                         = ", initialTrainingSetCorrectClassificationInPercent - currentTrainingSetCorrectClassificationInPercent];
						Print["currentTestSetCorrectClassificationInPercent     = ", currentTestSetCorrectClassificationInPercent];
						Print["Delta(initial - current)                         = ", initialTestSetCorrectClassificationInPercent - currentTestSetCorrectClassificationInPercent]
					];
					relevance = 
						{
							{N[numberOfRemovedInputs], currentTrainingSetCorrectClassificationInPercent}, 
							{N[numberOfRemovedInputs], currentTestSetCorrectClassificationInPercent}, 
							currentRemovedInputComponentList, 
							mlp1Info
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlp1CorrectClassificationInPercent[trainingSet, mlp1Info];
					If[isIntermediateOutput,
						Print["numberOfRemovedInputs                            = ", numberOfRemovedInputs];
						Print["currentRemovedInputComponentList                 = ", currentRemovedInputComponentList];
						Print["currentTrainingSetCorrectClassificationInPercent = ", currentTrainingSetCorrectClassificationInPercent];
						Print["Delta(initial - current)                         = ", initialTrainingSetCorrectClassificationInPercent - currentTrainingSetCorrectClassificationInPercent]
					];
					relevance = 
						{
							{N[numberOfRemovedInputs], currentTrainingSetCorrectClassificationInPercent}, 
							{}, 
							currentRemovedInputComponentList, 
							mlp1Info
						}
				]
			];	

			AppendTo[mlp1InputComponentRelevanceList, relevance];
			removedInputComponentList = currentRemovedInputComponentList,
			
			{k, Length[numberOfExclusionsPerStepList]}
		];
		
		Return[mlp1InputComponentRelevanceList]
	];
	
GetMlp1RegressionResult[
	
	(* Returns mlp1 regression result according to named property list.

	   Returns :
	   Mlp1 regression result according to named property *)

	(* Properties to be analyzed: 
	   Full list: 
	   {
	       "RMSE",
	       "SingleOutputRMSE",
	       "AbsoluteResidualsStatistics",
		   "RelativeResidualsStatistics",
		   "ModelVsData",
		   "CorrelationCoefficient",
		   "SortedModelVsData",
		   "AbsoluteSortedResiduals",
		   "RelativeSortedResiduals",
		   "AbsoluteResidualsDistribution",
		   "RelativeResidualsDistribution"
	    } *)
 	namedProperty_,
    
    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...} *)
    dataSet_,
    
	(* See "Frequently used data structures" *)
    mlp1Info_,
	
	(* Options *)
	opts___

    
	] :=
  
	Module[
    
    	{
    		numberOfIntervals,
    		pureFunction
    	},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    numberOfIntervals = GraphicsOptionNumberOfIntervals/.{opts}/.Options[GraphicsOptionsResidualsDistribution];
	    
		pureFunction = Function[inputs, CalculateMlp1Outputs[inputs, mlp1Info]];
	    Return[
	    	CIP`Graphics`GetSingleRegressionResult[
		    	namedProperty, 
		    	dataSet, 
		    	pureFunction,
		    	GraphicsOptionNumberOfIntervals -> numberOfIntervals
			]
		]
	];

GetMlp1SeriesClassificationResult[

	(* Shows result of Mlp1 series classifications for training and test set.

	   Returns: 
	   mlp1SeriesClassificationResult: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in mlp1InfoList, classification result in percent for training set}
	   testPoint[[i]]: {index i in mlp1InfoList, classification result in percent for test set} *)


    (* {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...}
	   testSet has the same structure and restrictions as trainingSet 
	   NOTE: Training and test set MUST be in original units *)
    trainingAndTestSet_,

	(* {mlp1Info1, mlp1Info2, ...}
	   mlp1Info (see "Frequently used data structures") *)
    mlp1InfoList_
    
	] :=
  
	Module[
    
		{
			testSet,
			trainingSet,
			pureFunction,
			i,
			trainingPoints2D,
			testPoints2D,
			correctClassificationInPercent
		},

    	trainingSet = trainingAndTestSet[[1]];
    	testSet = trainingAndTestSet[[2]];

		trainingPoints2D = {};
		testPoints2D = {};
		Do[
			pureFunction = Function[inputs, CalculateMlp1ClassNumbers[inputs, mlp1InfoList[[i]]]];
			correctClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[trainingSet, pureFunction];
			AppendTo[trainingPoints2D, {N[i], correctClassificationInPercent}];
			If[Length[testSet] > 0,
				correctClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[testSet, pureFunction];
				AppendTo[testPoints2D, {N[i], correctClassificationInPercent}]
			],
			
			{i, Length[mlp1InfoList]}
		];
		
		Return[{trainingPoints2D, testPoints2D}]
	];

GetMlp1SeriesRmse[

	(* Shows RMSE of Mlp1 series for training and test set.

	   Returns: 
	   mlp1SeriesRmse: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in mlp1InfoList, RMSE for training set}
	   testPoint[[i]]: {index i in mlp1InfoList, RMSE for test set} *)


    (* {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...}
	   testSet has the same structure and restrictions as trainingSet 
	   NOTE: Training and test set MUST be in original units *)
    trainingAndTestSet_,

	(* {mlp1Info1, mlp1Info2, ...}
	   mlp1Info (see "Frequently used data structures") *)
    mlp1InfoList_
    
	] :=
  
	Module[
    
		{
			testSet,
			trainingSet,
			pureFunction,
			i,
			trainingPoints2D,
			testPoints2D,
			rmse
		},

    	trainingSet = trainingAndTestSet[[1]];
    	testSet = trainingAndTestSet[[2]];

		trainingPoints2D = {};
		testPoints2D = {};
		Do[
			pureFunction = Function[inputs, CalculateMlp1Outputs[inputs, mlp1InfoList[[i]]]];
			rmse = Sqrt[CIP`Utility`GetMeanSquaredError[trainingSet, pureFunction]];
			AppendTo[trainingPoints2D, {N[i], rmse}];
			If[Length[testSet] > 0,
				rmse = Sqrt[CIP`Utility`GetMeanSquaredError[testSet, pureFunction]];
				AppendTo[testPoints2D, {N[i], rmse}]
			],
			
			{i, Length[mlp1InfoList]}
		];
		
		Return[{trainingPoints2D, testPoints2D}]
	];

GetMlp1Structure[

	(* Returns mlp1 structure for specified mlp1Info.

	   Returns:
	   {numberOfInputs, numberOfHiddenNeurons, numberOfOutputs} *)

    
  	(* See "Frequently used data structures" *)
    mlp1Info_
    
	] :=
  
	Module[
    
		{
			hiddenWeights,
			numberOfHiddenNeurons,
			numberOfInputs,
			numberOfOutputs,
			networks,
			outputWeights,
			weights
		},
    
    	networks = mlp1Info[[1]];
    	
		If[Length[networks] == 1,
	
			(* --------------------------------------------------------------------------------
			   One network
			   -------------------------------------------------------------------------------- *)		

	    	weights = networks[[1]];
	    	hiddenWeights = weights[[1]];
	    	outputWeights = weights[[2]];
	    	(* - 1: Subtract true unit *)
	    	numberOfInputs = Length[hiddenWeights[[1]]] - 1;
	    	numberOfHiddenNeurons = Length[hiddenWeights];
	    	numberOfOutputs = Length[outputWeights];
	    	Return[{numberOfInputs, numberOfHiddenNeurons, numberOfOutputs}],
			
			(* --------------------------------------------------------------------------------
			   Multiple networks (with ONE output value each)
			   -------------------------------------------------------------------------------- *)		

	    	weights = networks[[1]];
	    	hiddenWeights = weights[[1]];
	    	(* - 1: Subtract true unit *)
	    	numberOfInputs = Length[hiddenWeights[[1]]] - 1;
	    	numberOfHiddenNeurons = Length[hiddenWeights];
	    	numberOfOutputs = Length[networks];;
	    	Return[{numberOfInputs, numberOfHiddenNeurons, numberOfOutputs}]
		]
	];

GetMlp1TrainOptimization[

	(* Returns training set optimization result for mlp1 training.

	   Returns:
	   mlp1TrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlp1InfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlp1InfoList: List with mlp1Info
	   mlp1InfoList[[i]] refers to optimization step i *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,

	(* 0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFraction_?NumberQ,
	
	(* Number of training set optimization steps *)
	numberOfTrainingSetOptimizationSteps_?IntegerQ,

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			crossoverProbability,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			learningParameterMin,
			learningParameterMax,
			maximumNumberOfIterations,
			minimizationPrecision,
			momentumParameter,
			multipleMlp1s,
			mutationProbability,
			numberOfIterationsToImprove,
			populationSize,
			randomValueInitialization,
			activationAndScaling,
			normalizationType,
			optimizationMethod,

			clusterMethod,
			maximumNumberOfEpochs,
			scalarProductMinimumTreshold,
			maximumNumberOfTrialSteps,

			deviationCalculationMethod,
			blackListLength,
			
			i,
			testSet,
			trainingSet,
			clusterRepresentativesRelatedIndexLists,
			trainingSetIndexList,
			testSetIndexList,
			indexLists,
			mlp1Info,
			trainingSetRMSE,
			testSetRMSE,
			pureOutputFunction,
			trainingSetRmseList,
			testSetRmseList,
			trainingAndTestSetList,
			mlp1InfoList,
			selectionResult,
			blackList,
			parallelization,
			lambdaL2Regularization,
			costFunctionType
		},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp1 options *)
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];

		(* Training set optimization options *)
	    deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		
		(* Parallelization options *)
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		clusterRepresentativesRelatedIndexLists = 
			CIP`Cluster`GetClusterRepresentativesRelatedIndexLists[
				dataSet, 
				trainingFraction, 
				ClusterOptionMethod -> clusterMethod,
				ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				UtilityOptionRandomInitializationMode -> randomValueInitialization,
				Mlp1OptionActivationAndScaling -> activationAndScaling
			];
		trainingSetIndexList = clusterRepresentativesRelatedIndexLists[[1]];
		testSetIndexList = clusterRepresentativesRelatedIndexLists[[2]];
		indexLists = clusterRepresentativesRelatedIndexLists[[3]];

		trainingSetRmseList = {};
		testSetRmseList = {};
		trainingAndTestSetList = {};
		mlp1InfoList = {};
		blackList = {};
		Do[
			(* Fit training set and evaluate RMSE *)
			trainingSet = CIP`DataTransformation`GetDataSetPart[dataSet, trainingSetIndexList];
			testSet = CIP`DataTransformation`GetDataSetPart[dataSet, testSetIndexList];
			
			mlp1Info = 
				FitMlp1[
					trainingSet,
					numberOfHiddenNeurons,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
	    			Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp1OptionReportIteration -> 0,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp1OptionActivationAndScaling -> activationAndScaling,
	    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp1OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionsParallelization -> parallelization
				];
				
			trainingSetRMSE = CalculateMlp1DataSetRmse[trainingSet, mlp1Info];
			testSetRMSE = CalculateMlp1DataSetRmse[testSet, mlp1Info];

			(* Set iteration results *)
			AppendTo[trainingSetRmseList, {N[i], trainingSetRMSE}];
			AppendTo[testSetRmseList, {N[i], testSetRMSE}];
			AppendTo[trainingAndTestSetList, {trainingSet, testSet}];
			AppendTo[mlp1InfoList, mlp1Info];
			
			(* Break if necessary *)
			If[i == numberOfTrainingSetOptimizationSteps,
				Break[]
			];

			(* Select new training and test set index lists *)
			pureOutputFunction = Function[input, CalculateMlp1Output[input, mlp1Info]];
			selectionResult = 
				CIP`Utility`SelectNewTrainingAndTestSetIndexLists[
					dataSet, 
					trainingSetIndexList, 
					testSetIndexList,
					blackList,
					indexLists, 
					pureOutputFunction, 
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
					UtilityOptionBlackListLength -> blackListLength
				];
			trainingSetIndexList = selectionResult[[1]];
			testSetIndexList = selectionResult[[2]];
			blackList = selectionResult[[3]],
			
			{i, numberOfTrainingSetOptimizationSteps}
		];
		
		Return[
			{
				trainingSetRmseList,
				testSetRmseList,
				trainingAndTestSetList,
				mlp1InfoList
			}
		]
	];

GetMlp1Weights[

	(* Returns weights of specified network of mlp1Info.

	   Returns:
	   weights: {hiddenWeights, outputWeights}
	   hiddenWeights: Weights from input to hidden units
	   outputWeights : Weights from hidden to output units *)

    
  	(* See "Frequently used data structures" *)
    mlp1Info_,
    
	(* Index of network in mlp1Info *)
    indexOfNetwork_?IntegerQ
    
	] :=
  
	Module[
    
		{
			networks
		},
    
    	networks = mlp1Info[[1]];
   		Return[networks[[indexOfNetwork]]]
	];

GetWeightsStartVariables[

	(* Returns weights start variables, see code. *)

    
	(* Number of hidden units *)
    numberOfInputs_?IntegerQ,

	(* Number of hidden units *)
    numberOfHiddenNeurons_?IntegerQ,

	(* Number of hidden units *)
    numberOfOutputs_?IntegerQ,

    (* Variable for hidden weights *)
    wInputToHidden_,
    
    (* Variable for output weights *)
    wHiddenToOutput_,
    
    (* Hidden weights *)
    hiddenWeights_,
    
    (* Output weights *)
    outputWeights_

	] :=
  
	Module[
    
		{
			factor,
			hiddenWeigthsStartVariables,
			i,
			j,
			k,
			outputWeightsStartVariables
		},

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfInputs + 1];
		hiddenWeigthsStartVariables = {};
		Do[
	    	Do[
				AppendTo[hiddenWeigthsStartVariables, {Subscript[wInputToHidden, j*factor + i], hiddenWeights[[j, i]]}],
	    		
	    		{i, numberOfInputs + 1}
	    	],
	    
	    	{j, numberOfHiddenNeurons}	
		];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons + 1];
		outputWeightsStartVariables = {};
		Do[	
			Do[
				AppendTo[outputWeightsStartVariables, {Subscript[wHiddenToOutput, k*factor + j], outputWeights[[k, j]]}],
		    
		    	{j, numberOfHiddenNeurons + 1}	
			],
				
			{k, numberOfOutputs}
		];
		
		Return[Join[hiddenWeigthsStartVariables, outputWeightsStartVariables]]
	];

GetWeightsVariablesWithoutTrueUnitBias[

	(* Returns weights variables without true unit bias weights, see code. *)

    
	(* Number of hidden units *)
    numberOfInputs_?IntegerQ,

	(* Number of hidden units *)
    numberOfHiddenNeurons_?IntegerQ,

	(* Number of hidden units *)
    numberOfOutputs_?IntegerQ,

    (* Variable for hidden weights *)
    wInputToHidden_,
    
    (* Variable for output weights *)
    wHiddenToOutput_
    
	] :=
  
	Module[
    
		{
			factor,
			weigthVariables,
			i,
			j,
			k
		},

		(* NO true unit bias: Do NOT add 1 to numberOfInputs or numberOfHiddenNeurons *)

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfInputs + 1];
		weigthVariables = {};
		Do[
	    	Do[
				AppendTo[weigthVariables, Subscript[wInputToHidden, j*factor + i]],
	    		
	    		{i, numberOfInputs}
	    	],
	    
	    	{j, numberOfHiddenNeurons}	
		];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons + 1];
		Do[	
			Do[
				AppendTo[weigthVariables, Subscript[wHiddenToOutput, k*factor + j]],
		    
		    	{j, numberOfHiddenNeurons}	
			],
				
			{k, numberOfOutputs}
		];
		
		Return[weigthVariables]
	];

GetWeightsVariables[

	(* Returns weights variables, see code. *)

    
	(* Number of hidden units *)
    numberOfInputs_?IntegerQ,

	(* Number of hidden units *)
    numberOfHiddenNeurons_?IntegerQ,

	(* Number of hidden units *)
    numberOfOutputs_?IntegerQ,

    (* Variable for hidden weights *)
    wInputToHidden_,
    
    (* Variable for output weights *)
    wHiddenToOutput_

	] :=
  
	Module[
    
		{
			factor,
			hiddenWeightsVariables,
			i,
			j,
			k,
			outputWeightsVariables
		},

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfInputs + 1];
		hiddenWeightsVariables = 
			Table[
				Table[
					Subscript[wInputToHidden, j*factor + i], 
					
					{i, numberOfInputs + 1}
				], 
				
				{j, numberOfHiddenNeurons}
			];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons + 1];
		outputWeightsVariables = 
			Table[
				Table[
					Subscript[wHiddenToOutput, k*factor + j], 
					
					{j, numberOfHiddenNeurons + 1}
				], 
				
				{k, numberOfOutputs}
			];
		
		Return[{hiddenWeightsVariables, outputWeightsVariables}]		
	];

MutateChromosome[

	(* Returns a mutated chromosome

	   Returns:
	   chromosome: {hiddenWeights, outputWeights} *)

    
	(* Chromosome has form: {hiddenWeights, outputWeights} *)
    chromosome_,
    
    mutationProbability_?NumberQ,
    
    (* Mutated value will be in interval -mutatedValueBound <= mutated value <= +mutatedValueBound *)
    mutatedValueBound_?NumberQ
    
	] := 
	
	Module[
    
		{
			randomNumber,
			randomPosition1,
			randomPosition2,
			randomPosition3,
			mutatedChromosome
		},
    
	    mutatedChromosome = chromosome;
	    randomNumber = RandomReal[];
    
		(* Mutate with given mutation probability *)
		If[randomNumber <= mutationProbability,
			
			(* True -> Mutate *)
			(* First choose layer : Hidden or output weights *)
			randomPosition1 = RandomInteger[{1, 2}];
			(* Second choose neuron : Hidden or output neuron *)
			randomPosition2 = RandomInteger[{1, Length[chromosome[[randomPosition1]]]}];
			(* Third choose specific weight of hidden or output neuron *)
			randomPosition3 = RandomInteger[{1, Length[chromosome[[randomPosition1, randomPosition2]]]}];
			mutatedChromosome = ReplacePart[mutatedChromosome, {randomPosition1, randomPosition2, randomPosition3} -> RandomReal[{-mutatedValueBound, mutatedValueBound}]];
			Return[mutatedChromosome],
      
			(* False -> Do nothing and return input *)
			Return[chromosome]
		]
	];

ScanClassTrainingWithMlp1[

	(* Scans training and test set for different training fractions based on method FitMlp1, see code.
	
	   Returns:
	   mlp1ClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlp1Info1}, {trainingAndTestSet2, mlp1Info2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, classification result in percent for training set}, {trainingFraction, classification result in percent for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)

	
	(* classificationDataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} 
	   outputValue: 0/1
	   Data set must be a classification data set, i.e. the output components must 0/1 code a class,
	   i.e. class 4 of 5 must be coded {0, 0, 0, 1, 0} *)
    classificationDataSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			crossoverProbability,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			learningParameterMin,
			learningParameterMax,
			maximumNumberOfIterations,
			minimizationPrecision,
			momentumParameter,
			multipleMlp1s,
			mutationProbability,
			numberOfIterationsToImprove,
			populationSize,
			optimizationMethod,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
	    	activationAndScaling,
			normalizationType,

			numberOfTrainingSetOptimizationSteps,
			deviationCalculationMethod,
			blackListLength,

			parallelization,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp1 options *)
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* Training set optimization options *)
		numberOfTrainingSetOptimizationSteps = UtilityOptionOptimizationSteps/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    
		(* Parallelization options *)
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				ScanClassTrainingWithMlp1PC[
					classificationDataSet,
					numberOfHiddenNeurons,
					trainingFractionList,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
				    Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
					ClusterOptionMethod -> clusterMethod,
				    ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				    ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				    ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
			   	    Mlp1OptionActivationAndScaling -> activationAndScaling,
			   	    Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
			   	    Mlp1OptionCostFunctionType -> costFunctionType,
			   	    DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
				    UtilityOptionBlackListLength -> blackListLength
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				ScanClassTrainingWithMlp1SC[
					classificationDataSet,
					numberOfHiddenNeurons,
					trainingFractionList,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
				    Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
					ClusterOptionMethod -> clusterMethod,
				    ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				    ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				    ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
			   	    Mlp1OptionActivationAndScaling -> activationAndScaling,
			   	    Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
			   	    Mlp1OptionCostFunctionType -> costFunctionType,
			   	    DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
				    UtilityOptionBlackListLength -> blackListLength
				]
			]
		]
	];

ScanClassTrainingWithMlp1SC[

	(* Scans training and test set for different training fractions based on method FitMlp1, see code.
	
	   Returns:
	   mlp1ClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlp1Info1}, {trainingAndTestSet2, mlp1Info2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, classification result in percent for training set}, {trainingFraction, classification result in percent for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)

	
	(* classificationDataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} 
	   outputValue: 0/1
	   Data set must be a classification data set, i.e. the output components must 0/1 code a class,
	   i.e. class 4 of 5 must be coded {0, 0, 0, 1, 0} *)
    classificationDataSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			crossoverProbability,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			learningParameterMin,
			learningParameterMax,
			maximumNumberOfIterations,
			minimizationPrecision,
			momentumParameter,
			multipleMlp1s,
			mutationProbability,
			numberOfIterationsToImprove,
			populationSize,
			optimizationMethod,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
	    	activationAndScaling,
			normalizationType,

			numberOfTrainingSetOptimizationSteps,
			deviationCalculationMethod,
			blackListLength,

			i,
			scanReport,
			trainingAndTestSetsInfo,
			currentTrainingAndTestSet,
			currentTrainingSet,
			currentTestSet,
			currentMlp1Info,
			pureFunction,
			trainingSetCorrectClassificationInPercent,
			testSetCorrectClassificationInPercent,
			mlp1TrainOptimization,
			mlp1InfoList,
			trainingAndTestSetList,
			bestIndex,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp1 options *)
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* Training set optimization options *)
		numberOfTrainingSetOptimizationSteps = UtilityOptionOptimizationSteps/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];

		scanReport = {};
		trainingAndTestSetsInfo = {};
		Do[
			If[numberOfTrainingSetOptimizationSteps > 0,
				
				(* ------------------------------------------------------------------------------------------------------ *)
				(* Training set optimization *)
				mlp1TrainOptimization = 
					GetMlp1TrainOptimization[
						classificationDataSet, 
						numberOfHiddenNeurons, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						Mlp1OptionMultipleMlp1s -> multipleMlp1s,
		    			Mlp1OptionOptimizationMethod -> optimizationMethod,
						Mlp1OptionInitialWeights -> initialWeights,
						Mlp1OptionInitialNetworks -> initialNetworks,
						Mlp1OptionWeightsValueLimit -> weightsValueLimit,
						Mlp1OptionMinimizationPrecision -> minimizationPrecision,
						Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
						Mlp1OptionLearningParameterMin -> learningParameterMin,
						Mlp1OptionLearningParameterMax -> learningParameterMax,
						Mlp1OptionMomentumParameter -> momentumParameter,
						Mlp1OptionPopulationSize -> populationSize,
						Mlp1OptionCrossoverProbability -> crossoverProbability,
						Mlp1OptionMutationProbability -> mutationProbability,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						Mlp1OptionActivationAndScaling -> activationAndScaling,
						Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
						Mlp1OptionCostFunctionType -> costFunctionType,
						DataTransformationOptionNormalizationType -> normalizationType,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestMlp1ClassOptimization[mlp1TrainOptimization];
				trainingAndTestSetList = mlp1TrainOptimization[[3]];
				mlp1InfoList = mlp1TrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlp1Info = mlp1InfoList[[bestIndex]],
				
				(* ------------------------------------------------------------------------------------------------------ *)
				(* No training set optimization *)
				currentTrainingAndTestSet = 
					CIP`Cluster`GetClusterBasedTrainingAndTestSet[
						classificationDataSet,
						trainingFractionList[[i]],
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
						DataTransformationOptionTargetInterval -> activationAndScaling[[2, 1]]
					];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlp1Info = 
					FitMlp1[
						currentTrainingSet,
						numberOfHiddenNeurons,
						Mlp1OptionMultipleMlp1s -> multipleMlp1s,
		    			Mlp1OptionOptimizationMethod -> optimizationMethod,
						Mlp1OptionInitialWeights -> initialWeights,
						Mlp1OptionInitialNetworks -> initialNetworks,
						Mlp1OptionWeightsValueLimit -> weightsValueLimit,
						Mlp1OptionMinimizationPrecision -> minimizationPrecision,
						Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
			 			Mlp1OptionReportIteration -> 0,
						Mlp1OptionLearningParameterMin -> learningParameterMin,
						Mlp1OptionLearningParameterMax -> learningParameterMax,
						Mlp1OptionMomentumParameter -> momentumParameter,
						Mlp1OptionPopulationSize -> populationSize,
						Mlp1OptionCrossoverProbability -> crossoverProbability,
						Mlp1OptionMutationProbability -> mutationProbability,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			Mlp1OptionActivationAndScaling -> activationAndScaling,
		    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
		    			Mlp1OptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					]
			];
			
			pureFunction = Function[inputs, CalculateMlp1ClassNumbers[inputs, currentMlp1Info]];
			trainingSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTrainingSet, pureFunction];
			testSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTestSet, pureFunction];
			AppendTo[trainingAndTestSetsInfo, {currentTrainingAndTestSet, currentMlp1Info}];
			AppendTo[
				scanReport, 
				{
					{trainingFractionList[[i]], trainingSetCorrectClassificationInPercent},
					{trainingFractionList[[i]], testSetCorrectClassificationInPercent}
				}
			],
			
			{i, Length[trainingFractionList]}
		];

		Return[{trainingAndTestSetsInfo, scanReport}]
	];

ScanClassTrainingWithMlp1PC[

	(* Scans training and test set for different training fractions based on method FitMlp1, see code.
	
	   Returns:
	   mlp1ClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlp1Info1}, {trainingAndTestSet2, mlp1Info2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, classification result in percent for training set}, {trainingFraction, classification result in percent for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)

	
	(* classificationDataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} 
	   outputValue: 0/1
	   Data set must be a classification data set, i.e. the output components must 0/1 code a class,
	   i.e. class 4 of 5 must be coded {0, 0, 0, 1, 0} *)
    classificationDataSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			crossoverProbability,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			learningParameterMin,
			learningParameterMax,
			maximumNumberOfIterations,
			minimizationPrecision,
			momentumParameter,
			multipleMlp1s,
			mutationProbability,
			numberOfIterationsToImprove,
			populationSize,
			optimizationMethod,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
	    	activationAndScaling,
			normalizationType,

			numberOfTrainingSetOptimizationSteps,
			deviationCalculationMethod,
			blackListLength,

			i,
			scanReport,
			trainingAndTestSetsInfo,
			currentTrainingAndTestSet,
			currentTrainingSet,
			currentTestSet,
			currentMlp1Info,
			pureFunction,
			trainingSetCorrectClassificationInPercent,
			testSetCorrectClassificationInPercent,
			mlp1TrainOptimization,
			mlp1InfoList,
			trainingAndTestSetList,
			bestIndex,
			listOfTrainingAndTestSetsInfoAndScanReport,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp1 options *)
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* Training set optimization options *)
		numberOfTrainingSetOptimizationSteps = UtilityOptionOptimizationSteps/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];

		ParallelNeeds[{"CIP`Mlp1`", "CIP`Cluster`", "CIP`DataTransformation`", "CIP`Utility`", "Combinatorica`"}];
		DistributeDefinitions[numberOfTrainingSetOptimizationSteps, multipleMlp1s, optimizationMethod, initialWeights,
						initialNetworks, weightsValueLimit, minimizationPrecision, maximumNumberOfIterations, numberOfIterationsToImprove, 
						learningParameterMin, learningParameterMax, momentumParameter, populationSize, crossoverProbability, mutationProbability,
						clusterMethod, maximumNumberOfEpochs, scalarProductMinimumTreshold, maximumNumberOfTrialSteps, activationAndScaling, 
						normalizationType, randomValueInitialization, deviationCalculationMethod, blackListLength, lambdaL2Regularization, 
						costFunctionType];
		
		listOfTrainingAndTestSetsInfoAndScanReport = ParallelTable[
			If[numberOfTrainingSetOptimizationSteps > 0,
				
				(* ------------------------------------------------------------------------------------------------------ *)
				(* Training set optimization *)
				
				mlp1TrainOptimization = 
					GetMlp1TrainOptimization[
						classificationDataSet, 
						numberOfHiddenNeurons, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						Mlp1OptionMultipleMlp1s -> multipleMlp1s,
		    			Mlp1OptionOptimizationMethod -> optimizationMethod,
						Mlp1OptionInitialWeights -> initialWeights,
						Mlp1OptionInitialNetworks -> initialNetworks,
						Mlp1OptionWeightsValueLimit -> weightsValueLimit,
						Mlp1OptionMinimizationPrecision -> minimizationPrecision,
						Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
						Mlp1OptionLearningParameterMin -> learningParameterMin,
						Mlp1OptionLearningParameterMax -> learningParameterMax,
						Mlp1OptionMomentumParameter -> momentumParameter,
						Mlp1OptionPopulationSize -> populationSize,
						Mlp1OptionCrossoverProbability -> crossoverProbability,
						Mlp1OptionMutationProbability -> mutationProbability,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						Mlp1OptionActivationAndScaling -> activationAndScaling,
						Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
						Mlp1OptionCostFunctionType -> costFunctionType,
						DataTransformationOptionNormalizationType -> normalizationType,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestMlp1ClassOptimization[mlp1TrainOptimization];				
				trainingAndTestSetList = mlp1TrainOptimization[[3]];
				mlp1InfoList = mlp1TrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlp1Info = mlp1InfoList[[bestIndex]],
				
				(* ------------------------------------------------------------------------------------------------------ *)
				(* No training set optimization *)
				currentTrainingAndTestSet = 
					CIP`Cluster`GetClusterBasedTrainingAndTestSet[
						classificationDataSet,
						trainingFractionList[[i]],
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
						DataTransformationOptionTargetInterval -> activationAndScaling[[2, 1]]
					];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				
				currentMlp1Info = 
					FitMlp1[
						currentTrainingSet,
						numberOfHiddenNeurons,
						Mlp1OptionMultipleMlp1s -> multipleMlp1s,
		    			Mlp1OptionOptimizationMethod -> optimizationMethod,
						Mlp1OptionInitialWeights -> initialWeights,
						Mlp1OptionInitialNetworks -> initialNetworks,
						Mlp1OptionWeightsValueLimit -> weightsValueLimit,
						Mlp1OptionMinimizationPrecision -> minimizationPrecision,
						Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
			 			Mlp1OptionReportIteration -> 0,
						Mlp1OptionLearningParameterMin -> learningParameterMin,
						Mlp1OptionLearningParameterMax -> learningParameterMax,
						Mlp1OptionMomentumParameter -> momentumParameter,
						Mlp1OptionPopulationSize -> populationSize,
						Mlp1OptionCrossoverProbability -> crossoverProbability,
						Mlp1OptionMutationProbability -> mutationProbability,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			Mlp1OptionActivationAndScaling -> activationAndScaling,
		    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
		    			Mlp1OptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					];
			];
			
			pureFunction = Function[inputs, CalculateMlp1ClassNumbers[inputs, currentMlp1Info]];
			trainingSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTrainingSet, pureFunction];
			testSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTestSet, pureFunction];
			{
				{currentTrainingAndTestSet, currentMlp1Info},
				
				{
					{trainingFractionList[[i]], trainingSetCorrectClassificationInPercent},
					{trainingFractionList[[i]], testSetCorrectClassificationInPercent}
				}
			},
			
			{i, Length[trainingFractionList]}
		];	

		trainingAndTestSetsInfo = Table[listOfTrainingAndTestSetsInfoAndScanReport[[i,1]], {i, Length[trainingFractionList]}];
		scanReport = Table[listOfTrainingAndTestSetsInfoAndScanReport[[i,2]], {i, Length[trainingFractionList]}];
		
		Return[{trainingAndTestSetsInfo, scanReport}]
	];

ScanRegressTrainingWithMlp1[

	(* Scans training and test set for different training fractions based on method FitMlp1, see code.
	
	   Returns:
	   mlp1RegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlp1Info1}, {trainingAndTestSet2, mlp1Info2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, RMSE for training set}, {trainingFraction, RMSE for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)

	
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output: {outputComponent1, outputComponent2, ...}
	   NOTE: Data set MUST be in original units *)
	dataSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			crossoverProbability,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			learningParameterMin,
			learningParameterMax,
			maximumNumberOfIterations,
			minimizationPrecision,
			momentumParameter,
			multipleMlp1s,
			mutationProbability,
			numberOfIterationsToImprove,
			populationSize,
			optimizationMethod,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
	    	activationAndScaling,
			normalizationType,

			numberOfTrainingSetOptimizationSteps,
			deviationCalculationMethod,
			blackListLength,

			parallelization,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp1 options *)
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* Training set optimization options *)
		numberOfTrainingSetOptimizationSteps = UtilityOptionOptimizationSteps/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    
		(* Parallelization options *)
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				ScanRegressTrainingWithMlp1PC[
					dataSet,
					numberOfHiddenNeurons,
					trainingFractionList,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
				    Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
					ClusterOptionMethod -> clusterMethod,
				    ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				    ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				    ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
			   	    Mlp1OptionActivationAndScaling -> activationAndScaling,
			   	    Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
			   	    Mlp1OptionCostFunctionType -> costFunctionType,
			   	    DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
				    UtilityOptionBlackListLength -> blackListLength
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				ScanRegressTrainingWithMlp1SC[
					dataSet,
					numberOfHiddenNeurons,
					trainingFractionList,
					Mlp1OptionMultipleMlp1s -> multipleMlp1s,
				    Mlp1OptionOptimizationMethod -> optimizationMethod,
					Mlp1OptionInitialWeights -> initialWeights,
					Mlp1OptionInitialNetworks -> initialNetworks,
					Mlp1OptionWeightsValueLimit -> weightsValueLimit,
					Mlp1OptionMinimizationPrecision -> minimizationPrecision,
					Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
					Mlp1OptionLearningParameterMin -> learningParameterMin,
					Mlp1OptionLearningParameterMax -> learningParameterMax,
					Mlp1OptionMomentumParameter -> momentumParameter,
					Mlp1OptionPopulationSize -> populationSize,
					Mlp1OptionCrossoverProbability -> crossoverProbability,
					Mlp1OptionMutationProbability -> mutationProbability,
					ClusterOptionMethod -> clusterMethod,
				    ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				    ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				    ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
			   	    Mlp1OptionActivationAndScaling -> activationAndScaling,
			   	    Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
			   	    Mlp1OptionCostFunctionType -> costFunctionType,
			   	    DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
				    UtilityOptionBlackListLength -> blackListLength
				]
			]
		]
	];

ScanRegressTrainingWithMlp1SC[

	(* Scans training and test set for different training fractions based on method FitMlp1, see code.
	
	   Returns:
	   mlp1RegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlp1Info1}, {trainingAndTestSet2, mlp1Info2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, RMSE for training set}, {trainingFraction, RMSE for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)

	
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output: {outputComponent1, outputComponent2, ...}
	   NOTE: Data set MUST be in original units *)
	dataSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			crossoverProbability,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			learningParameterMin,
			learningParameterMax,
			maximumNumberOfIterations,
			minimizationPrecision,
			momentumParameter,
			multipleMlp1s,
			mutationProbability,
			numberOfIterationsToImprove,
			populationSize,
			optimizationMethod,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
	    	activationAndScaling,
			normalizationType,

			numberOfTrainingSetOptimizationSteps,
			deviationCalculationMethod,
			blackListLength,

			i,
			scanReport,
			trainingAndTestSetsInfo,
			currentTrainingAndTestSet,
			currentTrainingSet,
			currentTestSet,
			currentMlp1Info,
			pureFunction,
			trainingSetRMSE,
			testSetRMSE,
			mlp1TrainOptimization,
			trainingAndTestSetList,
			mlp1InfoList,
			bestIndex,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp1 options *)
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* Training set optimization options *)
		numberOfTrainingSetOptimizationSteps = UtilityOptionOptimizationSteps/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];

		scanReport = {};
		trainingAndTestSetsInfo = {};
		Do[
			If[numberOfTrainingSetOptimizationSteps > 0,
				
				(* ------------------------------------------------------------------------------------------------------ *)
				(* Training set optimization *)
				mlp1TrainOptimization = 
					GetMlp1TrainOptimization[
						dataSet, 
						numberOfHiddenNeurons, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						Mlp1OptionMultipleMlp1s -> multipleMlp1s,
		    			Mlp1OptionOptimizationMethod -> optimizationMethod,
						Mlp1OptionInitialWeights -> initialWeights,
						Mlp1OptionInitialNetworks -> initialNetworks,
						Mlp1OptionWeightsValueLimit -> weightsValueLimit,
						Mlp1OptionMinimizationPrecision -> minimizationPrecision,
						Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
						Mlp1OptionLearningParameterMin -> learningParameterMin,
						Mlp1OptionLearningParameterMax -> learningParameterMax,
						Mlp1OptionMomentumParameter -> momentumParameter,
						Mlp1OptionPopulationSize -> populationSize,
						Mlp1OptionCrossoverProbability -> crossoverProbability,
						Mlp1OptionMutationProbability -> mutationProbability,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						Mlp1OptionActivationAndScaling -> activationAndScaling,
						Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
						Mlp1OptionCostFunctionType -> costFunctionType,
						DataTransformationOptionNormalizationType -> normalizationType,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestMlp1RegressOptimization[mlp1TrainOptimization];
				trainingAndTestSetList = mlp1TrainOptimization[[3]];
				mlp1InfoList = mlp1TrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlp1Info = mlp1InfoList[[bestIndex]],
				
				(* ------------------------------------------------------------------------------------------------------ *)
				(* No training set optimization *)
				currentTrainingAndTestSet = 
					CIP`Cluster`GetClusterBasedTrainingAndTestSet[
						dataSet,
						trainingFractionList[[i]],
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
						DataTransformationOptionTargetInterval -> activationAndScaling[[2, 1]]
				];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlp1Info = 
					FitMlp1[
						currentTrainingSet,
						numberOfHiddenNeurons,
						Mlp1OptionMultipleMlp1s -> multipleMlp1s,
		    			Mlp1OptionOptimizationMethod -> optimizationMethod,
						Mlp1OptionInitialWeights -> initialWeights,
						Mlp1OptionInitialNetworks -> initialNetworks,
						Mlp1OptionWeightsValueLimit -> weightsValueLimit,
						Mlp1OptionMinimizationPrecision -> minimizationPrecision,
						Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
			 			Mlp1OptionReportIteration -> 0,
						Mlp1OptionLearningParameterMin -> learningParameterMin,
						Mlp1OptionLearningParameterMax -> learningParameterMax,
						Mlp1OptionMomentumParameter -> momentumParameter,
						Mlp1OptionPopulationSize -> populationSize,
						Mlp1OptionCrossoverProbability -> crossoverProbability,
						Mlp1OptionMutationProbability -> mutationProbability,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			Mlp1OptionActivationAndScaling -> activationAndScaling,
		    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
		    			Mlp1OptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					]
			];
			
			pureFunction = Function[inputs, CalculateMlp1Outputs[inputs, currentMlp1Info]];
			trainingSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTrainingSet, pureFunction]];
			testSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTestSet, pureFunction]];
			AppendTo[trainingAndTestSetsInfo, {currentTrainingAndTestSet, currentMlp1Info}];
			AppendTo[
				scanReport, 
				{
					{trainingFractionList[[i]], trainingSetRMSE},
					{trainingFractionList[[i]], testSetRMSE}
				}
			],
			
			{i, Length[trainingFractionList]}
		];

		Return[{trainingAndTestSetsInfo, scanReport}]
	];

ScanRegressTrainingWithMlp1PC[

	(* Scans training and test set for different training fractions based on method FitMlp1, see code.
	
	   Returns:
	   mlp1RegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlp1Info1}, {trainingAndTestSet2, mlp1Info2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, RMSE for training set}, {trainingFraction, RMSE for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)

	
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output: {outputComponent1, outputComponent2, ...}
	   NOTE: Data set MUST be in original units *)
	dataSet_,

	(* Number of hidden neurons *)
	numberOfHiddenNeurons_?IntegerQ,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			crossoverProbability,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			learningParameterMin,
			learningParameterMax,
			maximumNumberOfIterations,
			minimizationPrecision,
			momentumParameter,
			multipleMlp1s,
			mutationProbability,
			numberOfIterationsToImprove,
			populationSize,
			optimizationMethod,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
	    	activationAndScaling,
			normalizationType,

			numberOfTrainingSetOptimizationSteps,
			deviationCalculationMethod,
			blackListLength,

			i,
			scanReport,
			trainingAndTestSetsInfo,
			currentTrainingAndTestSet,
			currentTrainingSet,
			currentTestSet,
			currentMlp1Info,
			pureFunction,
			trainingSetRMSE,
			testSetRMSE,
			mlp1TrainOptimization,
			trainingAndTestSetList,
			mlp1InfoList,
			bestIndex,
			listOfTrainingAndTestSetsInfoAndScanReport,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp1 options *)
		multipleMlp1s = Mlp1OptionMultipleMlp1s/.{opts}/.Options[Mlp1OptionsTraining];
	    optimizationMethod = Mlp1OptionOptimizationMethod/.{opts}/.Options[Mlp1OptionsTraining];
		initialWeights = Mlp1OptionInitialWeights/.{opts}/.Options[Mlp1OptionsOptimization];
		initialNetworks = Mlp1OptionInitialNetworks/.{opts}/.Options[Mlp1OptionsOptimization];
		weightsValueLimit = Mlp1OptionWeightsValueLimit/.{opts}/.Options[Mlp1OptionsOptimization];
		minimizationPrecision = Mlp1OptionMinimizationPrecision/.{opts}/.Options[Mlp1OptionsOptimization];
		maximumNumberOfIterations = Mlp1OptionMaximumIterations/.{opts}/.Options[Mlp1OptionsOptimization];
		numberOfIterationsToImprove = Mlp1OptionIterationsToImprove/.{opts}/.Options[Mlp1OptionsOptimization];
		learningParameterMin = Mlp1OptionLearningParameterMin/.{opts}/.Options[Mlp1OptionsBackpropagation];
		learningParameterMax = Mlp1OptionLearningParameterMax/.{opts}/.Options[Mlp1OptionsBackpropagation];
		momentumParameter = Mlp1OptionMomentumParameter/.{opts}/.Options[Mlp1OptionsBackpropagation];
		populationSize = Mlp1OptionPopulationSize/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		crossoverProbability = Mlp1OptionCrossoverProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];
		mutationProbability = Mlp1OptionMutationProbability/.{opts}/.Options[Mlp1OptionsGeneticAlgorithm];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = Mlp1OptionActivationAndScaling/.{opts}/.Options[Mlp1OptionsTraining];
	    lambdaL2Regularization = Mlp1OptionLambdaL2Regularization/.{opts}/.Options[Mlp1OptionsTraining];
	    costFunctionType = Mlp1OptionCostFunctionType/.{opts}/.Options[Mlp1OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* Training set optimization options *)
		numberOfTrainingSetOptimizationSteps = UtilityOptionOptimizationSteps/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		
		ParallelNeeds[{"CIP`Mlp1`", "CIP`Cluster`", "CIP`DataTransformation`", "CIP`Utility`", "Combinatorica`"}];
		DistributeDefinitions[numberOfTrainingSetOptimizationSteps, multipleMlp1s, optimizationMethod, initialWeights,
						initialNetworks, weightsValueLimit, minimizationPrecision, maximumNumberOfIterations, numberOfIterationsToImprove, 
						learningParameterMin, learningParameterMax, momentumParameter, populationSize, crossoverProbability, mutationProbability,
						clusterMethod, maximumNumberOfEpochs, scalarProductMinimumTreshold, maximumNumberOfTrialSteps, activationAndScaling, 
						normalizationType, randomValueInitialization, deviationCalculationMethod, blackListLength, lambdaL2Regularization];
			
		listOfTrainingAndTestSetsInfoAndScanReport = ParallelTable[
			If[numberOfTrainingSetOptimizationSteps > 0,
				
				(* ------------------------------------------------------------------------------------------------------ *)
				(* Training set optimization *)
				
				mlp1TrainOptimization = 
					GetMlp1TrainOptimization[
						dataSet, 
						numberOfHiddenNeurons, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						Mlp1OptionMultipleMlp1s -> multipleMlp1s,
		    			Mlp1OptionOptimizationMethod -> optimizationMethod,
						Mlp1OptionInitialWeights -> initialWeights,
						Mlp1OptionInitialNetworks -> initialNetworks,
						Mlp1OptionWeightsValueLimit -> weightsValueLimit,
						Mlp1OptionMinimizationPrecision -> minimizationPrecision,
						Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
						Mlp1OptionLearningParameterMin -> learningParameterMin,
						Mlp1OptionLearningParameterMax -> learningParameterMax,
						Mlp1OptionMomentumParameter -> momentumParameter,
						Mlp1OptionPopulationSize -> populationSize,
						Mlp1OptionCrossoverProbability -> crossoverProbability,
						Mlp1OptionMutationProbability -> mutationProbability,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						Mlp1OptionActivationAndScaling -> activationAndScaling,
						Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
						Mlp1OptionCostFunctionType -> costFunctionType,
						DataTransformationOptionNormalizationType -> normalizationType,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
					
				bestIndex = GetBestMlp1RegressOptimization[mlp1TrainOptimization];
				trainingAndTestSetList = mlp1TrainOptimization[[3]];
				mlp1InfoList = mlp1TrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlp1Info = mlp1InfoList[[bestIndex]],
				
				(* ------------------------------------------------------------------------------------------------------ *)
				(* No training set optimization *)
				currentTrainingAndTestSet = 
					CIP`Cluster`GetClusterBasedTrainingAndTestSet[
						dataSet,
						trainingFractionList[[i]],
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
						DataTransformationOptionTargetInterval -> activationAndScaling[[2, 1]]
				];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				
				currentMlp1Info = 
					FitMlp1[
						currentTrainingSet,
						numberOfHiddenNeurons,
						Mlp1OptionMultipleMlp1s -> multipleMlp1s,
	   		 			Mlp1OptionOptimizationMethod -> optimizationMethod,
						Mlp1OptionInitialWeights -> initialWeights,
						Mlp1OptionInitialNetworks -> initialNetworks,
						Mlp1OptionWeightsValueLimit -> weightsValueLimit,
						Mlp1OptionMinimizationPrecision -> minimizationPrecision,
						Mlp1OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp1OptionIterationsToImprove -> numberOfIterationsToImprove,
			 			Mlp1OptionReportIteration -> 0,
						Mlp1OptionLearningParameterMin -> learningParameterMin,
						Mlp1OptionLearningParameterMax -> learningParameterMax,
						Mlp1OptionMomentumParameter -> momentumParameter,
						Mlp1OptionPopulationSize -> populationSize,
						Mlp1OptionCrossoverProbability -> crossoverProbability,
						Mlp1OptionMutationProbability -> mutationProbability,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			Mlp1OptionActivationAndScaling -> activationAndScaling,
		    			Mlp1OptionLambdaL2Regularization -> lambdaL2Regularization,
		    			Mlp1OptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					];
			];
			
			pureFunction = Function[inputs, CalculateMlp1Outputs[inputs, currentMlp1Info]];
			trainingSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTrainingSet, pureFunction]];
			testSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTestSet, pureFunction]];
			{
				{currentTrainingAndTestSet, currentMlp1Info},
				
				{
					{trainingFractionList[[i]], trainingSetRMSE},
					{trainingFractionList[[i]], testSetRMSE}
				}
			},
			
			{i, Length[trainingFractionList]}
		];

		trainingAndTestSetsInfo = Table[listOfTrainingAndTestSetsInfoAndScanReport[[i,1]], {i, Length[trainingFractionList]}];
		scanReport = Table[listOfTrainingAndTestSetsInfoAndScanReport[[i,2]], {i, Length[trainingFractionList]}];
	
		Return[{trainingAndTestSetsInfo, scanReport}]
		
	];

ShowMlp1Output2D[

	(* Shows 2D mlp1 output.

	   Returns: Nothing *)


    (* Index of input neuron that receives argumentValue *)
    indexOfInput_?IntegerQ,

    (* Index of output neuron that returns function value *)
    indexOfFunctionValueOutput_?IntegerQ,
    
    (* Mlp1 input in original units: 
       inputsInOriginalUnits = {inputValue1, inputValue2, ...} 
       Value of input neuron with specified index (indexOfInput) is replaced by argumentValue *)
    input_/;VectorQ[input, NumberQ],
    
    (* Arguments to be displayed as points:
       arguments: {argumentValue1, argumentValue2, ...} *)
    arguments_/;VectorQ[arguments, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlp1Info_
      
	] :=
    
	Module[
      
		{	
			dataSetScaleInfo,
			inputsMinMaxList,
			labels,
			points,
			xMin,
			xMax
		},
	
		If[Length[arguments] > 0,
			
			points = 
				Table[
					{
						arguments[[i]], 
						CalculateMlp1Value2D[arguments[[i]], indexOfInput, indexOfFunctionValueOutput, input, mlp1Info]
					},
						
					{i, Length[arguments]}
				],
				
			points = {}
		];
		
		dataSetScaleInfo = mlp1Info[[2]];
		inputsMinMaxList = dataSetScaleInfo[[1, 1]];
		xMin = inputsMinMaxList[[indexOfInput, 1]];
		xMax = inputsMinMaxList[[indexOfInput, 2]];
		
		labels = 
			{
				StringJoin["Argument Value of Input ", ToString[indexOfInput]],
				StringJoin["Value of Output ", ToString[indexOfFunctionValueOutput]],
				"Mlp1 Output"
			};
		Print[
			CIP`Graphics`PlotPoints2DAboveFunction[
				points, 
				Function[x, CalculateMlp1Value2D[x, indexOfInput, indexOfFunctionValueOutput, input, mlp1Info]], 
				labels,
				GraphicsOptionArgumentRange2D -> {xMin, xMax}
			]
		]
	];

ShowMlp1Output3D[

	(* Shows 3D mlp1 output.

	   Returns: Graphics3D *)


    (* Index of input neuron that receives argumentValue1 *)
    indexOfInput1_?IntegerQ,

    (* Index of input neuron that receives argumentValue2 *)
    indexOfInput2_?IntegerQ,

    (* Index of output neuron that returns function value *)
    indexOfFunctionValueOutput_?IntegerQ,
    
    (* Mlp1 input in original units: 
       inputsInOriginalUnits = {inputValue1, inputValue2, ...} 
       Value of input neuron with specified index (indexOfInput) is replaced by argumentValue *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlp1Info_,
    
	(* Options *)
	opts___
      
	] :=
    
	Module[
      
		{	
			GraphicsOptionDisplayFunction,
			GraphicsOptionViewPoint3D,
			GraphicsOptionNumberOfPlotPoints,
			GraphicsOptionColorFunction,
			GraphicsOptionIsMesh,
			GraphicsOptionMeshStyle,
			GraphicsOptionPlotStyle3D,
			dataSetScaleInfo,
			inputsMinMaxList,
			labels,
			x1Min,
			x1Max,
			x2Min,
			x2Max
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    GraphicsOptionDisplayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    GraphicsOptionViewPoint3D = GraphicsOptionViewPoint3D/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    GraphicsOptionNumberOfPlotPoints = GraphicsOptionNumberOfPlotPoints/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    GraphicsOptionColorFunction = GraphicsOptionColorFunction/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    GraphicsOptionIsMesh = GraphicsOptionIsMesh/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    GraphicsOptionMeshStyle = GraphicsOptionMeshStyle/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    GraphicsOptionPlotStyle3D = GraphicsOptionPlotStyle3D/.{opts}/.Options[GraphicsOptionsGraphics3D];
	
		dataSetScaleInfo = mlp1Info[[2]];
		inputsMinMaxList = dataSetScaleInfo[[1, 1]];
		x1Min = inputsMinMaxList[[indexOfInput1, 1]];
		x1Max = inputsMinMaxList[[indexOfInput1, 2]];
		x2Min = inputsMinMaxList[[indexOfInput2, 1]];
		x2Max = inputsMinMaxList[[indexOfInput2, 2]];
		labels = 
			{
				StringJoin["In ", ToString[indexOfInput1]],
				StringJoin["In ", ToString[indexOfInput2]],
				StringJoin["Out ", ToString[indexOfFunctionValueOutput]]
			};
		
		Return[
			CIP`Graphics`PlotFunction3D[
				Function[{x1, x2}, CalculateMlp1Value3D[x1, x2, indexOfInput1, indexOfInput2, indexOfFunctionValueOutput, input, mlp1Info]], 
				{x1Min, x1Max}, 
				{x2Min, x2Max}, 
				labels, 
				GraphicsOptionViewPoint3D -> GraphicsOptionViewPoint3D,
				GraphicsOptionNumberOfPlotPoints -> GraphicsOptionNumberOfPlotPoints,
				GraphicsOptionColorFunction -> GraphicsOptionColorFunction,
				GraphicsOptionIsMesh -> GraphicsOptionIsMesh,
				GraphicsOptionMeshStyle -> GraphicsOptionMeshStyle,
				GraphicsOptionPlotStyle3D -> GraphicsOptionPlotStyle3D,
				GraphicsOptionDisplayFunction -> GraphicsOptionDisplayFunction
			]
		]
	];

ShowMlp1ClassificationResult[

	(* Shows result of mlp1 classification for training and test set according to named property list.

	   Returns: Nothing *)


	(* Properties to be analyzed: 
	   Full list: 
	   {
	       "CorrectClassification",
	       "CorrectClassificationPerClass",
	       "WrongClassificationDistribution",
	       "WrongClassificationPairs"
	    } *)
 	namedPropertyList_,
    
	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
    trainingAndTestSet_,
    
  	(* See "Frequently used data structures" *)
    mlp1Info_,
    
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			minMaxIndex,
			imageSize,
			testSet,
			trainingSet
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    minMaxIndex = GraphicsOptionMinMaxIndex/.{opts}/.Options[GraphicsOptionsIndex];

    	trainingSet = trainingAndTestSet[[1]];
    	testSet =  trainingAndTestSet[[2]];
    	
		Print["Training Set:"];
		ShowMlp1SingleClassification[
			namedPropertyList,
			trainingSet, 
			mlp1Info,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionMinMaxIndex -> minMaxIndex
		];
		
		(* Analyze test set *)
		If[Length[testSet] > 0,
			Print["Test Set:"];
			ShowMlp1SingleClassification[
				namedPropertyList,
				testSet, 
				mlp1Info,
				GraphicsOptionImageSize -> imageSize,
				GraphicsOptionMinMaxIndex -> minMaxIndex
			];
		]
	];

ShowMlp1SingleClassification[

	(* Shows result of mlp1 classification for data set according to named property list.

	   Returns: Nothing *)

    
	(* Properties to be analyzed: 
	   Full list: 
	   {
	       "CorrectClassification",
	       "CorrectClassificationPerClass",
	       "WrongClassificationDistribution",
	       "WrongClassificationPairs"
	   } *)
 	namedPropertyList_,
    
	(* classificationDataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} 
	   outputValue: 0/1
	   Data set must be a classification data set, i.e. the output components must 0/1 code a class,
	   i.e. class 4 of 5 must be coded {0, 0, 0, 1, 0} *)
    classificationDataSet_,
    
  	(* See "Frequently used data structures" *)
    mlp1Info_,
    
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			minMaxIndex,
			imageSize,
			pureFunction
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    minMaxIndex = GraphicsOptionMinMaxIndex/.{opts}/.Options[GraphicsOptionsIndex];

   		pureFunction = Function[inputs, CalculateMlp1ClassNumbers[inputs, mlp1Info]];
		CIP`Graphics`ShowClassificationResult[
			namedPropertyList,
			classificationDataSet, 
			pureFunction,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionMinMaxIndex -> minMaxIndex
		]
	];

ShowMlp1ClassificationScan[

	(* Shows result of Mlp1 based classification scan of clustered training sets.

	   Returns: Nothing *)


	(* mlp1ClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlp1Info1}, {trainingAndTestSet2, mlp1Info2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, classification result in percent for training set}, {trainingFraction, classification result in percent for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)
	mlp1ClassificationScan_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			imageSize,
			displayFunction
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];

		CIP`Graphics`ShowClassificationScan[
			mlp1ClassificationScan,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMlp1InputRelevanceClass[

	(* Shows mlp1InputComponentRelevanceListForClassification.

	   Returns: Nothing *)


	(* mlp1InputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlp1Info}
	   trainingSetResult: {numberOfRemovedInputs, (best) correct classification in percent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best correct classification in percent of test set} *)
	mlp1InputComponentRelevanceListForClassification_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			displayFunction,
			imageSize
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];

		CIP`Graphics`ShowInputRelevanceClass[
			mlp1InputComponentRelevanceListForClassification,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMlp1InputRelevanceRegress[

	(* Shows mlp1InputComponentRelevanceListForRegression.

	   Returns: Nothing *)


	(* mlp1InputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlp1Info}
	   trainingSetResult: {numberOfRemovedInputs, (best) RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)
	mlp1InputComponentRelevanceListForRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			displayFunction,
			imageSize
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];

		CIP`Graphics`ShowInputRelevanceRegress[
			mlp1InputComponentRelevanceListForRegression,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];
	
ShowMlp1RegressionResult[

	(* Shows result of mlp1 regression for training and test set according to named property list.
	
	   Returns: Nothing *)


	(* Properties to be analyzed: 
	   Full list: 
	   {
	       "RMSE",
	       "SingleOutputRMSE",
	       "AbsoluteResidualsStatistics",
		   "RelativeResidualsStatistics",
		   "ModelVsDataPlot",
		   "CorrelationCoefficient",
		   "SortedModelVsDataPlot",
		   "AbsoluteSortedResidualsPlot",
		   "RelativeSortedResidualsPlot"
	    } *)
 	namedPropertyList_,

	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
    trainingAndTestSet_,
    
  	(* See "Frequently used data structures" *)
    mlp1Info_,
    
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			pointSize,
			pointColor,
			
			testSet,
			trainingSet		
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];

    	trainingSet = trainingAndTestSet[[1]];
    	testSet =  trainingAndTestSet[[2]];

		(* Analyze training set *)
	    Print["Training Set:"];
		ShowMlp1SingleRegression[
			namedPropertyList,
			trainingSet, 
			mlp1Info,
			GraphicsOptionPointSize -> pointSize,
			GraphicsOptionPointColor -> pointColor
		];
		
		(* Analyze test set *)
		If[Length[testSet] > 0,
			Print["Test Set:"];
			ShowMlp1SingleRegression[
				namedPropertyList,
				testSet, 
				mlp1Info,
				GraphicsOptionPointSize -> pointSize,
				GraphicsOptionPointColor -> pointColor
			]
		]
	];

ShowMlp1SingleRegression[
    
	(* Shows result of mlp1 regression for data set according to named property list.
	
	   Returns: Nothing *)


	(* Properties to be analyzed: 
	   Full list: 
	   {
	       "RMSE",
	       "SingleOutputRMSE",
	       "AbsoluteResidualsStatistics",
		   "RelativeResidualsStatistics",
		   "ModelVsDataPlot",
		   "CorrelationCoefficient",
		   "SortedModelVsDataPlot",
		   "AbsoluteSortedResidualsPlot",
		   "RelativeSortedResidualsPlot"
	    } *)
 	namedPropertyList_,

    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...}
	   NOTE: Data set MUST be in original units *)
    dataSet_,
    
  	(* See "Frequently used data structures" *)
    mlp1Info_,
    
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			pureFunction,
    		pointSize,
    		pointColor
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];

		pureFunction = Function[inputs, CalculateMlp1Outputs[inputs, mlp1Info]];
		CIP`Graphics`ShowRegressionResult[
			namedPropertyList,
			dataSet, 
			pureFunction,
			GraphicsOptionPointSize -> pointSize,
			GraphicsOptionPointColor -> pointColor
		]
	];

ShowMlp1RegressionScan[

	(* Shows result of Mlp1 based regression scan of clustered training sets.

	   Returns: Nothing *)


	(* mlp1RegressionScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlp1Info1}, {trainingAndTestSet2, mlp1Info2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, RMSE for training set}, {trainingFraction, RMSE for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)
	mlp1RegressionScan_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			imageSize,
			displayFunction
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];

		CIP`Graphics`ShowRegressionScan[
			mlp1RegressionScan,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMlp1SeriesClassificationResult[

	(* Shows result of Mlp1 series classifications for training and test set.

	   Returns: Nothing *)


	(* mlp1SeriesClassificationResult: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in mlp1InfoList, classification result in percent for training set}
	   testPoint[[i]]: {index i in mlp1InfoList, classification result in percent for test set} *)
	mlp1SeriesClassificationResult_,
    
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			i,
			trainingPoints2D,
			testPoints2D,
			labels,
			displayFunction,
			imageSize,
			trainingPoints2DWithPlotStyle,
			testPoints2DWithPlotStyle,
			points2DWithPlotStyleList,
			bestIndexList,
			maximum
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];

		trainingPoints2D = mlp1SeriesClassificationResult[[1]];
		testPoints2D = mlp1SeriesClassificationResult[[2]];
		
		If[Length[testPoints2D] > 0,

			(* Training and test set *)
			labels = {"mlp1Info index", "Correct classifications [%]", "Training (green), Test (red)"};
			trainingPoints2DWithPlotStyle = {trainingPoints2D, {Thickness[0.005], Green}};
			testPoints2DWithPlotStyle = {testPoints2D, {Thickness[0.005], Red}};
			points2DWithPlotStyleList = {trainingPoints2DWithPlotStyle, testPoints2DWithPlotStyle};
			Print[
				CIP`Graphics`PlotMultipleLines2D[
					points2DWithPlotStyleList, 
					labels,
					GraphicsOptionImageSize -> imageSize,
					GraphicsOptionDisplayFunction -> displayFunction
				]
			];
			bestIndexList = {};
			maximum = -1.0;
			Do[
				If[testPoints2D[[i, 2]] > maximum,
					
					maximum = testPoints2D[[i, 2]];
					bestIndexList = {i},
					
					
					If[testPoints2D[[i, 2]] == maximum, AppendTo[bestIndexList, i]]
				],
				
				{i, Length[testPoints2D]}
			];
			Print["Best test set classification with mlp1Info index = ", bestIndexList],
		
			(* Training set only *)
			labels = {"mlp1Info index", "Correct classifications [%]", "Training (green)"};
			trainingPoints2DWithPlotStyle = {trainingPoints2D, {Thickness[0.005], Green}};
			points2DWithPlotStyleList = {trainingPoints2DWithPlotStyle};
			Print[
				CIP`Graphics`PlotMultipleLines2D[
					points2DWithPlotStyleList, 
					labels,
					GraphicsOptionImageSize -> imageSize,
					GraphicsOptionDisplayFunction -> displayFunction
				]
			];
			bestIndexList = {};
			maximum = -1.0;
			Do[
				If[trainingPoints2D[[i, 2]] > maximum,
					
					maximum = trainingPoints2D[[i, 2]];
					bestIndexList = {i},
					
					
					If[trainingPoints2D[[i, 2]] == maximum, AppendTo[bestIndexList, i]]
				],
				
				{i, Length[trainingPoints2D]}
			];
			Print["Best training set classification with mlp1Info index = ", bestIndexList]			
		]
	];

ShowMlp1SeriesRmse[

	(* Shows RMSE of Mlp1 series for training and test set.

	   Returns: Nothing *)


	(* mlp1SeriesRmse: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in mlp1InfoList, RMSE for training set}
	   testPoint[[i]]: {index i in mlp1InfoList, RMSE for test set} *)
	mlp1SeriesRmse_,
    
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			i,
			trainingPoints2D,
			testPoints2D,
			labels,
			displayFunction,
			imageSize,
			trainingPoints2DWithPlotStyle,
			testPoints2DWithPlotStyle,
			points2DWithPlotStyleList,
			bestIndexList,
			minimum
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];

		trainingPoints2D = mlp1SeriesRmse[[1]];
		testPoints2D = mlp1SeriesRmse[[2]];

		If[Length[testPoints2D] > 0,
			
			(* Training and test set *)
			labels = {"mlp1Info index", "RMSE", "Training (green), Test (red)"};
			trainingPoints2DWithPlotStyle = {trainingPoints2D, {Thickness[0.005], Green}};
			testPoints2DWithPlotStyle = {testPoints2D, {Thickness[0.005], Red}};
			points2DWithPlotStyleList = {trainingPoints2DWithPlotStyle, testPoints2DWithPlotStyle};
			Print[
				CIP`Graphics`PlotMultipleLines2D[
					points2DWithPlotStyleList, 
					labels,
					GraphicsOptionImageSize -> imageSize,
					GraphicsOptionDisplayFunction -> displayFunction
				]
			];
			bestIndexList = {};
			minimum = Infinity;
			Do[
				If[testPoints2D[[i, 2]] < minimum,
					
					minimum = testPoints2D[[i, 2]];
					bestIndexList = {i},
					
					If[testPoints2D[[i, 2]] == minimum, AppendTo[bestIndexList, i]]
				],
				
				{i, Length[testPoints2D]}
			];
			Print["Best test set regression with mlp1Info index = ", bestIndexList],

			(* Training set only *)
			labels = {"mlp1Info index", "RMSE", "Training (green)"};
			trainingPoints2DWithPlotStyle = {trainingPoints2D, {Thickness[0.005], Green}};
			points2DWithPlotStyleList = {trainingPoints2DWithPlotStyle};
			Print[
				CIP`Graphics`PlotMultipleLines2D[
					points2DWithPlotStyleList, 
					labels,
					GraphicsOptionImageSize -> imageSize,
					GraphicsOptionDisplayFunction -> displayFunction
				]
			];
			bestIndexList = {};
			minimum = Infinity;
			Do[
				If[trainingPoints2D[[i, 2]] < minimum,
					
					minimum = trainingPoints2D[[i, 2]];
					bestIndexList = {i},
					
					If[trainingPoints2D[[i, 2]] == minimum, AppendTo[bestIndexList, i]]
				],
				
				{i, Length[trainingPoints2D]}
			];
			Print["Best training set regression with mlp1Info index = ", bestIndexList]			
		]
	];

ShowMlp1Training[

	(* Shows training of mlp1.

	   Returns: Nothing *)


  	(* See "Frequently used data structures" *)
    mlp1Info_
    
	] :=
  
	Module[
    
		{
			i,
			labels,
			mlp1TrainingResults,
			trainingSetMeanSquaredErrorList,
			testSetMeanSquaredErrorList
		},

		mlp1TrainingResults = mlp1Info[[3]];
		Do[

			If[Length[mlp1TrainingResults] == 1,
				
				labels = 
					{
						"Number of Iterations",
						"Mean Squared Error [Training Units]",
						"Blue: Training Set, Red: Test Set"
					},
					
				labels = 
					{
						"Number of Iterations",
						"Mean Squared Error [Training Units]",
						StringJoin["Component ", ToString[i], " - Blue: Training Set, Red: Test Set"]
					}
			];
			
			trainingSetMeanSquaredErrorList = mlp1TrainingResults[[i, 1]];
			testSetMeanSquaredErrorList = mlp1TrainingResults[[i, 2]];
			Print[
		    	CIP`Graphics`PlotUpToFourLines2D[
		    		testSetMeanSquaredErrorList, {}, trainingSetMeanSquaredErrorList, {}, 
					labels
				]
			];
			Print[""];
			Print["Best mean squared error of training set (in training units) = ", trainingSetMeanSquaredErrorList[[Length[trainingSetMeanSquaredErrorList], 2]]];
			If[Length[testSetMeanSquaredErrorList] > 0,
				Print["Best mean squared error of test set (in training units)     = ", testSetMeanSquaredErrorList[[Length[testSetMeanSquaredErrorList], 2]]];
			];
			Print[""],
		
			{i, Length[mlp1TrainingResults]}
		]
	];

ShowMlp1TrainOptimization[

	(* Shows training set optimization result of mlp1.

	   Returns: Nothing *)


	(* mlp1TrainOptimization = {trainingSetRmseList, testSetRmseList, not interesting, not interesting}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set} *)
	mlp1TrainOptimization_,
    
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			displayFunction,
			imageSize
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];

		CIP`Graphics`ShowTrainOptimization[
			mlp1TrainOptimization, 
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

SigmoidFunction[

	(* Sigmoid function

	   Returns:
	   Sigmoid function value *)

	(* Do NOT test with NumberQ because of overload with vectors *)
	x_
	
	] := 1./(1. + Exp[-x]);

(* ::Section:: *)
(* End of Package *)

End[]

EndPackage[]
