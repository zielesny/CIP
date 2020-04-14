(*
-----------------------------------------------------------------------
Computational Intelligence Packages (CIP): Package MLP3
(Multi-Layer Perceptron with 3 Hidden-Unit Layers 
or
Five-Layer Feed-Forward Neural Network)
Version 3.1 for Mathematica 11 or higher
-----------------------------------------------------------------------

Authors: Achim Zielesny 

GNWI - Gesellschaft fuer naturwissenschaftliche Informatik mbH, 
Dortmund, Germany

Citation:
Achim Zielesny, Computational Intelligence Packages (CIP), Version 3.1, 
GNWI mbH (http://www.gnwi.de), Dortmund, Germany, 2020.

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
mlp3Info: {networks, dataSetScaleInfo, mlp3TrainingResults, normalizationInfo, activationAndScaling, optimizationMethod} 

	networks: {weights1, weights2, ...}
	weights: {hidden1Weights, hidden2Weights, hidden3Weights, outputWeights}
	hidden1Weights: Weights from input to hidden1 units
	hidden2Weights: Weights from hidden1 to hidden2 units
	hidden3Weights: Weights from hidden2 to hidden3 units
	outputWeights : Weights from hidden3 to output units
	dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs}, see GetDataSetScaleInfo
	mlp3TrainingResults: {singleTrainingResult1, singleTrainingResult2, ...}
	singleTrainingResult[[i]] corresponds to weights[[i]]
	singleTrainingResult: {trainingMeanSquaredErrorList, testMeanSquaredErrorList}
	trainingMeanSquaredErrorList: {reportPair1, reportPair2, ...}
	reportPair: {reportIteration, mean squared error of report iteration}
	testMeanSquaredErrorList: Same structure as trainingMeanSquaredErrorList
	normalizationInfo: {normalizationType, meanAndStandardDeviationList}, see CIP`DataTransformation`GetDataMatrixNormalizationInfo
	activationAndScaling: See option Mlp3OptionActivationAndScaling
	optimizationMethod: Optimization method
-----------------------------------------------------------------------
*)

(* ::Section:: *)
(* Package and dependencies *)

BeginPackage["CIP`MLP3`", {"CIP`Utility`", "CIP`Graphics`", "CIP`DataTransformation`", "CIP`Cluster`"}]

(* ::Section:: *)
(* Off settings *)

Off[General::"spell1"]
Off[General::shdw]
Off[FindMinimum::cvmit]
Off[General::compat]

(* ::Section:: *)
(* Options *)

Options[Mlp3OptionsTraining] = 
{
	(* True: Multiple mlp3s may be used (one mlp3 for every single output component), False: One mlp3 is used only *)
    Mlp3OptionMultipleMlp3s -> True,
	
	(* Optimization method: "FindMinimum", "NMinimize" *)
    Mlp3OptionOptimizationMethod -> "FindMinimum",
    
    (* Test set *)
    Mlp3OptionTestSet -> {},
    
	(* activationAndScaling: Definition of activation function and corresponding input/output scaling of data
       activationAndScaling: {activation, inputOutputTargetIntervals}
       e.g.
       {{"Sigmoid", "Sigmoid", "Sigmoid", "Sigmoid"}, {{-0.9, 0.9}, {0.1, 0.9}}}
       {{"Tanh", "Tanh", "Tanh", "Sigmoid"}, {{-0.9, 0.9}, {0.1, 0.9}}}
       {{"Tanh", "Tanh", "Tanh", "Tanh"}, {{-0.9, 0.9}, {-0.9, 0.9}}}
	   
	   activation: 
	   {<Activation function for hidden1 neurons>, <... for hidden2 neurons>, <... for hidden3 neurons>, <... for output neurons>}
	   Activation function for hidden1/2/3/output neurons: "Sigmoid", "Tanh"
	   
	   inputOutputTargetIntervals: {inputTargetInterval, outputTargetInterval}
	   inputTargetInterval/outputTargetInterval contains the desired minimum and maximum value for each column of inputs and outputs
	   inputTargetInterval/outputTargetInterval: {targetMin, targetMax} 
	   targetMin: Minimum value for each column 
	   targetMax: Maximum value for each column *)
	Mlp3OptionActivationAndScaling -> {{"Sigmoid", "Sigmoid", "Sigmoid", "Sigmoid"}, {{-0.9, 0.9}, {0.1, 0.9}}},
	
	(* Lambda parameter for L2 regularization: A value of 0.0 means NO L2 regularization *)
	Mlp3OptionLambdaL2Regularization -> 0.0,
	
	(* Cost function type: "SquaredError", "Cross-Entropy" *)
	Mlp3OptionCostFunctionType -> "SquaredError"
}

Options[Mlp3OptionsOptimization] = 
{
	(* Initial weights to be improved (may be empty list)
	   initialWeights: {hidden1Weights, hidden2Weights, hidden3Weights, outputWeights}
	   hidden1Weights: Weights from input to hidden1 units
	   hidden2Weights: Weights from hidden1 to hidden2 units
	   hidden3Weights: Weights from hidden2 to hidden3 units
	   outputWeights: Weights from hidden3 to output units *)
	Mlp3OptionInitialWeights -> {},

	(* Initial networks for multiple mlp2s training to be improved (may be empty list)
	   networks: {weights1, weights2, ...}
	   weights: {hidden1Weights, hidden2Weights, hidden3Weights, outputWeights}
	   hidden1Weights: Weights from input to hidden1 units
	   hidden2Weights: Weights from hidden1 to hidden2 units
	   hidden3Weights: Weights from hidden2 to hidden3 units
	   outputWeights: Weights from hidden3 to output units *)
	Mlp3OptionInitialNetworks -> {},
	
    (* Weights for NMinimize will be in interval 
       -Mlp2OptionWeightsValueLimit <= weight value <= +Mlp2OptionWeightsValueLimit*)
	Mlp3OptionWeightsValueLimit -> 1000.0,
	
    (* Number of digits for AccuracyGoal and PrecisionGoal (MUST be smaller than MachinePrecision) *)
    Mlp3OptionMinimizationPrecision -> 5,
    
    (* Maximum number of minimization steps *)
    Mlp3OptionMaximumIterations -> 10000,

    (* Number of iterations to improve *)
    Mlp3OptionIterationsToImprove -> 1000,
    
    (* The meanSquaredErrorLists (training protocol) will be filled every reportIteration steps.
       reportIteration <= 0 means no internal reports during training/minimization procedure. *)
    Mlp3OptionReportIteration -> 0
}

Options[Mlp3OptionsUnused1] = 
{
    (* Unused *)
    Mlp3OptionUnused11 -> 0.0,
    
    (* Unused *)
    Mlp3OptionUnused12 -> 0.0,
    
    (* Unused *)
    Mlp3OptionUnused13 -> 0.0
}

Options[Mlp3OptionsUnused2] =
{
    (* Unused *)
    Mlp3OptionUnused21 -> 0.0,
    
    (* Unused *)
    Mlp3OptionUnused22 -> 0.0,
    
    (* Unused *)
    Mlp3OptionUnused23 -> 0.0
}

(* ::Section:: *)
(* Declarations *)

BumpFunction::usage = 
	"BumpFunction[x, interval]"

BumpSum::usage = 
	"BumpSum[x, intervals]"

CalculateMlp3Value2D::usage = 
	"CalculateMlp3Value2D[argumentValue, indexOfInput, indexOfFunctionValueOutput, input, mlp3Info]"

CalculateMlp3Value3D::usage = 
	"CalculateMlp3Value3D[argumentValue1, argumentValue2, indexOfInput1, indexOfInput2, indexOfFunctionValueOutput, input, mlp3Info]"

CalculateMlp3ClassNumber::usage = 
	"CalculateMlp3ClassNumber[input, mlp3Info]"

CalculateMlp3ClassNumbers::usage = 
	"CalculateMlp3ClassNumbers[inputs, mlp3Info]"

CalculateMlp3DataSetRmse::usage = 
	"CalculateMlp3DataSetRmse[dataSet, mlp3Info]"

CalculateMlp3Output::usage = 
	"CalculateMlp3Output[input, mlp3Info]"

CalculateMlp3Outputs::usage = 
	"CalculateMlp3Outputs[inputs, mlp3Info]"

FitMlp3::usage = 
	"FitMlp3[dataSet, numberOfHiddenNeurons, options]"

FitMlp3Series::usage = 
	"FitMlp3Series[dataSet, numberOfHiddenNeuronsList, options]"

GetBestMlp3ClassOptimization::usage = 
	"GetBestMlp3ClassOptimization[mlp3TrainOptimization, options]"

GetBestMlp3RegressOptimization::usage = 
	"GetBestMlp3RegressOptimization[mlp3TrainOptimization, options]"

GetNumberOfHiddenNeurons::usage = 
	"GetNumberOfHiddenNeurons[mlp3Info]"

GetMlp3InputInclusionClass::usage = 
	"GetMlp3InputInclusionClass[trainingAndTestSet, numberOfHiddenNeurons, options]"

GetMlp3InputInclusionRegress::usage = 
	"GetMlp3InputInclusionRegress[trainingAndTestSet, numberOfHiddenNeurons, options]"

GetMlp3InputRelevanceClass::usage = 
	"GetMlp3InputRelevanceClass[trainingAndTestSet, numberOfHiddenNeurons, options]"

GetMlp3ClassRelevantComponents::usage = 
    "GetMlp3ClassRelevantComponents[mlp3InputComponentRelevanceListForClassification, numberOfComponents]"

GetMlp3InputRelevanceRegress::usage = 
	"GetMlp3InputRelevanceRegress[trainingAndTestSet, numberOfHiddenNeurons, options]"

GetMlp3RegressRelevantComponents::usage = 
    "GetMlp3RegressRelevantComponents[mlp3InputComponentRelevanceListForRegression, numberOfComponents]"

GetMlp3RegressionResult::usage = 
	"GetMlp3RegressionResult[namedProperty, dataSet, mlp3Info, options]"

GetMlp3SeriesClassificationResult::usage = 
	"GetMlp3SeriesClassificationResult[trainingAndTestSet, mlp3InfoList]"

GetMlp3SeriesRmse::usage = 
	"GetMlp3SeriesRmse[trainingAndTestSet, mlp3InfoList]"

GetMlp3Structure::usage = 
	"GetMlp3Structure[mlp3Info]"

GetMlp3TrainOptimization::usage = 
	"GetMlp3TrainOptimization[dataSet, numberOfHiddenNeurons, trainingFraction, numberOfTrainingSetOptimizationSteps, options]"

GetMlp3Weights::usage = 
	"GetMlp3Weights[mlp3Info, indexOfNetwork]"

ScanClassTrainingWithMlp3::usage = 
	"ScanClassTrainingWithMlp3[dataSet, numberOfHiddenNeurons, trainingFractionList, options]"

ScanRegressTrainingWithMlp3::usage = 
	"ScanRegressTrainingWithMlp3[dataSet, numberOfHiddenNeurons, trainingFractionList, options]"

ShowMlp3Output2D::usage = 
	"ShowMlp3Output2D[indexOfInput, indexOfFunctionValueOutput, input, arguments, mlp3Info]"

ShowMlp3Output3D::usage = 
	"ShowMlp3Output3D[indexOfInput1, indexOfInput2, indexOfFunctionValueOutput, input, mlp3Info, options]"

ShowMlp3ClassificationResult::usage = 
	"ShowMlp3ClassificationResult[namedPropertyList, trainingAndTestSet, mlp3Info]"

ShowMlp3SingleClassification::usage = 
	"ShowMlp3SingleClassification[namedPropertyList, classificationDataSet, mlp3Info]"

ShowMlp3ClassificationScan::usage = 
	"ShowMlp3ClassificationScan[mlp3ClassificationScan, options]"

ShowMlp3InputRelevanceClass::usage = 
	"ShowMlp3InputRelevanceClass[mlp3InputComponentRelevanceListForClassification, options]"
	
ShowMlp3InputRelevanceRegress::usage = 
	"ShowMlp3InputRelevanceRegress[mlp3InputComponentRelevanceListForRegression, options]"

ShowMlp3RegressionResult::usage = 
	"ShowMlp3RegressionResult[namedPropertyList, trainingAndTestSet, mlp3Info]"

ShowMlp3SingleRegression::usage = 
	"ShowMlp3SingleRegression[namedPropertyList, dataSet, mlp3Info]"

ShowMlp3RegressionScan::usage = 
	"ShowMlp3RegressionScan[mlp3RegressionScan, options]"

ShowMlp3SeriesClassificationResult::usage = 
	"ShowMlp3SeriesClassificationResult[mlp3SeriesClassificationResult, options]"

ShowMlp3SeriesRmse::usage = 
	"ShowMlp3SeriesRmse[mlp3SeriesRmse, options]"

ShowMlp3Training::usage = 
	"ShowMlp3Training[mlp3Info]"

ShowMlp3TrainOptimization::usage = 
	"ShowMlp3TrainOptimization[mlp3TrainOptimization, options]"

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

CalculateMlp3Value2D[

	(* Calculates 2D output for specified argument and input for specified mlp3.
	   This special method assumes an input and an output with one component only.

	   Returns:
	   Value of specified output neuron for argument *)

    (* Argument value for neuron with index indexOfInput *)
    argumentValue_?NumberQ,
    
  	(* See "Frequently used data structures" *)
    mlp3Info_
    
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
			CalculateMlp3Value2D[argumentValue, indexOfInput, indexOfFunctionValueOutput, input, mlp3Info]
		]
	];

CalculateMlp3Value2D[

	(* Calculates 2D output for specified argument and input for specified mlp3.

	   Returns:
	   Value of specified output neuron for argument and input *)

    (* Argument value for neuron with index indexOfInput *)
    argumentValue_?NumberQ,
    
    (* Index of input neuron that receives argumentValue *)
    indexOfInput_?IntegerQ,

    (* Index of output neuron that returns function value *)
    indexOfFunctionValueOutput_?IntegerQ,
    
    (* Mlp3 input in original units: 
       inputsInOriginalUnits = {inputValue1, inputValue2, ...} 
       Value of input neuron with specified index (indexOfInput) is replaced by argumentValue *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlp3Info_
    
	] :=
  
	Module[
    
		{
			currentInput,
			output
		},
		
		currentInput = ReplacePart[input, {indexOfInput -> argumentValue}];
		output = CalculateMlp3Output[currentInput, mlp3Info];
		Return[output[[indexOfFunctionValueOutput]]];
	];

CalculateMlp3Value3D[

	(* Calculates 3D output for specified arguments for specified mlp3. 
	   This specific methods assumes a mlp3 with 2 input neurons and 1 output neuron.

	   Returns:
	   Value of the single output neuron for arguments *)


    (* Argument value for neuron with index indexOfInput1 *)
    argumentValue1_?NumberQ,
    
    (* Argument value for neuron with index indexOfInput2 *)
    argumentValue2_?NumberQ,
    
  	(* See "Frequently used data structures" *)
    mlp3Info_
    
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
			CalculateMlp3Value3D[argumentValue1, argumentValue2, indexOfInput1, indexOfInput2, indexOfOutput, input, mlp3Info]
		]
	];

CalculateMlp3Value3D[

	(* Calculates 3D output for specified arguments and input for specified mlp3.

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
    
    (* Mlp3 input in original units: 
       inputsInOriginalUnits = {inputValue1, inputValue2, ...} 
       Value of input neurons with specified indices (indexOfInput1, indexOfInput2) are replaced by argument values *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlp3Info_
    
	] :=
  
	Module[
    
		{
			currentInput,
			output
		},
		
		currentInput = ReplacePart[input, {indexOfInput1 -> argumentValue1, indexOfInput2 -> argumentValue2}];
		output = CalculateMlp3Output[currentInput, mlp3Info];
		Return[output[[indexOfFunctionValueOutput]]];
	];

CalculateMlp3ClassNumber[

	(* Returns class number for specified input for classification mlp3 with specified weights.

	   Returns:
	   Class number of input *)

    
    (* input in original units: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    input_/;VectorQ[input, NumberQ],
        
  	(* See "Frequently used data structures" *)
    mlp3Info_
    
	] :=
  
	Module[
    
		{
			combinedOutputs,
			hidden1Weights,
			hidden2Weights,
			hidden3Weights,
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
    
    	networks = mlp3Info[[1]];
    	dataSetScaleInfo = mlp3Info[[2]];
    	normalizationInfo = mlp3Info[[4]];
    	activationAndScaling = mlp3Info[[5]];

		If[Length[networks] == 1,
	
			(* --------------------------------------------------------------------------------
			   One network
			   -------------------------------------------------------------------------------- *)		

			weights = networks[[1]];
    		hidden1Weights = weights[[1]];
    		hidden2Weights = weights[[2]];
    		hidden3Weights = weights[[3]];
			outputWeights = weights[[4]];
			(* Transform original input *)
			scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo];
			outputs = GetInternalMlp3Outputs[scaledInputs, hidden1Weights, hidden2Weights, hidden3Weights, outputWeights, activationAndScaling];
			Return[CIP`Utility`GetPositionOfMaximumValue[outputs[[1]]]],
			
			(* --------------------------------------------------------------------------------
			   Multiple networks (with ONE output value each)
			   -------------------------------------------------------------------------------- *)		

			combinedOutputs =
				Table[
					weights = networks[[i]];
		    		hidden1Weights = weights[[1]];
		    		hidden2Weights = weights[[2]];
		    		hidden3Weights = weights[[3]];
					outputWeights = weights[[4]];
					(* Transform original input *)
					scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo];
					outputs = GetInternalMlp3Outputs[scaledInputs, hidden1Weights, hidden2Weights, hidden3Weights, outputWeights, activationAndScaling];
					outputs[[1, 1]],
					
					{i, Length[networks]}
				];
			Return[CIP`Utility`GetPositionOfMaximumValue[combinedOutputs]]
		]
	];

CalculateMlp3ClassNumbers[

	(* Returns class numbers for specified inputs for classification mlp3 with specified weights.

	   Returns:
	   {class number of input1, class number of input2, ...} *)

    
    (* {inputsInOriginalUnit1, inputsInOriginalUnit2, ...}
        inputsInOriginalUnit: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    inputs_/;MatrixQ[inputs, NumberQ],
        
  	(* See "Frequently used data structures" *)
    mlp3Info_
    
	] :=
  
	Module[
    
		{
			combinedOutputs,
			correspondingOutput,
			hidden1Weights,
			hidden2Weights,
			hidden3Weights,
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

    	networks = mlp3Info[[1]];
    	dataSetScaleInfo = mlp3Info[[2]];
    	normalizationInfo = mlp3Info[[4]];
    	activationAndScaling = mlp3Info[[5]];

		If[Length[networks] == 1,
	
			(* --------------------------------------------------------------------------------
			   One network
			   -------------------------------------------------------------------------------- *)		

			weights = networks[[1]];
    		hidden1Weights = weights[[1]];
    		hidden2Weights = weights[[2]];
    		hidden3Weights = weights[[3]];
			outputWeights = weights[[4]];
			(* Transform original inputs *)
			scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
			outputs = GetInternalMlp3Outputs[scaledInputs, hidden1Weights, hidden2Weights, hidden3Weights, outputWeights, activationAndScaling];
			Return[
				Table[CIP`Utility`GetPositionOfMaximumValue[outputs[[i]]], {i, Length[outputs]}]
			],
			
			(* --------------------------------------------------------------------------------
			   Multiple networks (with ONE output value each)
			   -------------------------------------------------------------------------------- *)		

			combinedOutputs =
				Table[
					weights = networks[[i]];
		    		hidden1Weights = weights[[1]];
		    		hidden2Weights = weights[[2]];
		    		hidden3Weights = weights[[3]];
					outputWeights = weights[[4]];
					(* Transform original input *)
					scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
					outputs = GetInternalMlp3Outputs[scaledInputs, hidden1Weights, hidden2Weights, hidden3Weights, outputWeights, activationAndScaling];
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

CalculateMlp3CorrectClassificationInPercent[

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
    mlp3Info_
    
	] :=
  
	Module[
    
		{
			pureFunction
		},

		pureFunction = Function[inputs, CalculateMlp3ClassNumbers[inputs, mlp3Info]];
		Return[CIP`Utility`GetCorrectClassificationInPercent[classificationDataSet, pureFunction]]
	];

CalculateMlp3DataSetRmse[

	(* Returns RMSE of data set.

	   Returns: 
	   RMSE of data set *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

  	(* See "Frequently used data structures" *)
    mlp3Info_
    
	] :=
  
	Module[
    
		{
			pureFunction,
			rmse
		},

		pureFunction = Function[inputs, CalculateMlp3Outputs[inputs, mlp3Info]];
		rmse = Sqrt[CIP`Utility`GetMeanSquaredError[dataSet, pureFunction]];
		Return[rmse]
	];

CalculateMlp3Output[

	(* Calculates output for specified input for specified mlp3.

	   Returns:
	   output: {transformedValueOfOutput1, transformedValueOfOutput2, ...} *)

    
    (* Input in original units: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlp3Info_
    
	] :=
  
	Module[
    
		{
			combinedOutputs,
			dataMatrixScaleInfo,
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			hidden1Weights,
			hidden2Weights,
			hidden3Weights,
			i,
			networks,
			outputsInOriginalUnits,
			scaledOutputs,
			outputWeights,
			scaledInputs,
			weights
		},
    
    	networks = mlp3Info[[1]];
    	dataSetScaleInfo = mlp3Info[[2]];
    	normalizationInfo = mlp3Info[[4]];
    	activationAndScaling = mlp3Info[[5]];

		If[Length[networks] == 1,
	
			(* --------------------------------------------------------------------------------
			   One network (with multiple output values)
			   -------------------------------------------------------------------------------- *)		

			weights = networks[[1]];
    		hidden1Weights = weights[[1]];
    		hidden2Weights = weights[[2]];
    		hidden3Weights = weights[[3]];
			outputWeights = weights[[4]];
			(* Transform original input *)
			scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo];
			scaledOutputs = GetInternalMlp3Outputs[scaledInputs, hidden1Weights, hidden2Weights, hidden3Weights, outputWeights, activationAndScaling];
			(* Transform outputs to original units *)
			outputsInOriginalUnits = CIP`DataTransformation`ScaleDataMatrixReverse[scaledOutputs, dataSetScaleInfo[[2]]];
			Return[First[outputsInOriginalUnits]],
			
			(* --------------------------------------------------------------------------------
			   Multiple networks (with ONE output value each)
			   -------------------------------------------------------------------------------- *)		

			combinedOutputs =
				Table[
					weights = networks[[i]];
		    		hidden1Weights = weights[[1]];
		    		hidden2Weights = weights[[2]];
		    		hidden3Weights = weights[[3]];
					outputWeights = weights[[4]];
					(* Transform original input *)
					scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo];
					scaledOutputs = GetInternalMlp3Outputs[scaledInputs, hidden1Weights, hidden2Weights, hidden3Weights, outputWeights, activationAndScaling];

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

CalculateMlp3Outputs[

	(* Calculates outputs for specified inputs for specified mlp3.

	   Returns:
	   outputs: {output1, output2, ...} 
	   output: {transformedValueOfOutput1, transformedValueOfOutput1, ...} *)

    
    (* {inputsInOriginalUnit1, inputsInOriginalUnit2, ...}
        inputsInOriginalUnit: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    inputs_/;MatrixQ[inputs, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlp3Info_
    
	] :=
  
	Module[
    
		{
			combinedOutputs,
			dataMatrixScaleInfo,
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			hidden1Weights,
			hidden2Weights,
			hidden3Weights,
			i,
			networks,
			outputsInOriginalUnits,
			scaledOutputs,
			outputWeights,
			scaledInputs,
			weights
		},
		
    	networks = mlp3Info[[1]];
    	dataSetScaleInfo = mlp3Info[[2]];
    	normalizationInfo = mlp3Info[[4]];
    	activationAndScaling = mlp3Info[[5]];

		If[Length[networks] == 1,
	
			(* --------------------------------------------------------------------------------
			   One network (with multiple output values)
			   -------------------------------------------------------------------------------- *)		

			weights = networks[[1]];
    		hidden1Weights = weights[[1]];
    		hidden2Weights = weights[[2]];
    		hidden3Weights = weights[[3]];
			outputWeights = weights[[4]];
			(* Transform original inputs *)
			scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
			scaledOutputs = GetInternalMlp3Outputs[scaledInputs, hidden1Weights, hidden2Weights, hidden3Weights, outputWeights, activationAndScaling];
			(* Transform outputs to original units *)
			outputsInOriginalUnits = CIP`DataTransformation`ScaleDataMatrixReverse[scaledOutputs, dataSetScaleInfo[[2]]];
			Return[outputsInOriginalUnits],
			
			(* --------------------------------------------------------------------------------
			   Multiple networks (with ONE output value each)
			   -------------------------------------------------------------------------------- *)		

			combinedOutputs =
				Table[
					weights = networks[[i]];
		    		hidden1Weights = weights[[1]];
		    		hidden2Weights = weights[[2]];
		    		hidden3Weights = weights[[3]];
					outputWeights = weights[[4]];
					(* Transform original input *)
					scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
					scaledOutputs = GetInternalMlp3Outputs[scaledInputs, hidden1Weights, hidden2Weights, hidden3Weights, outputWeights, activationAndScaling];

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

FitMultipleMlp3SC[

	(* Training of multiple (1 mlp3 per output component of data set) Mlp3.
	
	   Returns:
	   mlp3Info (see "Frequently used data structures") *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,
	
	(* Options *)
	opts___

	] :=
  
	Module[
    
		{
			unusedOptionParameter22,
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			normalizationType,
			i,
			initialNetworks,
			initialWeights,
			unusedOptionParameter11,
			unusedOptionParameter12,
			maximumNumberOfIterations,
			minimizationPrecision,
			unusedOptionParameter13,
			multipleTestSet,
			multipleTrainingSet,
			unusedOptionParameter23,
			networks,
			numberOfIterationsToImprove,
			mlp3Info,
			mlp3TrainingResults,
			unusedOptionParameter21,
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
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		reportIteration = Mlp3OptionReportIteration/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
		mlp3TrainingResults = {};
		Do[
			(* If initial networks are available overwrite initialWeights *)
			If[Length[initialNetworks] > 0 && Length[initialNetworks] == Length[multipleTrainingSet],
				initialWeights = initialNetworks[[i]];
			];
			mlp3Info = 
				FitSingleMlp3[
					{multipleTrainingSet[[i]], multipleTestSet[[i]]},
					numberOfHiddenNeurons,
	    			Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp3OptionReportIteration -> reportIteration,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp3OptionActivationAndScaling -> activationAndScaling,
	    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp3OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				];
			AppendTo[networks, mlp3Info[[1, 1]]];
			AppendTo[mlp3TrainingResults, mlp3Info[[3, 1]]],
			
			{i, Length[multipleTrainingSet]}
		];

		(* ----------------------------------------------------------------------------------------------------
		   Return mlp3Info
		   ---------------------------------------------------------------------------------------------------- *)
		Return[{networks, dataSetScaleInfo, mlp3TrainingResults, normalizationInfo, activationAndScaling}]		
	];
	
FitMultipleMlp3PC[

	(* Training of multiple (1 mlp3 per output component of data set) Mlp3.
	
	   Returns:
	   mlp3Info (see "Frequently used data structures") *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,
	
	(* Options *)
	opts___

	] :=
  
	Module[
    
		{
			unusedOptionParameter22,
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			normalizationType,
			i,
			initialNetworks,
			initialWeights,
			unusedOptionParameter11,
			unusedOptionParameter12,
			maximumNumberOfIterations,
			minimizationPrecision,
			unusedOptionParameter13,
			multipleTestSet,
			multipleTrainingSet,
			unusedOptionParameter23,
			networks,
			numberOfIterationsToImprove,
			mlp3TrainingResults,
			unusedOptionParameter21,
			randomValueInitialization,
			reportIteration,
			testSet,
			optimizationMethod,
			trainingSet,
			weightsValueLimit,
			mlp3InfoList,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		reportIteration = Mlp3OptionReportIteration/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
		
		ParallelNeeds[{"CIP`Mlp3`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[initialNetworks, multipleTrainingSet, multipleTestSet, optimizationMethod, initialWeights, 
			weightsValueLimit, minimizationPrecision, maximumNumberOfIterations, numberOfIterationsToImprove, 
			reportIteration, unusedOptionParameter11, unusedOptionParameter12, unusedOptionParameter13, unusedOptionParameter21, 
			unusedOptionParameter22, unusedOptionParameter23, randomValueInitialization, activationAndScaling, normalizationType, 
			lambdaL2Regularization, costFunctionType];

		(* ----------------------------------------------------------------------------------------------------
		   Training
		   ---------------------------------------------------------------------------------------------------- *)
		mlp3InfoList = ParallelTable[
			(* If initial networks are available overwrite initialWeights *)
			If[Length[initialNetworks] > 0 && Length[initialNetworks] == Length[multipleTrainingSet],
				initialWeights = initialNetworks[[i]]
			];
			
			FitSingleMlp3[
				{multipleTrainingSet[[i]], multipleTestSet[[i]]},
				numberOfHiddenNeurons,
	    		Mlp3OptionOptimizationMethod -> optimizationMethod,
				Mlp3OptionInitialWeights -> initialWeights,
				Mlp3OptionWeightsValueLimit -> weightsValueLimit,
				Mlp3OptionMinimizationPrecision -> minimizationPrecision,
				Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
				Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
	 			Mlp3OptionReportIteration -> reportIteration,
				Mlp3OptionUnused11 -> unusedOptionParameter11,
				Mlp3OptionUnused12 -> unusedOptionParameter12,
				Mlp3OptionUnused13 -> unusedOptionParameter13,
				Mlp3OptionUnused21 -> unusedOptionParameter21,
				Mlp3OptionUnused22 -> unusedOptionParameter22,
				Mlp3OptionUnused23 -> unusedOptionParameter23,
	    		UtilityOptionRandomInitializationMode -> randomValueInitialization,
	   			Mlp3OptionActivationAndScaling -> activationAndScaling,
	   			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
	   			Mlp3OptionCostFunctionType -> costFunctionType,
	   			DataTransformationOptionNormalizationType -> normalizationType
			],
			{i, Length[multipleTrainingSet]}
		];
		networks = Table[mlp3InfoList[[i, 1, 1]], {i, Length[multipleTrainingSet]}];
		mlp3TrainingResults = Table[mlp3InfoList[[i, 3, 1]], {i, Length[multipleTrainingSet]}];
		(* ----------------------------------------------------------------------------------------------------
		   Return mlp3Info
		   ---------------------------------------------------------------------------------------------------- *)
		Return[{networks, dataSetScaleInfo, mlp3TrainingResults, normalizationInfo, activationAndScaling}]		
	];

FitMlp3[

	(* Training of single or multiple Mlp3(s).

	   Returns:
	   mlp3Info (see "Frequently used data structures") *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			unusedOptionParameter22,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			unusedOptionParameter11,
			unusedOptionParameter12,
			maximumNumberOfIterations,
			minimizationPrecision,
			unusedOptionParameter13,
			multipleMlp3s,
			unusedOptionParameter23,
			numberOfIterationsToImprove,
			unusedOptionParameter21,
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
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
	    testSet = Mlp3OptionTestSet/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		reportIteration = Mlp3OptionReportIteration/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		(* ----------------------------------------------------------------------------------------------------
		   Switch training method
		   ---------------------------------------------------------------------------------------------------- *)
		trainingAndTestSet = {dataSet, testSet};
		
		If[multipleMlp3s,
			
			Switch[parallelization,
			
				(* ------------------------------------------------------------------------------- *)
				"ParallelCalculation",
				Return[
					FitMultipleMlp3PC[
						trainingAndTestSet,
						numberOfHiddenNeurons,
		    			Mlp3OptionOptimizationMethod -> optimizationMethod,
						Mlp3OptionInitialWeights -> initialWeights,
						Mlp3OptionInitialNetworks -> initialNetworks,
						Mlp3OptionWeightsValueLimit -> weightsValueLimit,
						Mlp3OptionMinimizationPrecision -> minimizationPrecision,
						Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
			 			Mlp3OptionReportIteration -> reportIteration,
						Mlp3OptionUnused11 -> unusedOptionParameter11,
						Mlp3OptionUnused12 -> unusedOptionParameter12,
						Mlp3OptionUnused13 -> unusedOptionParameter13,
						Mlp3OptionUnused21 -> unusedOptionParameter21,
						Mlp3OptionUnused22 -> unusedOptionParameter22,
						Mlp3OptionUnused23 -> unusedOptionParameter23,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			Mlp3OptionActivationAndScaling -> activationAndScaling,
		    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
		    			Mlp3OptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					]
				],

				(* ------------------------------------------------------------------------------- *)
				"SequentialCalculation",
				Return[
					FitMultipleMlp3SC[
						trainingAndTestSet,
						numberOfHiddenNeurons,
		    			Mlp3OptionOptimizationMethod -> optimizationMethod,
						Mlp3OptionInitialWeights -> initialWeights,
						Mlp3OptionInitialNetworks -> initialNetworks,
						Mlp3OptionWeightsValueLimit -> weightsValueLimit,
						Mlp3OptionMinimizationPrecision -> minimizationPrecision,
						Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
			 			Mlp3OptionReportIteration -> reportIteration,
						Mlp3OptionUnused11 -> unusedOptionParameter11,
						Mlp3OptionUnused12 -> unusedOptionParameter12,
						Mlp3OptionUnused13 -> unusedOptionParameter13,
						Mlp3OptionUnused21 -> unusedOptionParameter21,
						Mlp3OptionUnused22 -> unusedOptionParameter22,
						Mlp3OptionUnused23 -> unusedOptionParameter23,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			Mlp3OptionActivationAndScaling -> activationAndScaling,
		    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
		    			Mlp3OptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					]
				]
			],
			
			Return[
				FitSingleMlp3[
					trainingAndTestSet,
					numberOfHiddenNeurons,
	    			Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp3OptionReportIteration -> reportIteration,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp3OptionActivationAndScaling -> activationAndScaling,
	    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp3OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				]
			]
		]
	];

FitMlp3Series[

	(* Trains of a series of single or multiple Mlp3(s).

	   Returns:
	   mlp3InfoList: {mlp3Info1, mlp3Info2, ...}
	   mlp3Info[[i]] corresponds to numberOfHiddenNeuronsList[[i]]
	   (see "Frequently used data structures") *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* List with numbers of hidden neurons:
	   numberOfHiddenNeuronsList: {numberOfHiddenNeurons1, numberOfHiddenNeurons2, ...}
	   numberOfHiddenNeurons: {number of neurons in hidden1, number of neurons in hidden2} *)
	numberOfHiddenNeuronsList_/;MatrixQ[numberOfHiddenNeuronsList, NumberQ],

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			unusedOptionParameter22,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			unusedOptionParameter11,
			unusedOptionParameter12,
			maximumNumberOfIterations,
			minimizationPrecision,
			unusedOptionParameter13,
			multipleMlp3s,
			unusedOptionParameter23,
			numberOfIterationsToImprove,
			unusedOptionParameter21,
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
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
	    testSet = Mlp3OptionTestSet/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		reportIteration = Mlp3OptionReportIteration/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
	    (* Parallelization options *)
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				FitMlp3SeriesPC[
					dataSet,
					numberOfHiddenNeuronsList,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
				    Mlp3OptionOptimizationMethod -> optimizationMethod,
				    Mlp3OptionTestSet -> testSet,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
					Mlp3OptionReportIteration -> reportIteration,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
				    Mlp3OptionActivationAndScaling -> activationAndScaling,
				    Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
				    Mlp3OptionCostFunctionType -> costFunctionType,
				    DataTransformationOptionNormalizationType -> normalizationType
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				FitMlp3SeriesSC[
					dataSet,
					numberOfHiddenNeuronsList,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
				    Mlp3OptionOptimizationMethod -> optimizationMethod,
				    Mlp3OptionTestSet -> testSet,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
					Mlp3OptionReportIteration -> reportIteration,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
				    Mlp3OptionActivationAndScaling -> activationAndScaling,
				    Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
				    Mlp3OptionCostFunctionType -> costFunctionType,
				    DataTransformationOptionNormalizationType -> normalizationType
				]
			]
		]
	];

FitMlp3SeriesSC[

	(* Trains of a series of single or multiple Mlp3(s).

	   Returns:
	   mlp3InfoList: {mlp3Info1, mlp3Info2, ...}
	   mlp3Info[[i]] corresponds to numberOfHiddenNeuronsList[[i]]
	   (see "Frequently used data structures") *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* List with numbers of hidden neurons:
	   numberOfHiddenNeuronsList: {numberOfHiddenNeurons1, numberOfHiddenNeurons2, ...}
	   numberOfHiddenNeurons: {number of neurons in hidden1, number of neurons in hidden2} *)
	numberOfHiddenNeuronsList_/;MatrixQ[numberOfHiddenNeuronsList, NumberQ],

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			i,
			unusedOptionParameter22,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			unusedOptionParameter11,
			unusedOptionParameter12,
			maximumNumberOfIterations,
			minimizationPrecision,
			unusedOptionParameter13,
			multipleMlp3s,
			unusedOptionParameter23,
			numberOfIterationsToImprove,
			unusedOptionParameter21,
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
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
	    testSet = Mlp3OptionTestSet/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		reportIteration = Mlp3OptionReportIteration/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		Return[
			Table[
				FitMlp3[
					dataSet,
					numberOfHiddenNeuronsList[[i]],
					Mlp3OptionTestSet -> testSet,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
	    			Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp3OptionReportIteration -> reportIteration,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp3OptionActivationAndScaling -> activationAndScaling,
	    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp3OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				],
				
				{i, Length[numberOfHiddenNeuronsList]}
			]			
		]
	];
	
FitMlp3SeriesPC[

	(* Trains of a series of single or multiple Mlp3(s).

	   Returns:
	   mlp3InfoList: {mlp3Info1, mlp3Info2, ...}
	   mlp3Info[[i]] corresponds to numberOfHiddenNeuronsList[[i]]
	   (see "Frequently used data structures") *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* List with numbers of hidden neurons:
	   numberOfHiddenNeuronsList: {numberOfHiddenNeurons1, numberOfHiddenNeurons2, ...}
	   numberOfHiddenNeurons: {number of neurons in hidden1, number of neurons in hidden2} *)
	numberOfHiddenNeuronsList_/;MatrixQ[numberOfHiddenNeuronsList, NumberQ],

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			i,
			unusedOptionParameter22,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			unusedOptionParameter11,
			unusedOptionParameter12,
			maximumNumberOfIterations,
			minimizationPrecision,
			unusedOptionParameter13,
			multipleMlp3s,
			unusedOptionParameter23,
			numberOfIterationsToImprove,
			unusedOptionParameter21,
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
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
	    testSet = Mlp3OptionTestSet/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		reportIteration = Mlp3OptionReportIteration/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		
		ParallelNeeds[{"CIP`Mlp3`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[testSet, multipleMlp3s, optimizationMethod, initialWeights, initialNetworks, 
			weightsValueLimit, minimizationPrecision, maximumNumberOfIterations, numberOfIterationsToImprove, 
			reportIteration, unusedOptionParameter11, unusedOptionParameter12, unusedOptionParameter13, unusedOptionParameter21, 
			unusedOptionParameter22, unusedOptionParameter23, randomValueInitialization, activationAndScaling, normalizationType, 
			lambdaL2Regularization, costFunctionType];

		Return[
			ParallelTable[
				FitMlp3[
					dataSet,
					numberOfHiddenNeuronsList[[i]],
					Mlp3OptionTestSet -> testSet,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
   	 				Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
	 				Mlp3OptionReportIteration -> reportIteration,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
    				UtilityOptionRandomInitializationMode -> randomValueInitialization,
    				Mlp3OptionActivationAndScaling -> activationAndScaling,
    				Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
    				Mlp3OptionCostFunctionType -> costFunctionType,
    				DataTransformationOptionNormalizationType -> normalizationType
				],
			
				{i, Length[numberOfHiddenNeuronsList]}
			]			
		]
	];

FitMlp3WithFindMinimum[

	(* Training of mlp3 with FindMinimum and "ConjugateGradient" method.
	
	   Returns:
	   mlp3Info (see "Frequently used data structures") *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,
	
	(* Options *)
	opts___

	] :=
  
	Module[
    
		{
			costFunctionType,
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			normalizationType,
			hidden1Weights,
			hidden2Weights,
			hidden3Weights,
			hidden1WeightsValueLimit,
			hidden2WeightsValueLimit,
			hidden3WeightsValueLimit,
			hidden1WeightsVariables,
			hidden2WeightsVariables,
			hidden3WeightsVariables,
			i,
			initialWeights,
			inputs,
			intermediateResult1,
			intermediateResult2,
			intermediateResult3,
			j,
			k,
			lambdaL2Regularization,
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
			outputWeightsValueLimit,
			outputWeightsVariables,
			mlp3Outputs,
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
			wInputToHidden1,
			wHidden1ToHidden2,
			wHidden2ToHidden3,
			wHidden3ToOutput,
			weightsRules,
			weightsVariables,
			weights,
			weightsVariablesWithoutTrueUnitBias
		},


		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
	    minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
	    maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
	    reportIteration = Mlp3OptionReportIteration/.{opts}/.Options[Mlp3OptionsOptimization];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
		   hidden1WeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons[[1]])];
		   hidden2WeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfHiddenNeurons[[1]] + 1 + numberOfHiddenNeurons[[2]])];
		   hidden3WeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfHiddenNeurons[[2]] + 1 + numberOfHiddenNeurons[[3]])];
		   outputWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfHiddenNeurons[[3]] + 1 + numberOfOutputs)];
		   
		   Wight initialization for tanh activation neurons:
		   hidden1WeightsValueLimit = Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons[[1]])];
		   hidden2WeightsValueLimit = Sqrt[6.0/(numberOfHiddenNeurons[[1]] + 1 + numberOfHiddenNeurons[[2]])];
		   hidden3WeightsValueLimit = Sqrt[6.0/(numberOfHiddenNeurons[[2]] + 1 + numberOfHiddenNeurons[[3]])]; 
		   outputWeightsValueLimit = Sqrt[6.0/(numberOfHiddenNeurons[[3]] + 1 + numberOfOutputs)];
		*)
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			hidden1WeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons[[1]])],
			
			"Tanh",
			hidden1WeightsValueLimit = Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons[[1]])]
		];
		Switch[activationAndScaling[[1, 2]],
			
			"Sigmoid",
			hidden2WeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfHiddenNeurons[[1]] + 1 + numberOfHiddenNeurons[[2]])],
			
			"Tanh",
			hidden2WeightsValueLimit = Sqrt[6.0/(numberOfHiddenNeurons[[1]] + 1 + numberOfHiddenNeurons[[2]])]
		];
		Switch[activationAndScaling[[1, 3]],
			
			"Sigmoid",
			hidden3WeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfHiddenNeurons[[2]] + 1 + numberOfHiddenNeurons[[3]])],
			
			"Tanh",
			hidden3WeightsValueLimit = Sqrt[6.0/(numberOfHiddenNeurons[[2]] + 1 + numberOfHiddenNeurons[[3]])]
		];
		Switch[activationAndScaling[[1, 4]],
			
			"Sigmoid",
			outputWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfHiddenNeurons[[3]] + 1 + numberOfOutputs)],
			
			"Tanh",
			outputWeightsValueLimit = Sqrt[6.0/(numberOfHiddenNeurons[[3]] + 1 + numberOfOutputs)]
		];

	    (* Initialize hidden and output weights *)
	    If[Length[initialWeights] > 0,
      
			(* Use specified weights as initial weights *)
			hidden1Weights = initialWeights[[1]];
			hidden2Weights = initialWeights[[2]];
			hidden3Weights = initialWeights[[3]];
			outputWeights = initialWeights[[4]],
      
			(* Use random weights as initial weights *)
			(* 'True unit' : 'numberOfInputs + 1' and 'numberOfHiddenNeurons + 1' *)
			hidden1Weights = Table[RandomReal[{-hidden1WeightsValueLimit, hidden1WeightsValueLimit}, numberOfInputs + 1], {numberOfHiddenNeurons[[1]]}];
			hidden2Weights = Table[RandomReal[{-hidden2WeightsValueLimit, hidden2WeightsValueLimit}, numberOfHiddenNeurons[[1]] + 1], {numberOfHiddenNeurons[[2]]}];
			hidden3Weights = Table[RandomReal[{-hidden3WeightsValueLimit, hidden3WeightsValueLimit}, numberOfHiddenNeurons[[2]] + 1], {numberOfHiddenNeurons[[3]]}];
			outputWeights = Table[RandomReal[{-outputWeightsValueLimit, outputWeightsValueLimit}, numberOfHiddenNeurons[[3]] + 1], {numberOfOutputs}]
		];

		(* Initialize training protocol *)
		trainingMeanSquaredErrorList = {{0, GetInternalMeanSquaredErrorOfMlp3[scaledTrainingSet, hidden1Weights, hidden2Weights, hidden3Weights, outputWeights, activationAndScaling]}};
		If[Length[scaledTestSet] > 0,
		
			(* Test set exists *)
			testMeanSquaredErrorList = {{0, GetInternalMeanSquaredErrorOfMlp3[scaledTestSet, hidden1Weights, hidden2Weights, hidden3Weights, outputWeights, activationAndScaling]}},
		
			(* No test set*)
			testMeanSquaredErrorList = {}
		];

		(* ----------------------------------------------------------------------------------------------------
		   Definition of start variables
		   ---------------------------------------------------------------------------------------------------- *)
		startVariables = 
			GetWeightsStartVariables[
				numberOfInputs, 
				numberOfHiddenNeurons, 
				numberOfOutputs, 
				wInputToHidden1, 
				wHidden1ToHidden2,
				wHidden2ToHidden3, 
				wHidden3ToOutput, 
				hidden1Weights, 
				hidden2Weights,
				hidden3Weights, 
				outputWeights
			];

		(* ----------------------------------------------------------------------------------------------------
		   Mean squared error function to minimize
		   ---------------------------------------------------------------------------------------------------- *)
	    weightsVariables = 
	    	GetWeightsVariables[
	    		numberOfInputs, 
	    		numberOfHiddenNeurons, 
	    		numberOfOutputs, 
	    		wInputToHidden1, 
	    		wHidden1ToHidden2, 
	    		wHidden2ToHidden3, 
	    		wHidden3ToOutput
    		];
	    hidden1WeightsVariables = weightsVariables[[1]];
	    hidden2WeightsVariables = weightsVariables[[2]];
	    hidden3WeightsVariables = weightsVariables[[3]];
	    outputWeightsVariables = weightsVariables[[4]];

		weightsVariablesWithoutTrueUnitBias = 
			GetWeightsVariablesWithoutTrueUnitBias[
				numberOfInputs, 
				numberOfHiddenNeurons, 
				numberOfOutputs, 
				wInputToHidden1, 
				wHidden1ToHidden2, 
	    		wHidden2ToHidden3, 
				wHidden3ToOutput
			];
	    
	    (* Map: Add 'true unit' *)
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			intermediateResult1 = SigmoidFunction[Map[Append[#, 1] &, inputs].Transpose[hidden1WeightsVariables]],
			
			"Tanh",
			intermediateResult1 = Tanh[Map[Append[#, 1] &, inputs].Transpose[hidden1WeightsVariables]]
		];
		
		Switch[activationAndScaling[[1, 2]],
			
			"Sigmoid",
			intermediateResult2 = SigmoidFunction[Map[Append[#, 1] &, intermediateResult1].Transpose[hidden2WeightsVariables]],
			
			"Tanh",
			intermediateResult2 = Tanh[Map[Append[#, 1] &, intermediateResult1].Transpose[hidden2WeightsVariables]]
		];
		
		Switch[activationAndScaling[[1, 3]],
			
			"Sigmoid",
			intermediateResult3 = SigmoidFunction[Map[Append[#, 1] &, intermediateResult2].Transpose[hidden3WeightsVariables]],
			
			"Tanh",
			intermediateResult3 = Tanh[Map[Append[#, 1] &, intermediateResult2].Transpose[hidden3WeightsVariables]]
		];

		Switch[activationAndScaling[[1, 4]],
			
			"Sigmoid",
			mlp3Outputs = SigmoidFunction[Map[Append[#, 1] &, intermediateResult3].Transpose[outputWeightsVariables]],
			
			"Tanh",
			Switch[costFunctionType,
			
				"SquaredError",				
				mlp3Outputs = Tanh[Map[Append[#, 1] &, intermediateResult3].Transpose[outputWeightsVariables]],
				
				(* Cross-entropy cost function arguments MUST be in interval {0, 1} *)
				"Cross-Entropy",
				mlp3Outputs = 0.5 * (Tanh[Map[Append[#, 1] &, intermediateResult3].Transpose[outputWeightsVariables]] + 1.0)
			]
		];
	    
	    Switch[costFunctionType,
	    	
	    	"SquaredError",
		    If[lambdaL2Regularization == 0.0,
	
				(* NO L2 regularization *)
				costFunction =
					Sum[
						Sum[
							(outputs[[i, k]] - mlp3Outputs[[i, k]])^2,
							
							{k, numberOfOutputs}
						],
							    
						{i, numberOfIOPairs}	
					]/numberOfIOPairs,
		    	
				(* L2 regularization *)
				costFunction =
					Sum[
						Sum[
							(outputs[[i, k]] - mlp3Outputs[[i, k]])^2,
							
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
							outputs[[i, k]] * Log[mlp3Outputs[[i, k]]] + (1.0 - outputs[[i, k]]) * Log[1.0 - mlp3Outputs[[i, k]]],
							
							{k, numberOfOutputs}
						],
							    
						{i, numberOfIOPairs}	
					]/numberOfIOPairs,
		    	
				(* L2 regularization *)
				costFunction =
					Sum[
						Sum[
							outputs[[i, k]] * Log[mlp3Outputs[[i, k]]] + (1.0 - outputs[[i, k]]) * Log[1.0 - mlp3Outputs[[i, k]]],
							
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
				AppendTo[trainingMeanSquaredErrorList, 
					{stepNumber, GetInternalMeanSquaredErrorOfMlp3[scaledTrainingSet, weights[[1]], weights[[2]], weights[[3]], weights[[4]], activationAndScaling]}
				];
				If[Length[scaledTestSet] > 0,
					(* Test set exists *)
					AppendTo[testMeanSquaredErrorList, 
						{stepNumber, GetInternalMeanSquaredErrorOfMlp3[scaledTestSet, weights[[1]], weights[[2]], weights[[3]], weights[[4]], activationAndScaling]}
					]
				],
				
				{i, Length[minimumInfo[[2, 1]]]}
			]
		];
			
		(* ----------------------------------------------------------------------------------------------------
		   Set results
		   ---------------------------------------------------------------------------------------------------- *)
		weights = GetWeightsVariables[numberOfInputs, numberOfHiddenNeurons, numberOfOutputs, wInputToHidden1, wHidden1ToHidden2, wHidden2ToHidden3, wHidden3ToOutput]/.weightsRules;

		(* End of training protocol *)
		lastTrainingStep = Last[trainingMeanSquaredErrorList];
		If[lastTrainingStep[[1]] < steps,
			AppendTo[trainingMeanSquaredErrorList, 
				{steps, GetInternalMeanSquaredErrorOfMlp3[scaledTrainingSet, weights[[1]], weights[[2]], weights[[3]], weights[[4]], activationAndScaling]}
			];
			If[Length[scaledTestSet] > 0,
				(* Test set exists *)
				AppendTo[testMeanSquaredErrorList, 
					{steps, GetInternalMeanSquaredErrorOfMlp3[scaledTestSet, weights[[1]], weights[[2]], weights[[3]], weights[[4]], activationAndScaling]}
				]
			]
		];
		
		(* Return mlp3Info *)
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

FitMlp3WithNMinimize[

	(* Training of mlp3 with NMinimize and "DifferentialEvolution".
	
	   Returns:
	   mlp3Info (see "Frequently used data structures") *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,

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
			hidden1WeightsVariables,
			hidden2WeightsVariables,
			hidden3WeightsVariables,
			i,
			inputs,
			intermediateResult1,
			intermediateResult2,
			intermediateResult3,
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
			mlp3Outputs,
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
			wInputToHidden1,
			wHidden1ToHidden2,
			wHidden2ToHidden3,
			wHidden3ToOutput,
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
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
	    minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
	    maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
	    reportIteration = Mlp3OptionReportIteration/.{opts}/.Options[Mlp3OptionsOptimization];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
	    weightsVariables = 
	    	GetWeightsVariables[
	    		numberOfInputs, 
	    		numberOfHiddenNeurons, 
	    		numberOfOutputs, 
	    		wInputToHidden1, 
	    		wHidden1ToHidden2, 
				wHidden2ToHidden3,
	    		wHidden3ToOutput
    		];
	    hidden1WeightsVariables = weightsVariables[[1]];
	    hidden2WeightsVariables = weightsVariables[[2]];
	    hidden3WeightsVariables = weightsVariables[[3]];
	    outputWeightsVariables = weightsVariables[[4]];

		weightsVariablesWithoutTrueUnitBias = 
			GetWeightsVariablesWithoutTrueUnitBias[
				numberOfInputs, 
				numberOfHiddenNeurons, 
				numberOfOutputs, 
				wInputToHidden1, 
				wHidden1ToHidden2, 
				wHidden2ToHidden3,
				wHidden3ToOutput
			];

	    (* Map: Add 'true unit' *)
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			intermediateResult1 = SigmoidFunction[Map[Append[#, 1] &, inputs].Transpose[hidden1WeightsVariables]],
			
			"Tanh",
			intermediateResult1 = Tanh[Map[Append[#, 1] &, inputs].Transpose[hidden1WeightsVariables]]
		];
		
		Switch[activationAndScaling[[1, 2]],
			
			"Sigmoid",
			intermediateResult2 = SigmoidFunction[Map[Append[#, 1] &, intermediateResult1].Transpose[hidden2WeightsVariables]],
			
			"Tanh",
			intermediateResult2 = Tanh[Map[Append[#, 1] &, intermediateResult1].Transpose[hidden2WeightsVariables]]
		];
		
		Switch[activationAndScaling[[1, 3]],
			
			"Sigmoid",
			intermediateResult3 = SigmoidFunction[Map[Append[#, 1] &, intermediateResult2].Transpose[hidden3WeightsVariables]],
			
			"Tanh",
			intermediateResult3 = Tanh[Map[Append[#, 1] &, intermediateResult2].Transpose[hidden3WeightsVariables]]
		];

		Switch[activationAndScaling[[1, 4]],
			
			"Sigmoid",
			mlp3Outputs = SigmoidFunction[Map[Append[#, 1] &, intermediateResult3].Transpose[outputWeightsVariables]],
			
			"Tanh",
			Switch[costFunctionType,
			
				"SquaredError",				
				mlp3Outputs = Tanh[Map[Append[#, 1] &, intermediateResult3].Transpose[outputWeightsVariables]],
				
				(* Cross-entropy cost function arguments MUST be in interval {0, 1} *)
				"Cross-Entropy",
				mlp3Outputs = 0.5 * (Tanh[Map[Append[#, 1] &, intermediateResult3].Transpose[outputWeightsVariables]] + 1.0)
			]
		];
	    
	    Switch[costFunctionType,
	    	
	    	"SquaredError",
		    If[lambdaL2Regularization == 0.0,
	
				(* NO L2 regularization *)
				costFunction =
					Sum[
						Sum[
							(outputs[[i, k]] - mlp3Outputs[[i, k]])^2,
							
							{k, numberOfOutputs}
						],
							    
						{i, numberOfIOPairs}	
					]/numberOfIOPairs,
		    	
				(* L2 regularization *)
				costFunction =
					Sum[
						Sum[
							(outputs[[i, k]] - mlp3Outputs[[i, k]])^2,
							
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
							outputs[[i, k]] * Log[mlp3Outputs[[i, k]]] + (1.0 - outputs[[i, k]]) * Log[1.0 - mlp3Outputs[[i, k]]],
							
							{k, numberOfOutputs}
						],
							    
						{i, numberOfIOPairs}	
					]/numberOfIOPairs,
		    	
				(* L2 regularization *)
				costFunction =
					Sum[
						Sum[
							outputs[[i, k]] * Log[mlp3Outputs[[i, k]]] + (1.0 - outputs[[i, k]]) * Log[1.0 - mlp3Outputs[[i, k]]],
							
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
				AppendTo[trainingMeanSquaredErrorList, 
					{stepNumber, GetInternalMeanSquaredErrorOfMlp3[scaledTrainingSet, weights[[1]], weights[[2]], weights[[3]], weights[[4]], activationAndScaling]}
				];
				If[Length[scaledTestSet] > 0,
					(* Test set exists *)
					AppendTo[testMeanSquaredErrorList, 
						{stepNumber, GetInternalMeanSquaredErrorOfMlp3[scaledTestSet, weights[[1]], weights[[2]], weights[[3]], weights[[4]], activationAndScaling]}
					]
				],
				
				{i, Length[minimumInfo[[2, 1]]]}
			]
		];
			
		(* ----------------------------------------------------------------------------------------------------
		   Set results
		   ---------------------------------------------------------------------------------------------------- *)
		weights = GetWeightsVariables[numberOfInputs, numberOfHiddenNeurons, numberOfOutputs, wInputToHidden1, wHidden1ToHidden2, wHidden2ToHidden3, wHidden3ToOutput]/.weightsRules;

		(* End of training protocol *)
		lastTrainingStep = Last[trainingMeanSquaredErrorList];
		If[lastTrainingStep[[1]] < steps,
			AppendTo[trainingMeanSquaredErrorList, 
				{steps, GetInternalMeanSquaredErrorOfMlp3[scaledTrainingSet, weights[[1]], weights[[2]], weights[[3]], weights[[4]], activationAndScaling]}
			];
			If[Length[scaledTestSet] > 0,
				(* Test set exists *)
				AppendTo[testMeanSquaredErrorList, 
					{steps, GetInternalMeanSquaredErrorOfMlp3[scaledTestSet, weights[[1]], weights[[2]], weights[[3]], weights[[4]], activationAndScaling]}
				]
			]
		];
		
		(* Return mlp3Info *)
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

FitSingleMlp3[

	(* Training of single Mlp3.

	   Returns:
	   mlp3Info (see "Frequently used data structures") *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			unusedOptionParameter22,
			initialWeights,
			weightsValueLimit,
			unusedOptionParameter11,
			unusedOptionParameter12,
			maximumNumberOfIterations,
			minimizationPrecision,
			unusedOptionParameter13,
			unusedOptionParameter23,
			numberOfIterationsToImprove,
			unusedOptionParameter21,
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
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		reportIteration = Mlp3OptionReportIteration/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		
		(* ----------------------------------------------------------------------------------------------------
		   Switch training method
		   ---------------------------------------------------------------------------------------------------- *)
		Switch[optimizationMethod,
			
			"FindMinimum",
			Return[
				FitMlp3WithFindMinimum[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					Mlp3OptionInitialWeights -> initialWeights,
	    			Mlp3OptionMinimizationPrecision -> minimizationPrecision,
	    			Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionReportIteration -> reportIteration,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp3OptionActivationAndScaling -> activationAndScaling,
	    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp3OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				]
			],
			
			"NMinimize",
			Return[
				FitMlp3WithNMinimize[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
	    			Mlp3OptionMinimizationPrecision -> minimizationPrecision,
	    			Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionReportIteration -> reportIteration,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp3OptionActivationAndScaling -> activationAndScaling,
	    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp3OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				]
			]
		]
	];

GetBestMlp3ClassOptimization[

	(* Returns best training set optimization result of mlp3 for classification.

	   Returns: 
	   Best index for classification *)


	(* mlp3TrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlp3InfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlp3InfoList: List with mlp3Info
	   mlp3InfoList[[i]] refers to optimization step i *)
	mlp3TrainOptimization_,
	
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
				GetBestMlp3ClassOptimizationPC[
					mlp3TrainOptimization,
					UtilityOptionBestOptimization -> bestOptimization
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetBestMlp3ClassOptimizationSC[
					mlp3TrainOptimization,
					UtilityOptionBestOptimization -> bestOptimization
				]
			]
		]
	];

GetBestMlp3ClassOptimizationSC[

	(* Returns best training set optimization result of mlp3 for classification.

	   Returns: 
	   Best index for classification *)


	(* mlp3TrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlp3InfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlp3InfoList: List with mlp3Info
	   mlp3InfoList[[i]] refers to optimization step i *)
	mlp3TrainOptimization_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			k,
			trainingAndTestSetList,
			mlp3InfoList,
			maximumCorrectClassificationInPercent,
			mlp3Info,
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
			trainingAndTestSetList = mlp3TrainOptimization[[3]];
			mlp3InfoList = mlp3TrainOptimization[[4]];
			maximumCorrectClassificationInPercent = -1.0;
			Do[
				testSet = trainingAndTestSetList[[k, 2]];
				mlp3Info = mlp3InfoList[[k]];
				correctClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[testSet, mlp3Info];
				If[correctClassificationInPercent > maximumCorrectClassificationInPercent,
					maximumCorrectClassificationInPercent = correctClassificationInPercent;
					bestIndex = k
				],
				
				{k, Length[mlp3InfoList]}
			],
			
			(* ------------------------------------------------------------------------------- *)			
			"MinimumDeviation",
			trainingAndTestSetList = mlp3TrainOptimization[[3]];
			mlp3InfoList = mlp3TrainOptimization[[4]];
			minimumDeviation = Infinity;
			Do[
				trainingSet = trainingAndTestSetList[[k, 1]];
				testSet = trainingAndTestSetList[[k, 2]];
				mlp3Info = mlp3InfoList[[k]];
				trainingSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[trainingSet, mlp3Info];
				testSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[testSet, mlp3Info];
				deviation = Abs[testSetCorrectClassificationInPercent - trainingSetCorrectClassificationInPercent];
				If[deviation < minimumDeviation || (deviation == minimumDeviation && testSetCorrectClassificationInPercent < bestTestSetCorrectClassificationInPercent),
					minimumDeviation = deviation;
					bestTestSetCorrectClassificationInPercent = testSetCorrectClassificationInPercent;
					bestIndex = k
				],
				
				{k, Length[mlp3InfoList]}
			]
		];

		Return[bestIndex]
	];

GetBestMlp3ClassOptimizationPC[

	(* Returns best training set optimization result of mlp3 for classification.

	   Returns: 
	   Best index for classification *)


	(* mlp3TrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlp3InfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlp3InfoList: List with mlp3Info
	   mlp3InfoList[[i]] refers to optimization step i *)
	mlp3TrainOptimization_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			k,
			trainingAndTestSetList,
			mlp3InfoList,
			maximumCorrectClassificationInPercent,
			mlp3Info,
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
			trainingAndTestSetList = mlp3TrainOptimization[[3]];
			mlp3InfoList = mlp3TrainOptimization[[4]];
			maximumCorrectClassificationInPercent = -1.0;
			
			ParallelNeeds[{"CIP`Mlp3`", "CIP`DataTransformation`", "CIP`Utility`"}];
			DistributeDefinitions[trainingAndTestSetList, mlp3InfoList];
			
			correctClassificationInPercentList = ParallelTable[
				testSet = trainingAndTestSetList[[k, 2]];
				mlp3Info = mlp3InfoList[[k]];
				
				CalculateMlp3CorrectClassificationInPercent[testSet, mlp3Info],
				
				{k, Length[mlp3InfoList]}
			];
			
			Do[
				If[correctClassificationInPercentList[[k]] > maximumCorrectClassificationInPercent,
					maximumCorrectClassificationInPercent = correctClassificationInPercentList[[k]];
					bestIndex = k
				],
				
				{k, Length[mlp3InfoList]}
			],
			
			(* ------------------------------------------------------------------------------- *)			
			"MinimumDeviation",
			trainingAndTestSetList = mlp3TrainOptimization[[3]];
			mlp3InfoList = mlp3TrainOptimization[[4]];
			minimumDeviation = Infinity;
			
			ParallelNeeds[{"CIP`Mlp3`", "CIP`DataTransformation`", "CIP`Utility`"}];
			DistributeDefinitions[trainingAndTestSetList, mlp3InfoList];
			
			listOfTestSetCorrectClassificationInPercentAndDeviation = ParallelTable[
				trainingSet = trainingAndTestSetList[[k, 1]];
				testSet = trainingAndTestSetList[[k, 2]];
				mlp3Info = mlp3InfoList[[k]];
				trainingSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[trainingSet, mlp3Info];
				testSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[testSet, mlp3Info];
				deviation = Abs[testSetCorrectClassificationInPercent - trainingSetCorrectClassificationInPercent];
				
				{
					testSetCorrectClassificationInPercent,
					deviation
				},
				
				{k, Length[mlp3InfoList]}
			];
			
			Do[
				If[listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]] < minimumDeviation || (listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]] == minimumDeviation && listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 1]] < bestTestSetCorrectClassificationInPercent),
					minimumDeviation = listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]];
					bestTestSetCorrectClassificationInPercent = listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 1]];
					bestIndex = k
				],
				
				{k, Length[mlp3InfoList]}
			]
		];

		Return[bestIndex]
	];

GetBestMlp3RegressOptimization[

	(* Returns best optimization result of mlp3 for regression.

	   Returns: 
	   Best index for regression *)


	(* mlp3TrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlp3InfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlp3InfoList: List with mlp3Info
	   mlp3InfoList[[i]] refers to optimization step i *)
	mlp3TrainOptimization_,
	
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
				mlp3TrainOptimization, 
				UtilityOptionBestOptimization -> bestOptimization
			]
		]
	];

GetInternalMeanSquaredErrorOfMlp3[

	(* Calculates mean squared error of specified data set for Mlp3 with specified weights

	   Returns:
	   Mean squared error of data set *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output: {outputComponent1, outputComponent2, ...})
	   NOTE: Each component must be in [0, 1] *)
    dataSet_,
    
    (* Weights from input to hidden units *)
    hidden1Weights_/;MatrixQ[hidden1Weights, NumberQ],

    (* Weights from hidden1 to hidden2 units *)
    hidden2Weights_/;MatrixQ[hidden2Weights, NumberQ],

    (* Weights from hidden2 to hidden3 units *)
    hidden3Weights_/;MatrixQ[hidden3Weights, NumberQ],
    
    (* Weights from hidden3 to output units *)
    outputWeights_/;MatrixQ[outputWeights, NumberQ],
    
    (* Activation and scaling, see Mlp3OptionActivationAndScaling *)
    activationAndScaling_
    
	] :=
  
	Module[
    
		{
			errors,
			hidden1,
			hidden2,
			hidden3,
			inputs,
			machineOutputs,
			outputs
		},
    
		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];
		outputs = CIP`Utility`GetOutputsOfDataSet[dataSet];

	    (* Add 'true unit' to inputs *)
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			hidden1 = SigmoidFunction[Map[Append[#, 1.0] &, inputs].Transpose[hidden1Weights]],
			
			"Tanh",
			hidden1 = Tanh[Map[Append[#, 1.0] &, inputs].Transpose[hidden1Weights]]
		];

	    (* Add 'true unit' to hidden1 *)
		Switch[activationAndScaling[[1, 2]],
			
			"Sigmoid",
			hidden2 = SigmoidFunction[Map[Append[#, 1.0] &, hidden1].Transpose[hidden2Weights]],
			
			"Tanh",
			hidden2 = Tanh[Map[Append[#, 1.0] &, hidden1].Transpose[hidden2Weights]]
		];

	    (* Add 'true unit' to hidden2 *)
		Switch[activationAndScaling[[1, 3]],
			
			"Sigmoid",
			hidden3 = SigmoidFunction[Map[Append[#, 1.0] &, hidden2].Transpose[hidden3Weights]],
			
			"Tanh",
			hidden3 = Tanh[Map[Append[#, 1.0] &, hidden2].Transpose[hidden3Weights]]
		];

	    (* Add 'true unit' to hidden3 *)
		Switch[activationAndScaling[[1, 4]],
			
			"Sigmoid",
			machineOutputs = SigmoidFunction[Map[Append[#, 1.0] &, hidden3].Transpose[outputWeights]],
			
			"Tanh",
			machineOutputs = Tanh[Map[Append[#, 1.0] &, hidden3].Transpose[outputWeights]]
		];

	    errors = outputs - machineOutputs;
        Return[Apply[Plus, errors^2, {0,1}]/Length[dataSet]]
	];

GetInternalMlp3Output[

	(* Calculates internal output for specified input of mlp3 with specified weights.

	   Returns:
	   output: {valueOfOutput1, valueOfOutput2, ...} *)

    
    (* input: {valueForInput1, valueForInput1, ...} *)
    input_/;VectorQ[input, NumberQ],
    
    (* Weights from input to hidden units *)
    hidden1Weights_/;MatrixQ[hidden1Weights, NumberQ],

    (* Weights from hidden1 to hidden2 units *)
    hidden2Weights_/;MatrixQ[hidden2Weights, NumberQ],

    (* Weights from hidden2 to hidden3 units *)
    hidden3Weights_/;MatrixQ[hidden3Weights, NumberQ],
    
    (* Weights from hidden3 to output units *)
    outputWeights_/;MatrixQ[outputWeights, NumberQ],
    
    (* Activation and scaling, see Mlp3OptionActivationAndScaling *)
    activationAndScaling_
    
	] :=
  
	Module[
    
		{
	      hidden1,
	      hidden2,
	      hidden3,
	      internalInputs,
	      trueUnitHidden,
	      outputs
		},
    
		(* Add 'true unit' to inputs *)
		internalInputs = Append[input, 1.0];
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			hidden1 = SigmoidFunction[internalInputs.Transpose[hidden1Weights]],
			
			"Tanh",
			hidden1 = Tanh[internalInputs.Transpose[hidden1Weights]]
		];
	    
	    (* Add 'true unit' to hidden1 *)
		trueUnitHidden = Append[hidden1, 1.0];
		Switch[activationAndScaling[[1, 2]],
			
			"Sigmoid",
			hidden2 = SigmoidFunction[trueUnitHidden.Transpose[hidden2Weights]],
			
			"Tanh",
			hidden2 = Tanh[trueUnitHidden.Transpose[hidden2Weights]]
		];
	    
	    (* Add 'true unit' to hidden2 *)
		trueUnitHidden = Append[hidden2, 1.0];
		Switch[activationAndScaling[[1, 3]],
			
			"Sigmoid",
			hidden3 = SigmoidFunction[trueUnitHidden.Transpose[hidden3Weights]],
			
			"Tanh",
			hidden3 = Tanh[trueUnitHidden.Transpose[hidden3Weights]]
		];

	    (* Add 'true unit' to hidden3 *)
		trueUnitHidden = Append[hidden3, 1.0];
		Switch[activationAndScaling[[1, 4]],
			
			"Sigmoid",
			outputs = SigmoidFunction[trueUnitHidden.Transpose[outputWeights]],
			
			"Tanh",
			outputs = Tanh[trueUnitHidden.Transpose[outputWeights]]
		];
		
		Return[outputs];
    ];

GetInternalMlp3Outputs[

	(* Calculates internal outputs for specified inputs for mlp3 with specified weights.

	   Returns:
	   outputs: {output1, output2, ...} 
	   output: {valueOfOutput1, valueOfOutput2, ...} *)

    
    (* inputs: {input1, input2, ...} 
       input: {valueForInput1, valueForInput1, ...} *)
    inputs_/;MatrixQ[inputs, NumberQ],
    
    (* Weights from input to hidden units *)
    hidden1Weights_/;MatrixQ[hidden1Weights, NumberQ],

    (* Weights from hidden1 to hidden2 units *)
    hidden2Weights_/;MatrixQ[hidden2Weights, NumberQ],

    (* Weights from hidden2 to hidden3 units *)
    hidden3Weights_/;MatrixQ[hidden3Weights, NumberQ],
    
    (* Weights from hidden3 to output units *)
    outputWeights_/;MatrixQ[outputWeights, NumberQ],
    
    (* Activation and scaling, see Mlp3OptionActivationAndScaling *)
    activationAndScaling_
    
	] :=
  
	Module[
    
		{
	      hidden1,
	      hidden2,
	      hidden3,
	      internalInputs,
	      trueUnitHidden,
	      outputs
		},
    
		(* Add 'true unit' to inputs *)
		internalInputs = Map[Append[#, 1.0] &, inputs];
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			hidden1 = SigmoidFunction[internalInputs.Transpose[hidden1Weights]],
			
			"Tanh",
			hidden1 = Tanh[internalInputs.Transpose[hidden1Weights]]
		];
	    
	    (* Add 'true unit' to hidden1 *)
		trueUnitHidden = Map[Append[#, 1.0] &, hidden1];
		Switch[activationAndScaling[[1, 2]],
			
			"Sigmoid",
			hidden2 = SigmoidFunction[trueUnitHidden.Transpose[hidden2Weights]],
			
			"Tanh",
			hidden2 = Tanh[trueUnitHidden.Transpose[hidden2Weights]]
		];
	    
	    (* Add 'true unit' to hidden2 *)
		trueUnitHidden = Map[Append[#, 1.0] &, hidden2];
		Switch[activationAndScaling[[1, 3]],
			
			"Sigmoid",
			hidden3 = SigmoidFunction[trueUnitHidden.Transpose[hidden3Weights]],
			
			"Tanh",
			hidden3 = Tanh[trueUnitHidden.Transpose[hidden3Weights]]
		];
	    
	    (* Add 'true unit' to hidden3 *)
		trueUnitHidden = Map[Append[#, 1.0] &, hidden3];
		Switch[activationAndScaling[[1, 4]],
			
			"Sigmoid",
			outputs = SigmoidFunction[trueUnitHidden.Transpose[outputWeights]],
			
			"Tanh",
			outputs = Tanh[trueUnitHidden.Transpose[outputWeights]]
		];
		
		Return[outputs];
    ];

GetNumberOfHiddenNeurons[

	(* Returns number of hidden neurons for specified mlp3Info.

	   Returns:
	   numberOfHiddenNeurons: {numberOfHidden1Neurons, numberOfHidden2Neurons, numberOfHidden3Neurons} *)

    
  	(* See "Frequently used data structures" *)
    mlp3Info_
    
	] :=
  
	Module[
    
		{},
		
		Return[
			GetMlp3Structure[mlp3Info][[2]]
		]
	];

GetMlp3InputInclusionClass[

	(* Analyzes relevance of input components by successive get-one-in for classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlp3InputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mlp3Info}
	   trainingSetResult: {numberOfIncludedInputs, (best) classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, (best) classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlp3s,
			optimizationMethod,
			initialWeights,
			initialNetworks,
			weightsValueLimit,
			minimizationPrecision,
			maximumNumberOfIterations,
			numberOfIterationsToImprove,
			unusedOptionParameter11,
			unusedOptionParameter12,
			unusedOptionParameter13,
			unusedOptionParameter21,
			unusedOptionParameter22,
			unusedOptionParameter23,
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
		(* Mlp3 options *)   
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
				GetMlp3InputInclusionCalculationPC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
    				Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
	 				Mlp3OptionReportIteration -> 0,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
    				Mlp3OptionActivationAndScaling -> activationAndScaling,
    				Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
    				Mlp3OptionCostFunctionType -> costFunctionType,
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
				GetMlp3InputInclusionCalculationSC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
    				Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
	 				Mlp3OptionReportIteration -> 0,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
    				Mlp3OptionActivationAndScaling -> activationAndScaling,
    				Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
    				Mlp3OptionCostFunctionType -> costFunctionType,
    				DataTransformationOptionNormalizationType -> normalizationType,
    				UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			]
		]
	];

GetMlp3InputInclusionRegress[

	(* Analyzes relevance of input components by successive get-one-in for regression.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlp3InputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mlp3Info}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlp3s,
			optimizationMethod,
			initialWeights,
			initialNetworks,
			weightsValueLimit,
			minimizationPrecision,
			maximumNumberOfIterations,
			numberOfIterationsToImprove,
			unusedOptionParameter11,
			unusedOptionParameter12,
			unusedOptionParameter13,
			unusedOptionParameter21,
			unusedOptionParameter22,
			unusedOptionParameter23,
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
		(* Mlp3 options *)   
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
				GetMlp3InputInclusionCalculationPC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
    				Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
	 				Mlp3OptionReportIteration -> 0,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
    				Mlp3OptionActivationAndScaling -> activationAndScaling,
    				Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
    				Mlp3OptionCostFunctionType -> costFunctionType,
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
				GetMlp3InputInclusionCalculationSC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
    				Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
	 				Mlp3OptionReportIteration -> 0,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
    				Mlp3OptionActivationAndScaling -> activationAndScaling,
    				Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
    				Mlp3OptionCostFunctionType -> costFunctionType,
    				DataTransformationOptionNormalizationType -> normalizationType,
    				UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			]
		]
	];

GetMlp3InputInclusionCalculationSC[

	(* Analyzes relevance of input components by successive get-one-in for regression and classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlp3InputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mlp3Info}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlp3s,
			optimizationMethod,
			initialWeights,
			initialNetworks,
			weightsValueLimit,
			minimizationPrecision,
			maximumNumberOfIterations,
			numberOfIterationsToImprove,
			unusedOptionParameter11,
			unusedOptionParameter12,
			unusedOptionParameter13,
			unusedOptionParameter21,
			unusedOptionParameter22,
			unusedOptionParameter23,
			randomValueInitialization,
			activationAndScaling,
			normalizationType,
			currentIncludedInputComponentList,
			i,
			k,
			numberOfInputs,
			numberOfIncludedInputs,
			mlp3InputComponentRelevanceList,
	        mlp3Info,
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
		(* Mlp3 options *)   
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
		mlp3InputComponentRelevanceList = {};
    
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
					mlp3Info = 
						FitMlp3[
							trainingSet,
							numberOfHiddenNeurons,
							Mlp3OptionMultipleMlp3s -> multipleMlp3s,
			    			Mlp3OptionOptimizationMethod -> optimizationMethod,
							Mlp3OptionInitialWeights -> initialWeights,
							Mlp3OptionInitialNetworks -> initialNetworks,
							Mlp3OptionWeightsValueLimit -> weightsValueLimit,
							Mlp3OptionMinimizationPrecision -> minimizationPrecision,
							Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
							Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
				 			Mlp3OptionReportIteration -> 0,
							Mlp3OptionUnused11 -> unusedOptionParameter11,
							Mlp3OptionUnused12 -> unusedOptionParameter12,
							Mlp3OptionUnused13 -> unusedOptionParameter13,
							Mlp3OptionUnused21 -> unusedOptionParameter21,
							Mlp3OptionUnused22 -> unusedOptionParameter22,
							Mlp3OptionUnused23 -> unusedOptionParameter23,
			    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
			    			Mlp3OptionActivationAndScaling -> activationAndScaling,
			    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
			    			Mlp3OptionCostFunctionType -> costFunctionType,
			    			DataTransformationOptionNormalizationType -> normalizationType
						];
					If[Length[testSet] > 0,
            
						testSetRmse = CalculateMlp3DataSetRmse[testSet, mlp3Info];
						AppendTo[rmseList,{testSetRmse, i}],
          
						trainingSetRmse = CalculateMlp3DataSetRmse[trainingSet, mlp3Info];
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
			mlp3Info = 
				FitMlp3[
					trainingSet,
					numberOfHiddenNeurons,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
	    			Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp3OptionReportIteration -> 0,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp3OptionActivationAndScaling -> activationAndScaling,
	    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp3OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMlp3DataSetRmse[trainingSet, mlp3Info];
					currentTestSetRmse = CalculateMlp3DataSetRmse[testSet, mlp3Info];
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
							mlp3Info
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMlp3DataSetRmse[trainingSet, mlp3Info];
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
							mlp3Info
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[trainingSet, mlp3Info];
					currentTestSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[testSet, mlp3Info];
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
							mlp3Info
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[trainingSet, mlp3Info];
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
							mlp3Info
						}
				]
			];	

			AppendTo[mlp3InputComponentRelevanceList, relevance];
			includedInputComponentList = currentIncludedInputComponentList,
			
			{k, Length[numberOfInclusionsPerStepList]}
		];
		
		Return[mlp3InputComponentRelevanceList]
	];

GetMlp3InputInclusionCalculationPC[

	(* Analyzes relevance of input components by successive get-one-in for regression and classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlp3InputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mlp3Info}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlp3s,
			optimizationMethod,
			initialWeights,
			initialNetworks,
			weightsValueLimit,
			minimizationPrecision,
			maximumNumberOfIterations,
			numberOfIterationsToImprove,
			unusedOptionParameter11,
			unusedOptionParameter12,
			unusedOptionParameter13,
			unusedOptionParameter21,
			unusedOptionParameter22,
			unusedOptionParameter23,
			randomValueInitialization,
			activationAndScaling,
			normalizationType,
			currentIncludedInputComponentList,
			i,
			k,
			numberOfInputs,
			numberOfIncludedInputs,
			mlp3InputComponentRelevanceList,
	        mlp3Info,
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
		(* Mlp3 options *)   
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
		mlp3InputComponentRelevanceList = {};
    	
    	ParallelNeeds[{"CIP`Mlp3`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[multipleMlp3s, optimizationMethod, initialWeights,
			initialNetworks, weightsValueLimit, minimizationPrecision, maximumNumberOfIterations, numberOfIterationsToImprove,
			unusedOptionParameter11, unusedOptionParameter12, unusedOptionParameter13, unusedOptionParameter21, unusedOptionParameter22,
			unusedOptionParameter23, randomValueInitialization, activationAndScaling, normalizationType, lambdaL2Regularization, 
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
						
						mlp3Info = 
							FitMlp3[
								trainingSet,
								numberOfHiddenNeurons,
								Mlp3OptionMultipleMlp3s -> multipleMlp3s,
				    			Mlp3OptionOptimizationMethod -> optimizationMethod,
								Mlp3OptionInitialWeights -> initialWeights,
								Mlp3OptionInitialNetworks -> initialNetworks,
								Mlp3OptionWeightsValueLimit -> weightsValueLimit,
								Mlp3OptionMinimizationPrecision -> minimizationPrecision,
								Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
								Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
				 				Mlp3OptionReportIteration -> 0,
								Mlp3OptionUnused11 -> unusedOptionParameter11,
								Mlp3OptionUnused12 -> unusedOptionParameter12,
								Mlp3OptionUnused13 -> unusedOptionParameter13,
								Mlp3OptionUnused21 -> unusedOptionParameter21,
								Mlp3OptionUnused22 -> unusedOptionParameter22,
								Mlp3OptionUnused23 -> unusedOptionParameter23,
				    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
			    				Mlp3OptionActivationAndScaling -> activationAndScaling,
			    				Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
			    				Mlp3OptionCostFunctionType -> costFunctionType,
			    				DataTransformationOptionNormalizationType -> normalizationType
							];
						
						If[Length[testSet] > 0,
            
							testSetRmse = CalculateMlp3DataSetRmse[testSet, mlp3Info];
							{testSetRmse, i},
          	
							trainingSetRmse = CalculateMlp3DataSetRmse[trainingSet, mlp3Info];
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
			mlp3Info = 
				FitMlp3[
					trainingSet,
					numberOfHiddenNeurons,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
	    			Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp3OptionReportIteration -> 0,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp3OptionActivationAndScaling -> activationAndScaling,
	    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp3OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionsParallelization -> "ParallelCalculation"
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMlp3DataSetRmse[trainingSet, mlp3Info];
					currentTestSetRmse = CalculateMlp3DataSetRmse[testSet, mlp3Info];
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
							mlp3Info
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMlp3DataSetRmse[trainingSet, mlp3Info];
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
							mlp3Info
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[trainingSet, mlp3Info];
					currentTestSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[testSet, mlp3Info];
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
							mlp3Info
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[trainingSet, mlp3Info];
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
							mlp3Info
						}
				]
			];	

			AppendTo[mlp3InputComponentRelevanceList, relevance];
			includedInputComponentList = currentIncludedInputComponentList,
			
			{k, Length[numberOfInclusionsPerStepList]}
		];
		
		Return[mlp3InputComponentRelevanceList]
	];

GetMlp3InputRelevanceClass[

	(* Analyzes relevance of input components by successive leave-one-out for classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlp3InputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlp3Info}
	   trainingSetResult: {numberOfRemovedInputs, (best) classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, (best) classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlp3s,
			optimizationMethod,
			initialWeights,
			initialNetworks,
			weightsValueLimit,
			minimizationPrecision,
			maximumNumberOfIterations,
			numberOfIterationsToImprove,
			unusedOptionParameter11,
			unusedOptionParameter12,
			unusedOptionParameter13,
			unusedOptionParameter21,
			unusedOptionParameter22,
			unusedOptionParameter23,
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
		(* Mlp3 options *)   
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
				GetMlp3InputRelevanceCalculationPC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
	    			Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp3OptionReportIteration -> 0,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
	    			Mlp3OptionActivationAndScaling -> activationAndScaling,
	    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp3OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetMlp3InputRelevanceCalculationSC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
	    			Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp3OptionReportIteration -> 0,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
	    			Mlp3OptionActivationAndScaling -> activationAndScaling,
	    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp3OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			]
		]
	];

GetMlp3ClassRelevantComponents[

	(* Returns most-to-least-relevance sorted components from mlp3InputComponentRelevanceListForClassification.

	   Returns: Returns most-to-least-relevance sorted components *)


	(* mlp3InputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlp3Info}
	   trainingSetResult: {numberOfRemovedInputs, (best) correct classification in percent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best correct classification in percent of test set} *)
	mlp3InputComponentRelevanceListForClassification_,
	
	(* Number of components to return *)
	numberOfComponents_?IntegerQ
    
	] :=
  
	Module[
    
		{},

		Return[
			CIP`Graphics`GetClassRelevantComponents[mlp3InputComponentRelevanceListForClassification, numberOfComponents]
		]
	];

GetMlp3InputRelevanceRegress[

	(* Analyzes relevance of input components by successive leave-one-out for regression.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlp3InputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlp3Info}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlp3s,
			optimizationMethod,
			initialWeights,
			initialNetworks,
			weightsValueLimit,
			minimizationPrecision,
			maximumNumberOfIterations,
			numberOfIterationsToImprove,
			unusedOptionParameter11,
			unusedOptionParameter12,
			unusedOptionParameter13,
			unusedOptionParameter21,
			unusedOptionParameter22,
			unusedOptionParameter23,
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
		(* Mlp3 options *)   
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
				GetMlp3InputRelevanceCalculationPC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
	    			Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp3OptionReportIteration -> 0,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
	    			Mlp3OptionActivationAndScaling -> activationAndScaling,
	    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp3OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetMlp3InputRelevanceCalculationSC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
	    			Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp3OptionReportIteration -> 0,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
	    			Mlp3OptionActivationAndScaling -> activationAndScaling,
	    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp3OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			]
		]
	];

GetMlp3RegressRelevantComponents[

	(* Returns most-to-least-relevance sorted components from mlp3InputComponentRelevanceListForRegression.

	   Returns: Returns most-to-least-relevance sorted components *)


	(* mlp3InputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlp3Info}
	   trainingSetResult: {numberOfRemovedInputs, (best) RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)
	mlp3InputComponentRelevanceListForRegression_,
	
	(* Number of components to return *)
	numberOfComponents_?IntegerQ
    
	] :=
  
	Module[
    
		{},

		Return[
			CIP`Graphics`GetRegressRelevantComponents[mlp3InputComponentRelevanceListForRegression, numberOfComponents]
		]
	];

GetMlp3InputRelevanceCalculationSC[

	(* Analyzes relevance of input components by successive leave-one-out for regression and classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlp3InputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlp3Info}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlp3s,
			optimizationMethod,
			initialWeights,
			initialNetworks,
			weightsValueLimit,
			minimizationPrecision,
			maximumNumberOfIterations,
			numberOfIterationsToImprove,
			unusedOptionParameter11,
			unusedOptionParameter12,
			unusedOptionParameter13,
			unusedOptionParameter21,
			unusedOptionParameter22,
			unusedOptionParameter23,
			randomValueInitialization,
			activationAndScaling,
			normalizationType,
			currentRemovedInputComponentList,
			i,
			k,
			numberOfInputs,
			numberOfRemovedInputs,
			mlp3InputComponentRelevanceList,
	        mlp3Info,
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
		(* Mlp3 options *)   
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
		mlp3InputComponentRelevanceList = {};
    
		(* Result for no removal *)
		trainingSet = trainingAndTestSet[[1]];
		testSet = trainingAndTestSet[[2]];
		mlp3Info = 
			FitMlp3[
				trainingSet,
				numberOfHiddenNeurons,
				Mlp3OptionMultipleMlp3s -> multipleMlp3s,
    			Mlp3OptionOptimizationMethod -> optimizationMethod,
				Mlp3OptionInitialWeights -> initialWeights,
				Mlp3OptionInitialNetworks -> initialNetworks,
				Mlp3OptionWeightsValueLimit -> weightsValueLimit,
				Mlp3OptionMinimizationPrecision -> minimizationPrecision,
				Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
				Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
	 			Mlp3OptionReportIteration -> 0,
				Mlp3OptionUnused11 -> unusedOptionParameter11,
				Mlp3OptionUnused12 -> unusedOptionParameter12,
				Mlp3OptionUnused13 -> unusedOptionParameter13,
				Mlp3OptionUnused21 -> unusedOptionParameter21,
				Mlp3OptionUnused22 -> unusedOptionParameter22,
				Mlp3OptionUnused23 -> unusedOptionParameter23,
    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
    			Mlp3OptionActivationAndScaling -> activationAndScaling,
    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
    			Mlp3OptionCostFunctionType -> costFunctionType,
    			DataTransformationOptionNormalizationType -> normalizationType
			];
		
		initialTrainingSetRmse = CalculateMlp3DataSetRmse[trainingSet, mlp3Info];
		If[isRegression,
			
			(* Regression*)
			If[Length[testSet] > 0,
				
				(* Regression WITH test set*)
				initialTestSetRmse = CalculateMlp3DataSetRmse[testSet, mlp3Info];
				If[isIntermediateOutput,
					Print["initialTrainingSetRmse = ", initialTrainingSetRmse];
					Print["initialTestSetRmse     = ", initialTestSetRmse]
				];
				relevance = 
					{
						{0.0, initialTrainingSetRmse},
						{0.0, initialTestSetRmse},
						{}, 
						mlp3Info
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
						mlp3Info
					}
			],
			
			(* Classification *)
			initialTrainingSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[trainingSet, mlp3Info];
			If[Length[testSet] > 0,
				
				(* Classification WITH test set*)
				initialTestSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[testSet, mlp3Info];
				If[isIntermediateOutput,
					Print["initialTrainingSetCorrectClassificationInPercent = ", initialTrainingSetCorrectClassificationInPercent];
					Print["initialTestSetCorrectClassificationInPercent     = ", initialTestSetCorrectClassificationInPercent]
				];
				relevance = 
					{
						{0.0, initialTrainingSetCorrectClassificationInPercent},
						{0.0, initialTestSetCorrectClassificationInPercent},
						{}, 
						mlp3Info
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
						mlp3Info
					}
			]
		];	
		
		AppendTo[mlp3InputComponentRelevanceList, relevance];
    
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
					mlp3Info = 
						FitMlp3[
							trainingSet,
							numberOfHiddenNeurons,
							Mlp3OptionMultipleMlp3s -> multipleMlp3s,
			    			Mlp3OptionOptimizationMethod -> optimizationMethod,
							Mlp3OptionInitialWeights -> initialWeights,
							Mlp3OptionInitialNetworks -> initialNetworks,
							Mlp3OptionWeightsValueLimit -> weightsValueLimit,
							Mlp3OptionMinimizationPrecision -> minimizationPrecision,
							Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
							Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
				 			Mlp3OptionReportIteration -> 0,
							Mlp3OptionUnused11 -> unusedOptionParameter11,
							Mlp3OptionUnused12 -> unusedOptionParameter12,
							Mlp3OptionUnused13 -> unusedOptionParameter13,
							Mlp3OptionUnused21 -> unusedOptionParameter21,
							Mlp3OptionUnused22 -> unusedOptionParameter22,
							Mlp3OptionUnused23 -> unusedOptionParameter23,
			    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
			    			Mlp3OptionActivationAndScaling -> activationAndScaling,
			    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
			    			Mlp3OptionCostFunctionType -> costFunctionType,
			    			DataTransformationOptionNormalizationType -> normalizationType
						];
					If[Length[testSet] > 0,
            
						testSetRmse = CalculateMlp3DataSetRmse[testSet, mlp3Info];
						AppendTo[rmseList,{testSetRmse, i}],
          
						trainingSetRmse = CalculateMlp3DataSetRmse[trainingSet, mlp3Info];
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
			mlp3Info = 
				FitMlp3[
					trainingSet,
					numberOfHiddenNeurons,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
	    			Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp3OptionReportIteration -> 0,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp3OptionActivationAndScaling -> activationAndScaling,
	    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp3OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMlp3DataSetRmse[trainingSet, mlp3Info];
					currentTestSetRmse = CalculateMlp3DataSetRmse[testSet, mlp3Info];
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
							mlp3Info
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMlp3DataSetRmse[trainingSet, mlp3Info];
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
							mlp3Info
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[trainingSet, mlp3Info];
					currentTestSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[testSet, mlp3Info];
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
							mlp3Info
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[trainingSet, mlp3Info];
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
							mlp3Info
						}
				]
			];	

			AppendTo[mlp3InputComponentRelevanceList, relevance];
			removedInputComponentList = currentRemovedInputComponentList,
			
			{k, Length[numberOfExclusionsPerStepList]}
		];
		
		Return[mlp3InputComponentRelevanceList]
	];

GetMlp3InputRelevanceCalculationPC[

	(* Analyzes relevance of input components by successive leave-one-out for regression and classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlp3InputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlp3Info}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlp3s,
			optimizationMethod,
			initialWeights,
			initialNetworks,
			weightsValueLimit,
			minimizationPrecision,
			maximumNumberOfIterations,
			numberOfIterationsToImprove,
			unusedOptionParameter11,
			unusedOptionParameter12,
			unusedOptionParameter13,
			unusedOptionParameter21,
			unusedOptionParameter22,
			unusedOptionParameter23,
			randomValueInitialization,
			activationAndScaling,
			normalizationType,
			currentRemovedInputComponentList,
			i,
			k,
			numberOfInputs,
			numberOfRemovedInputs,
			mlp3InputComponentRelevanceList,
	        mlp3Info,
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
		(* Mlp3 options *)   
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
		mlp3InputComponentRelevanceList = {};
    
		(* Result for no removal *)
		trainingSet = trainingAndTestSet[[1]];
		testSet = trainingAndTestSet[[2]];
		mlp3Info = 
			FitMlp3[
				trainingSet,
				numberOfHiddenNeurons,
				Mlp3OptionMultipleMlp3s -> multipleMlp3s,
    			Mlp3OptionOptimizationMethod -> optimizationMethod,
				Mlp3OptionInitialWeights -> initialWeights,
				Mlp3OptionInitialNetworks -> initialNetworks,
				Mlp3OptionWeightsValueLimit -> weightsValueLimit,
				Mlp3OptionMinimizationPrecision -> minimizationPrecision,
				Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
				Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
	 			Mlp3OptionReportIteration -> 0,
				Mlp3OptionUnused11 -> unusedOptionParameter11,
				Mlp3OptionUnused12 -> unusedOptionParameter12,
				Mlp3OptionUnused13 -> unusedOptionParameter13,
				Mlp3OptionUnused21 -> unusedOptionParameter21,
				Mlp3OptionUnused22 -> unusedOptionParameter22,
				Mlp3OptionUnused23 -> unusedOptionParameter23,
    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
    			Mlp3OptionActivationAndScaling -> activationAndScaling,
    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
    			Mlp3OptionCostFunctionType -> costFunctionType,
    			DataTransformationOptionNormalizationType -> normalizationType,
    			UtilityOptionsParallelization -> "ParallelCalculation"
			];
			
		initialTrainingSetRmse = CalculateMlp3DataSetRmse[trainingSet, mlp3Info];
		If[isRegression,
			
			(* Regression*)
			If[Length[testSet] > 0,
				
				(* Regression WITH test set*)
				initialTestSetRmse = CalculateMlp3DataSetRmse[testSet, mlp3Info];
				If[isIntermediateOutput,
					Print["initialTrainingSetRmse = ", initialTrainingSetRmse];
					Print["initialTestSetRmse     = ", initialTestSetRmse]
				];
				relevance = 
					{
						{0.0, initialTrainingSetRmse},
						{0.0, initialTestSetRmse},
						{}, 
						mlp3Info
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
						mlp3Info
					}
			],
			
			(* Classification *)
			initialTrainingSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[trainingSet, mlp3Info];
			If[Length[testSet] > 0,
				
				(* Classification WITH test set*)
				initialTestSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[testSet, mlp3Info];
				If[isIntermediateOutput,
					Print["initialTrainingSetCorrectClassificationInPercent = ", initialTrainingSetCorrectClassificationInPercent];
					Print["initialTestSetCorrectClassificationInPercent     = ", initialTestSetCorrectClassificationInPercent]
				];
				relevance = 
					{
						{0.0, initialTrainingSetCorrectClassificationInPercent},
						{0.0, initialTestSetCorrectClassificationInPercent},
						{}, 
						mlp3Info
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
						mlp3Info
					}
			]
		];	
		
		AppendTo[mlp3InputComponentRelevanceList, relevance];
    
		ParallelNeeds[{"CIP`Mlp3`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[multipleMlp3s, optimizationMethod, initialWeights,
			initialNetworks, weightsValueLimit, minimizationPrecision, maximumNumberOfIterations, numberOfIterationsToImprove,
			unusedOptionParameter11, unusedOptionParameter12, unusedOptionParameter13, unusedOptionParameter21, unusedOptionParameter22,
			unusedOptionParameter23, randomValueInitialization, activationAndScaling, normalizationType];
			    
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
						
						mlp3Info = 
							FitMlp3[
								trainingSet,
								numberOfHiddenNeurons,
								Mlp3OptionMultipleMlp3s -> multipleMlp3s,
				    			Mlp3OptionOptimizationMethod -> optimizationMethod,
								Mlp3OptionInitialWeights -> initialWeights,
								Mlp3OptionInitialNetworks -> initialNetworks,
								Mlp3OptionWeightsValueLimit -> weightsValueLimit,
								Mlp3OptionMinimizationPrecision -> minimizationPrecision,
								Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
								Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
					 			Mlp3OptionReportIteration -> 0,
								Mlp3OptionUnused11 -> unusedOptionParameter11,
								Mlp3OptionUnused12 -> unusedOptionParameter12,
								Mlp3OptionUnused13 -> unusedOptionParameter13,
								Mlp3OptionUnused21 -> unusedOptionParameter21,
								Mlp3OptionUnused22 -> unusedOptionParameter22,
								Mlp3OptionUnused23 -> unusedOptionParameter23,
				    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
				    			Mlp3OptionActivationAndScaling -> activationAndScaling,
				    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
				    			Mlp3OptionCostFunctionType -> costFunctionType,
				    			DataTransformationOptionNormalizationType -> normalizationType
							];
								
						If[Length[testSet] > 0,
	            
							testSetRmse = CalculateMlp3DataSetRmse[testSet, mlp3Info];
							{testSetRmse, i},
	          
							trainingSetRmse = CalculateMlp3DataSetRmse[trainingSet, mlp3Info];
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
			mlp3Info = 
				FitMlp3[
					trainingSet,
					numberOfHiddenNeurons,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
	    			Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp3OptionReportIteration -> 0,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp3OptionActivationAndScaling -> activationAndScaling,
	    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp3OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionsParallelization -> "ParallelCalculation"
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMlp3DataSetRmse[trainingSet, mlp3Info];
					currentTestSetRmse = CalculateMlp3DataSetRmse[testSet, mlp3Info];
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
							mlp3Info
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMlp3DataSetRmse[trainingSet, mlp3Info];
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
							mlp3Info
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[trainingSet, mlp3Info];
					currentTestSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[testSet, mlp3Info];
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
							mlp3Info
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlp3CorrectClassificationInPercent[trainingSet, mlp3Info];
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
							mlp3Info
						}
				]
			];	

			AppendTo[mlp3InputComponentRelevanceList, relevance];
			removedInputComponentList = currentRemovedInputComponentList,
			
			{k, Length[numberOfExclusionsPerStepList]}
		];
		
		Return[mlp3InputComponentRelevanceList]
	];
	
GetMlp3RegressionResult[
	
	(* Returns mlp3 regression result according to named property list.

	   Returns :
	   Mlp3 regression result according to named property *)

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
    mlp3Info_,
	
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
	    
		pureFunction = Function[inputs, CalculateMlp3Outputs[inputs, mlp3Info]];
	    Return[
	    	CIP`Graphics`GetSingleRegressionResult[
		    	namedProperty, 
		    	dataSet, 
		    	pureFunction,
		    	GraphicsOptionNumberOfIntervals -> numberOfIntervals
			]
		]
	];

GetMlp3SeriesClassificationResult[

	(* Shows result of Mlp3 series classifications for training and test set.

	   Returns: 
	   mlp3SeriesClassificationResult: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in mlp3InfoList, classification result in percent for training set}
	   testPoint[[i]]: {index i in mlp3InfoList, classification result in percent for test set} *)


    (* {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...}
	   testSet has the same structure and restrictions as trainingSet 
	   NOTE: Training and test set MUST be in original units *)
    trainingAndTestSet_,

	(* {mlp3Info1, mlp3Info2, ...}
	   mlp3Info (see "Frequently used data structures") *)
    mlp3InfoList_
    
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
			pureFunction = Function[inputs, CalculateMlp3ClassNumbers[inputs, mlp3InfoList[[i]]]];
			correctClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[trainingSet, pureFunction];
			AppendTo[trainingPoints2D, {N[i], correctClassificationInPercent}];
			If[Length[testSet] > 0,
				correctClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[testSet, pureFunction];
				AppendTo[testPoints2D, {N[i], correctClassificationInPercent}]
			],
			
			{i, Length[mlp3InfoList]}
		];
		
		Return[{trainingPoints2D, testPoints2D}]
	];

GetMlp3SeriesRmse[

	(* Shows RMSE of Mlp3 series for training and test set.

	   Returns: 
	   mlp3SeriesRmse: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in mlp3InfoList, RMSE for training set}
	   testPoint[[i]]: {index i in mlp3InfoList, RMSE for test set} *)


    (* {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...}
	   testSet has the same structure and restrictions as trainingSet 
	   NOTE: Training and test set MUST be in original units *)
    trainingAndTestSet_,

	(* {mlp3Info1, mlp3Info2, ...}
	   mlp3Info (see "Frequently used data structures") *)
    mlp3InfoList_
    
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
			pureFunction = Function[inputs, CalculateMlp3Outputs[inputs, mlp3InfoList[[i]]]];
			rmse = Sqrt[CIP`Utility`GetMeanSquaredError[trainingSet, pureFunction]];
			AppendTo[trainingPoints2D, {N[i], rmse}];
			If[Length[testSet] > 0,
				rmse = Sqrt[CIP`Utility`GetMeanSquaredError[testSet, pureFunction]];
				AppendTo[testPoints2D, {N[i], rmse}]
			],
			
			{i, Length[mlp3InfoList]}
		];
		
		Return[{trainingPoints2D, testPoints2D}]
	];

GetMlp3Structure[

	(* Returns mlp3 structure for specified mlp3Info.

	   Returns:
	   {numberOfInputs, numberOfHiddenNeurons, numberOfOutputs} 
	   
	   numberOfHiddenNeurons: {numberOfHidden1Neurons, numberOfHidden2Neurons, numberOfHidden3Neurons} *)

    
  	(* See "Frequently used data structures" *)
    mlp3Info_
    
	] :=
  
	Module[
    
		{
			hidden1Weights,
			hidden2Weights,
			hidden3Weights,
			numberOfHidden1Neurons,
			numberOfHidden2Neurons,
			numberOfHidden3Neurons,
			numberOfInputs,
			numberOfOutputs,
			networks,
			outputWeights,
			weights
		},
    
    	networks = mlp3Info[[1]];
    	
		If[Length[networks] == 1,
	
			(* --------------------------------------------------------------------------------
			   One network
			   -------------------------------------------------------------------------------- *)		

	    	weights = networks[[1]];
	    	hidden1Weights = weights[[1]];
	    	hidden2Weights = weights[[2]];
	    	hidden3Weights = weights[[3]];
	    	outputWeights = weights[[4]];
	    	(* - 1: Subtract true unit *)
	    	numberOfInputs = Length[hidden1Weights[[1]]] - 1;
	    	numberOfHidden1Neurons = Length[hidden1Weights];
	    	numberOfHidden2Neurons = Length[hidden2Weights];
	    	numberOfHidden3Neurons = Length[hidden3Weights];
	    	numberOfOutputs = Length[outputWeights];
	    	Return[{numberOfInputs, {numberOfHidden1Neurons, numberOfHidden2Neurons, numberOfHidden3Neurons}, numberOfOutputs}],
			
			(* --------------------------------------------------------------------------------
			   Multiple networks (with ONE output value each)
			   -------------------------------------------------------------------------------- *)		

	    	weights = networks[[1]];
	    	hidden1Weights = weights[[1]];
	    	hidden2Weights = weights[[2]];
	    	hidden3Weights = weights[[3]];
	    	(* - 1: Subtract true unit *)
	    	numberOfInputs = Length[hidden1Weights[[1]]] - 1;
	    	numberOfHidden1Neurons = Length[hidden1Weights];
	    	numberOfHidden2Neurons = Length[hidden2Weights];
	    	numberOfHidden3Neurons = Length[hidden3Weights];
	    	numberOfOutputs = Length[networks];;
	    	Return[{numberOfInputs, {numberOfHidden1Neurons, numberOfHidden2Neurons, numberOfHidden3Neurons}, numberOfOutputs}]
		]
	];

GetMlp3TrainOptimization[

	(* Returns training set optimization result for mlp3 training.

	   Returns:
	   mlp3TrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlp3InfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlp3InfoList: List with mlp3Info
	   mlp3InfoList[[i]] refers to optimization step i *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,

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
			unusedOptionParameter22,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			unusedOptionParameter11,
			unusedOptionParameter12,
			maximumNumberOfIterations,
			minimizationPrecision,
			unusedOptionParameter13,
			multipleMlp3s,
			unusedOptionParameter23,
			numberOfIterationsToImprove,
			unusedOptionParameter21,
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
			mlp3Info,
			trainingSetRMSE,
			testSetRMSE,
			pureOutputFunction,
			trainingSetRmseList,
			testSetRmseList,
			trainingAndTestSetList,
			mlp3InfoList,
			selectionResult,
			blackList,
			parallelization,
			lambdaL2Regularization,
			costFunctionType
		},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp3 options *)
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
				Mlp3OptionActivationAndScaling -> activationAndScaling
			];
		trainingSetIndexList = clusterRepresentativesRelatedIndexLists[[1]];
		testSetIndexList = clusterRepresentativesRelatedIndexLists[[2]];
		indexLists = clusterRepresentativesRelatedIndexLists[[3]];

		trainingSetRmseList = {};
		testSetRmseList = {};
		trainingAndTestSetList = {};
		mlp3InfoList = {};
		blackList = {};
		Do[
			(* Fit training set and evaluate RMSE *)
			trainingSet = CIP`DataTransformation`GetDataSetPart[dataSet, trainingSetIndexList];
			testSet = CIP`DataTransformation`GetDataSetPart[dataSet, testSetIndexList];
			
			mlp3Info = 
				FitMlp3[
					trainingSet,
					numberOfHiddenNeurons,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
	    			Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
		 			Mlp3OptionReportIteration -> 0,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			Mlp3OptionActivationAndScaling -> activationAndScaling,
	    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
	    			Mlp3OptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionsParallelization -> parallelization
				];
				
			trainingSetRMSE = CalculateMlp3DataSetRmse[trainingSet, mlp3Info];
			testSetRMSE = CalculateMlp3DataSetRmse[testSet, mlp3Info];

			(* Set iteration results *)
			AppendTo[trainingSetRmseList, {N[i], trainingSetRMSE}];
			AppendTo[testSetRmseList, {N[i], testSetRMSE}];
			AppendTo[trainingAndTestSetList, {trainingSet, testSet}];
			AppendTo[mlp3InfoList, mlp3Info];
			
			(* Break if necessary *)
			If[i == numberOfTrainingSetOptimizationSteps,
				Break[]
			];

			(* Select new training and test set index lists *)
			pureOutputFunction = Function[input, CalculateMlp3Output[input, mlp3Info]];
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
				mlp3InfoList
			}
		]
	];

GetMlp3Weights[

	(* Returns weights of specified network of mlp3Info.

	   Returns:
	   weights: {hidden1Weights, hidden2Weights, outputWeights}
	   hidden1Weights: Weights from input to hidden units
	   hidden2Weights: Weights from hidden1 to hidden2 units
	   hidden3Weights: Weights from hidden2 to hidden3 units
	   outputWeights : Weights from hidden3 to output units *)

    
  	(* See "Frequently used data structures" *)
    mlp3Info_,
    
	(* Index of network in mlp3Info *)
    indexOfNetwork_?IntegerQ
    
	] :=
  
	Module[
    
		{
			networks
		},
    
    	networks = mlp3Info[[1]];
   		Return[networks[[indexOfNetwork]]]
	];

GetWeightsStartVariables[

	(* Returns weights start variables, see code. *)

    
	(* Number of hidden units *)
    numberOfInputs_?IntegerQ,

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,

	(* Number of hidden units *)
    numberOfOutputs_?IntegerQ,

    (* Variable for hidden1 weights *)
    wInputToHidden1_,

    (* Variable for hidden2 weights *)
    wHidden1ToHidden2_,

    (* Variable for hidden3 weights *)
    wHidden2ToHidden3_,
    
    (* Variable for output weights *)
    wHidden3ToOutput_,
    
    (* Hidden1 weights *)
    hidden1Weights_,

    (* Hidden2 weights *)
    hidden2Weights_,

    (* Hidden3 weights *)
    hidden3Weights_,
    
    (* Output weights *)
    outputWeights_

	] :=
  
	Module[
    
		{
			factor,
			hidden1WeigthsStartVariables,
			hidden2WeigthsStartVariables,
			hidden3WeigthsStartVariables,
			i,
			j,
			outputWeightsStartVariables
		},

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfInputs + 1];
		hidden1WeigthsStartVariables = {};
		Do[
	    	Do[
				AppendTo[hidden1WeigthsStartVariables, {Subscript[wInputToHidden1, j*factor + i], hidden1Weights[[j, i]]}],
	    		
	    		{i, numberOfInputs + 1}
	    	],
	    
	    	{j, numberOfHiddenNeurons[[1]]}	
		];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons[[1]] + 1];
		hidden2WeigthsStartVariables = {};
		Do[
	    	Do[
				AppendTo[hidden2WeigthsStartVariables, {Subscript[wHidden1ToHidden2, j * factor + i], hidden2Weights[[j, i]]}],
	    		
	    		{i, numberOfHiddenNeurons[[1]] + 1}
	    	],
	    
	    	{j, numberOfHiddenNeurons[[2]]}	
		];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons[[2]] + 1];
		hidden3WeigthsStartVariables = {};
		Do[
	    	Do[
				AppendTo[hidden3WeigthsStartVariables, {Subscript[wHidden2ToHidden3, j * factor + i], hidden3Weights[[j, i]]}],
	    		
	    		{i, numberOfHiddenNeurons[[2]] + 1}
	    	],
	    
	    	{j, numberOfHiddenNeurons[[3]]}	
		];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons[[3]] + 1];
		outputWeightsStartVariables = {};
		Do[	
			Do[
				AppendTo[outputWeightsStartVariables, {Subscript[wHidden3ToOutput, j * factor + i], outputWeights[[j, i]]}],
		    
		    	{i, numberOfHiddenNeurons[[3]] + 1}	
			],
				
			{j, numberOfOutputs}
		];
		
		Return[Join[hidden1WeigthsStartVariables, hidden2WeigthsStartVariables, hidden3WeigthsStartVariables, outputWeightsStartVariables]]
	];

GetWeightsVariablesWithoutTrueUnitBias[

	(* Returns weights variables without true unit bias weights, see code. *)

    
	(* Number of hidden units *)
    numberOfInputs_?IntegerQ,

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,

	(* Number of hidden units *)
    numberOfOutputs_?IntegerQ,

    (* Variable for hidden weights *)
    wInputToHidden1_,

    (* Variable for hidden2 weights *)
    wHidden1ToHidden2_,

    (* Variable for hidden3 weights *)
    wHidden2ToHidden3_,
    
    (* Variable for output weights *)
    wHidden3ToOutput_
    
	] :=
  
	Module[
    
		{
			factor,
			weigthVariables,
			i,
			j
		},

		(* NO true unit bias: Do NOT add 1 to numberOfInputs or numberOfHiddenNeurons *)

		weigthVariables = {};

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfInputs + 1];
		Do[
	    	Do[
				AppendTo[weigthVariables, Subscript[wInputToHidden1, j * factor + i]],
	    		
	    		{i, numberOfInputs}
	    	],
	    
	    	{j, numberOfHiddenNeurons[[1]]}	
		];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons[[1]] + 1];
		Do[
	    	Do[
				AppendTo[weigthVariables, Subscript[wHidden1ToHidden2, j * factor + i]],
	    		
	    		{i,  numberOfHiddenNeurons[[1]]}
	    	],
	    
	    	{j,  numberOfHiddenNeurons[[2]]}	
		];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons[[2]] + 1];
		Do[
	    	Do[
				AppendTo[weigthVariables, Subscript[wHidden2ToHidden3, j * factor + i]],
	    		
	    		{i,  numberOfHiddenNeurons[[2]]}
	    	],
	    
	    	{j,  numberOfHiddenNeurons[[3]]}	
		];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons[[3]] + 1];
		Do[	
			Do[
				AppendTo[weigthVariables, Subscript[wHidden3ToOutput, j * factor + i]],
		    
		    	{i, numberOfHiddenNeurons[[3]]}	
			],
				
			{j, numberOfOutputs}
		];
		
		Return[weigthVariables]
	];

GetWeightsVariables[

	(* Returns weights variables, see code. *)

    
	(* Number of hidden units *)
    numberOfInputs_?IntegerQ,

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,

	(* Number of hidden units *)
    numberOfOutputs_?IntegerQ,

    (* Variable for hidden weights *)
    wInputToHidden1_,

    (* Variable for hidden2 weights *)
    wHidden1ToHidden2_,

    (* Variable for hidden3 weights *)
    wHidden2ToHidden3_,
    
    (* Variable for output weights *)
    wHidden3ToOutput_

	] :=
  
	Module[
    
		{
			factor,
			hidden1WeightsVariables,
			hidden2WeightsVariables,
			hidden3WeightsVariables,
			i,
			j,
			outputWeightsVariables
		},

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfInputs + 1];
		hidden1WeightsVariables = 
			Table[
				Table[
					Subscript[wInputToHidden1, j * factor + i], 
					
					{i, numberOfInputs + 1}
				], 
				
				{j, numberOfHiddenNeurons[[1]]}
			];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons[[1]] + 1];
		hidden2WeightsVariables = 
			Table[
				Table[
					Subscript[wHidden1ToHidden2, j * factor + i], 
					
					{i, numberOfHiddenNeurons[[1]] + 1}
				], 
				
				{j, numberOfHiddenNeurons[[2]]}
			];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons[[2]] + 1];
		hidden3WeightsVariables = 
			Table[
				Table[
					Subscript[wHidden2ToHidden3, j * factor + i], 
					
					{i, numberOfHiddenNeurons[[2]] + 1}
				], 
				
				{j, numberOfHiddenNeurons[[3]]}
			];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons[[3]] + 1];
		outputWeightsVariables = 
			Table[
				Table[
					Subscript[wHidden3ToOutput, j * factor + i], 
					
					{i, numberOfHiddenNeurons[[3]] + 1}
				], 
				
				{j, numberOfOutputs}
			];
		
		Return[{hidden1WeightsVariables, hidden2WeightsVariables, hidden3WeightsVariables, outputWeightsVariables}]		
	];

ScanClassTrainingWithMlp3[

	(* Scans training and test set for different training fractions based on method FitMlp3, see code.
	
	   Returns:
	   mlp3ClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlp3Info1}, {trainingAndTestSet2, mlp3Info2}, ...}
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

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			unusedOptionParameter22,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			unusedOptionParameter11,
			unusedOptionParameter12,
			maximumNumberOfIterations,
			minimizationPrecision,
			unusedOptionParameter13,
			multipleMlp3s,
			unusedOptionParameter23,
			numberOfIterationsToImprove,
			unusedOptionParameter21,
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
		(* Mlp3 options *)
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
				ScanClassTrainingWithMlp3PC[
					classificationDataSet,
					numberOfHiddenNeurons,
					trainingFractionList,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
				    Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
					ClusterOptionMethod -> clusterMethod,
				    ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				    ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				    ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
			   	    Mlp3OptionActivationAndScaling -> activationAndScaling,
			   	    Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
			   	    Mlp3OptionCostFunctionType -> costFunctionType,
			   	    DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
				    UtilityOptionBlackListLength -> blackListLength
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				ScanClassTrainingWithMlp3SC[
					classificationDataSet,
					numberOfHiddenNeurons,
					trainingFractionList,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
				    Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
					ClusterOptionMethod -> clusterMethod,
				    ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				    ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				    ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
			   	    Mlp3OptionActivationAndScaling -> activationAndScaling,
			   	    Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
			   	    Mlp3OptionCostFunctionType -> costFunctionType,
			   	    DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
				    UtilityOptionBlackListLength -> blackListLength
				]
			]
		]
	];

ScanClassTrainingWithMlp3SC[

	(* Scans training and test set for different training fractions based on method FitMlp3, see code.
	
	   Returns:
	   mlp3ClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlp3Info1}, {trainingAndTestSet2, mlp3Info2}, ...}
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

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			unusedOptionParameter22,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			unusedOptionParameter11,
			unusedOptionParameter12,
			maximumNumberOfIterations,
			minimizationPrecision,
			unusedOptionParameter13,
			multipleMlp3s,
			unusedOptionParameter23,
			numberOfIterationsToImprove,
			unusedOptionParameter21,
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
			currentMlp3Info,
			pureFunction,
			trainingSetCorrectClassificationInPercent,
			testSetCorrectClassificationInPercent,
			mlp3TrainOptimization,
			mlp3InfoList,
			trainingAndTestSetList,
			bestIndex,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp3 options *)
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
				mlp3TrainOptimization = 
					GetMlp3TrainOptimization[
						classificationDataSet, 
						numberOfHiddenNeurons, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						Mlp3OptionMultipleMlp3s -> multipleMlp3s,
		    			Mlp3OptionOptimizationMethod -> optimizationMethod,
						Mlp3OptionInitialWeights -> initialWeights,
						Mlp3OptionInitialNetworks -> initialNetworks,
						Mlp3OptionWeightsValueLimit -> weightsValueLimit,
						Mlp3OptionMinimizationPrecision -> minimizationPrecision,
						Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
						Mlp3OptionUnused11 -> unusedOptionParameter11,
						Mlp3OptionUnused12 -> unusedOptionParameter12,
						Mlp3OptionUnused13 -> unusedOptionParameter13,
						Mlp3OptionUnused21 -> unusedOptionParameter21,
						Mlp3OptionUnused22 -> unusedOptionParameter22,
						Mlp3OptionUnused23 -> unusedOptionParameter23,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						Mlp3OptionActivationAndScaling -> activationAndScaling,
						Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
						Mlp3OptionCostFunctionType -> costFunctionType,
						DataTransformationOptionNormalizationType -> normalizationType,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestMlp3ClassOptimization[mlp3TrainOptimization];
				trainingAndTestSetList = mlp3TrainOptimization[[3]];
				mlp3InfoList = mlp3TrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlp3Info = mlp3InfoList[[bestIndex]],
				
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
				currentMlp3Info = 
					FitMlp3[
						currentTrainingSet,
						numberOfHiddenNeurons,
						Mlp3OptionMultipleMlp3s -> multipleMlp3s,
		    			Mlp3OptionOptimizationMethod -> optimizationMethod,
						Mlp3OptionInitialWeights -> initialWeights,
						Mlp3OptionInitialNetworks -> initialNetworks,
						Mlp3OptionWeightsValueLimit -> weightsValueLimit,
						Mlp3OptionMinimizationPrecision -> minimizationPrecision,
						Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
			 			Mlp3OptionReportIteration -> 0,
						Mlp3OptionUnused11 -> unusedOptionParameter11,
						Mlp3OptionUnused12 -> unusedOptionParameter12,
						Mlp3OptionUnused13 -> unusedOptionParameter13,
						Mlp3OptionUnused21 -> unusedOptionParameter21,
						Mlp3OptionUnused22 -> unusedOptionParameter22,
						Mlp3OptionUnused23 -> unusedOptionParameter23,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			Mlp3OptionActivationAndScaling -> activationAndScaling,
		    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
		    			Mlp3OptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					]
			];
			
			pureFunction = Function[inputs, CalculateMlp3ClassNumbers[inputs, currentMlp3Info]];
			trainingSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTrainingSet, pureFunction];
			testSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTestSet, pureFunction];
			AppendTo[trainingAndTestSetsInfo, {currentTrainingAndTestSet, currentMlp3Info}];
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

ScanClassTrainingWithMlp3PC[

	(* Scans training and test set for different training fractions based on method FitMlp3, see code.
	
	   Returns:
	   mlp3ClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlp3Info1}, {trainingAndTestSet2, mlp3Info2}, ...}
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

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			unusedOptionParameter22,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			unusedOptionParameter11,
			unusedOptionParameter12,
			maximumNumberOfIterations,
			minimizationPrecision,
			unusedOptionParameter13,
			multipleMlp3s,
			unusedOptionParameter23,
			numberOfIterationsToImprove,
			unusedOptionParameter21,
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
			currentMlp3Info,
			pureFunction,
			trainingSetCorrectClassificationInPercent,
			testSetCorrectClassificationInPercent,
			mlp3TrainOptimization,
			mlp3InfoList,
			trainingAndTestSetList,
			bestIndex,
			listOfTrainingAndTestSetsInfoAndScanReport,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp3 options *)
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* Training set optimization options *)
		numberOfTrainingSetOptimizationSteps = UtilityOptionOptimizationSteps/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];

		ParallelNeeds[{"CIP`Mlp3`", "CIP`Cluster`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[numberOfTrainingSetOptimizationSteps, multipleMlp3s, optimizationMethod, initialWeights,
						initialNetworks, weightsValueLimit, minimizationPrecision, maximumNumberOfIterations, numberOfIterationsToImprove, 
						unusedOptionParameter11, unusedOptionParameter12, unusedOptionParameter13, unusedOptionParameter21, unusedOptionParameter22, unusedOptionParameter23,
						clusterMethod, maximumNumberOfEpochs, scalarProductMinimumTreshold, maximumNumberOfTrialSteps, activationAndScaling, 
						normalizationType, randomValueInitialization, deviationCalculationMethod, blackListLength, lambdaL2Regularization, 
						costFunctionType];
		
		listOfTrainingAndTestSetsInfoAndScanReport = ParallelTable[
			If[numberOfTrainingSetOptimizationSteps > 0,
				
				(* ------------------------------------------------------------------------------------------------------ *)
				(* Training set optimization *)
				
				mlp3TrainOptimization = 
					GetMlp3TrainOptimization[
						classificationDataSet, 
						numberOfHiddenNeurons, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						Mlp3OptionMultipleMlp3s -> multipleMlp3s,
		    			Mlp3OptionOptimizationMethod -> optimizationMethod,
						Mlp3OptionInitialWeights -> initialWeights,
						Mlp3OptionInitialNetworks -> initialNetworks,
						Mlp3OptionWeightsValueLimit -> weightsValueLimit,
						Mlp3OptionMinimizationPrecision -> minimizationPrecision,
						Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
						Mlp3OptionUnused11 -> unusedOptionParameter11,
						Mlp3OptionUnused12 -> unusedOptionParameter12,
						Mlp3OptionUnused13 -> unusedOptionParameter13,
						Mlp3OptionUnused21 -> unusedOptionParameter21,
						Mlp3OptionUnused22 -> unusedOptionParameter22,
						Mlp3OptionUnused23 -> unusedOptionParameter23,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						Mlp3OptionActivationAndScaling -> activationAndScaling,
						Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
						Mlp3OptionCostFunctionType -> costFunctionType,
						DataTransformationOptionNormalizationType -> normalizationType,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestMlp3ClassOptimization[mlp3TrainOptimization];				
				trainingAndTestSetList = mlp3TrainOptimization[[3]];
				mlp3InfoList = mlp3TrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlp3Info = mlp3InfoList[[bestIndex]],
				
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
				
				currentMlp3Info = 
					FitMlp3[
						currentTrainingSet,
						numberOfHiddenNeurons,
						Mlp3OptionMultipleMlp3s -> multipleMlp3s,
		    			Mlp3OptionOptimizationMethod -> optimizationMethod,
						Mlp3OptionInitialWeights -> initialWeights,
						Mlp3OptionInitialNetworks -> initialNetworks,
						Mlp3OptionWeightsValueLimit -> weightsValueLimit,
						Mlp3OptionMinimizationPrecision -> minimizationPrecision,
						Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
			 			Mlp3OptionReportIteration -> 0,
						Mlp3OptionUnused11 -> unusedOptionParameter11,
						Mlp3OptionUnused12 -> unusedOptionParameter12,
						Mlp3OptionUnused13 -> unusedOptionParameter13,
						Mlp3OptionUnused21 -> unusedOptionParameter21,
						Mlp3OptionUnused22 -> unusedOptionParameter22,
						Mlp3OptionUnused23 -> unusedOptionParameter23,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			Mlp3OptionActivationAndScaling -> activationAndScaling,
		    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
		    			Mlp3OptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					];
			];
			
			pureFunction = Function[inputs, CalculateMlp3ClassNumbers[inputs, currentMlp3Info]];
			trainingSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTrainingSet, pureFunction];
			testSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTestSet, pureFunction];
			{
				{currentTrainingAndTestSet, currentMlp3Info},
				
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

ScanRegressTrainingWithMlp3[

	(* Scans training and test set for different training fractions based on method FitMlp3, see code.
	
	   Returns:
	   mlp3RegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlp3Info1}, {trainingAndTestSet2, mlp3Info2}, ...}
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

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			unusedOptionParameter22,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			unusedOptionParameter11,
			unusedOptionParameter12,
			maximumNumberOfIterations,
			minimizationPrecision,
			unusedOptionParameter13,
			multipleMlp3s,
			unusedOptionParameter23,
			numberOfIterationsToImprove,
			unusedOptionParameter21,
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
		(* Mlp3 options *)
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
				ScanRegressTrainingWithMlp3PC[
					dataSet,
					numberOfHiddenNeurons,
					trainingFractionList,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
				    Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
					ClusterOptionMethod -> clusterMethod,
				    ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				    ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				    ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
			   	    Mlp3OptionActivationAndScaling -> activationAndScaling,
			   	    Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
			   	    Mlp3OptionCostFunctionType -> costFunctionType,
			   	    DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
				    UtilityOptionBlackListLength -> blackListLength
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				ScanRegressTrainingWithMlp3SC[
					dataSet,
					numberOfHiddenNeurons,
					trainingFractionList,
					Mlp3OptionMultipleMlp3s -> multipleMlp3s,
				    Mlp3OptionOptimizationMethod -> optimizationMethod,
					Mlp3OptionInitialWeights -> initialWeights,
					Mlp3OptionInitialNetworks -> initialNetworks,
					Mlp3OptionWeightsValueLimit -> weightsValueLimit,
					Mlp3OptionMinimizationPrecision -> minimizationPrecision,
					Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
					Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
					Mlp3OptionUnused11 -> unusedOptionParameter11,
					Mlp3OptionUnused12 -> unusedOptionParameter12,
					Mlp3OptionUnused13 -> unusedOptionParameter13,
					Mlp3OptionUnused21 -> unusedOptionParameter21,
					Mlp3OptionUnused22 -> unusedOptionParameter22,
					Mlp3OptionUnused23 -> unusedOptionParameter23,
					ClusterOptionMethod -> clusterMethod,
				    ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				    ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				    ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
			   	    Mlp3OptionActivationAndScaling -> activationAndScaling,
			   	    Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
			   	    Mlp3OptionCostFunctionType -> costFunctionType,
			   	    DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
				    UtilityOptionBlackListLength -> blackListLength
				]
			]
		]
	];

ScanRegressTrainingWithMlp3SC[

	(* Scans training and test set for different training fractions based on method FitMlp3, see code.
	
	   Returns:
	   mlp3RegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlp3Info1}, {trainingAndTestSet2, mlp3Info2}, ...}
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

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			unusedOptionParameter22,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			unusedOptionParameter11,
			unusedOptionParameter12,
			maximumNumberOfIterations,
			minimizationPrecision,
			unusedOptionParameter13,
			multipleMlp3s,
			unusedOptionParameter23,
			numberOfIterationsToImprove,
			unusedOptionParameter21,
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
			currentMlp3Info,
			pureFunction,
			trainingSetRMSE,
			testSetRMSE,
			mlp3TrainOptimization,
			trainingAndTestSetList,
			mlp3InfoList,
			bestIndex,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp3 options *)
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
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
				mlp3TrainOptimization = 
					GetMlp3TrainOptimization[
						dataSet, 
						numberOfHiddenNeurons, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						Mlp3OptionMultipleMlp3s -> multipleMlp3s,
		    			Mlp3OptionOptimizationMethod -> optimizationMethod,
						Mlp3OptionInitialWeights -> initialWeights,
						Mlp3OptionInitialNetworks -> initialNetworks,
						Mlp3OptionWeightsValueLimit -> weightsValueLimit,
						Mlp3OptionMinimizationPrecision -> minimizationPrecision,
						Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
						Mlp3OptionUnused11 -> unusedOptionParameter11,
						Mlp3OptionUnused12 -> unusedOptionParameter12,
						Mlp3OptionUnused13 -> unusedOptionParameter13,
						Mlp3OptionUnused21 -> unusedOptionParameter21,
						Mlp3OptionUnused22 -> unusedOptionParameter22,
						Mlp3OptionUnused23 -> unusedOptionParameter23,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						Mlp3OptionActivationAndScaling -> activationAndScaling,
						Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
						Mlp3OptionCostFunctionType -> costFunctionType,
						DataTransformationOptionNormalizationType -> normalizationType,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestMlp3RegressOptimization[mlp3TrainOptimization];
				trainingAndTestSetList = mlp3TrainOptimization[[3]];
				mlp3InfoList = mlp3TrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlp3Info = mlp3InfoList[[bestIndex]],
				
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
				currentMlp3Info = 
					FitMlp3[
						currentTrainingSet,
						numberOfHiddenNeurons,
						Mlp3OptionMultipleMlp3s -> multipleMlp3s,
		    			Mlp3OptionOptimizationMethod -> optimizationMethod,
						Mlp3OptionInitialWeights -> initialWeights,
						Mlp3OptionInitialNetworks -> initialNetworks,
						Mlp3OptionWeightsValueLimit -> weightsValueLimit,
						Mlp3OptionMinimizationPrecision -> minimizationPrecision,
						Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
			 			Mlp3OptionReportIteration -> 0,
						Mlp3OptionUnused11 -> unusedOptionParameter11,
						Mlp3OptionUnused12 -> unusedOptionParameter12,
						Mlp3OptionUnused13 -> unusedOptionParameter13,
						Mlp3OptionUnused21 -> unusedOptionParameter21,
						Mlp3OptionUnused22 -> unusedOptionParameter22,
						Mlp3OptionUnused23 -> unusedOptionParameter23,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			Mlp3OptionActivationAndScaling -> activationAndScaling,
		    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
		    			Mlp3OptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					]
			];
			
			pureFunction = Function[inputs, CalculateMlp3Outputs[inputs, currentMlp3Info]];
			trainingSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTrainingSet, pureFunction]];
			testSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTestSet, pureFunction]];
			AppendTo[trainingAndTestSetsInfo, {currentTrainingAndTestSet, currentMlp3Info}];
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

ScanRegressTrainingWithMlp3PC[

	(* Scans training and test set for different training fractions based on method FitMlp3, see code.
	
	   Returns:
	   mlp3RegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlp3Info1}, {trainingAndTestSet2, mlp3Info2}, ...}
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

	(* Number of hidden neurons: {number of neurons in hidden1, ... in hidden2, ... in hidden3} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ] && Length[numberOfHiddenNeurons] == 3,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			unusedOptionParameter22,
			initialNetworks,
			initialWeights,
			weightsValueLimit,
			unusedOptionParameter11,
			unusedOptionParameter12,
			maximumNumberOfIterations,
			minimizationPrecision,
			unusedOptionParameter13,
			multipleMlp3s,
			unusedOptionParameter23,
			numberOfIterationsToImprove,
			unusedOptionParameter21,
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
			currentMlp3Info,
			pureFunction,
			trainingSetRMSE,
			testSetRMSE,
			mlp3TrainOptimization,
			trainingAndTestSetList,
			mlp3InfoList,
			bestIndex,
			listOfTrainingAndTestSetsInfoAndScanReport,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp3 options *)
		multipleMlp3s = Mlp3OptionMultipleMlp3s/.{opts}/.Options[Mlp3OptionsTraining];
	    optimizationMethod = Mlp3OptionOptimizationMethod/.{opts}/.Options[Mlp3OptionsTraining];
		initialWeights = Mlp3OptionInitialWeights/.{opts}/.Options[Mlp3OptionsOptimization];
		initialNetworks = Mlp3OptionInitialNetworks/.{opts}/.Options[Mlp3OptionsOptimization];
		weightsValueLimit = Mlp3OptionWeightsValueLimit/.{opts}/.Options[Mlp3OptionsOptimization];
		minimizationPrecision = Mlp3OptionMinimizationPrecision/.{opts}/.Options[Mlp3OptionsOptimization];
		maximumNumberOfIterations = Mlp3OptionMaximumIterations/.{opts}/.Options[Mlp3OptionsOptimization];
		numberOfIterationsToImprove = Mlp3OptionIterationsToImprove/.{opts}/.Options[Mlp3OptionsOptimization];
		unusedOptionParameter11 = Mlp3OptionUnused11/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter12 = Mlp3OptionUnused12/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter13 = Mlp3OptionUnused13/.{opts}/.Options[Mlp3OptionsUnused1];
		unusedOptionParameter21 = Mlp3OptionUnused21/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter22 = Mlp3OptionUnused22/.{opts}/.Options[Mlp3OptionsUnused2];
		unusedOptionParameter23 = Mlp3OptionUnused23/.{opts}/.Options[Mlp3OptionsUnused2];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = Mlp3OptionActivationAndScaling/.{opts}/.Options[Mlp3OptionsTraining];
	    lambdaL2Regularization = Mlp3OptionLambdaL2Regularization/.{opts}/.Options[Mlp3OptionsTraining];
	    costFunctionType = Mlp3OptionCostFunctionType/.{opts}/.Options[Mlp3OptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* Training set optimization options *)
		numberOfTrainingSetOptimizationSteps = UtilityOptionOptimizationSteps/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		
		ParallelNeeds[{"CIP`Mlp3`", "CIP`Cluster`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[numberOfTrainingSetOptimizationSteps, multipleMlp3s, optimizationMethod, initialWeights,
						initialNetworks, weightsValueLimit, minimizationPrecision, maximumNumberOfIterations, numberOfIterationsToImprove, 
						unusedOptionParameter11, unusedOptionParameter12, unusedOptionParameter13, unusedOptionParameter21, unusedOptionParameter22, unusedOptionParameter23,
						clusterMethod, maximumNumberOfEpochs, scalarProductMinimumTreshold, maximumNumberOfTrialSteps, activationAndScaling, 
						normalizationType, randomValueInitialization, deviationCalculationMethod, blackListLength, lambdaL2Regularization];
			
		listOfTrainingAndTestSetsInfoAndScanReport = ParallelTable[
			If[numberOfTrainingSetOptimizationSteps > 0,
				
				(* ------------------------------------------------------------------------------------------------------ *)
				(* Training set optimization *)
				
				mlp3TrainOptimization = 
					GetMlp3TrainOptimization[
						dataSet, 
						numberOfHiddenNeurons, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						Mlp3OptionMultipleMlp3s -> multipleMlp3s,
		    			Mlp3OptionOptimizationMethod -> optimizationMethod,
						Mlp3OptionInitialWeights -> initialWeights,
						Mlp3OptionInitialNetworks -> initialNetworks,
						Mlp3OptionWeightsValueLimit -> weightsValueLimit,
						Mlp3OptionMinimizationPrecision -> minimizationPrecision,
						Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
						Mlp3OptionUnused11 -> unusedOptionParameter11,
						Mlp3OptionUnused12 -> unusedOptionParameter12,
						Mlp3OptionUnused13 -> unusedOptionParameter13,
						Mlp3OptionUnused21 -> unusedOptionParameter21,
						Mlp3OptionUnused22 -> unusedOptionParameter22,
						Mlp3OptionUnused23 -> unusedOptionParameter23,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						Mlp3OptionActivationAndScaling -> activationAndScaling,
						Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
						Mlp3OptionCostFunctionType -> costFunctionType,
						DataTransformationOptionNormalizationType -> normalizationType,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
					
				bestIndex = GetBestMlp3RegressOptimization[mlp3TrainOptimization];
				trainingAndTestSetList = mlp3TrainOptimization[[3]];
				mlp3InfoList = mlp3TrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlp3Info = mlp3InfoList[[bestIndex]],
				
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
				
				currentMlp3Info = 
					FitMlp3[
						currentTrainingSet,
						numberOfHiddenNeurons,
						Mlp3OptionMultipleMlp3s -> multipleMlp3s,
	   		 			Mlp3OptionOptimizationMethod -> optimizationMethod,
						Mlp3OptionInitialWeights -> initialWeights,
						Mlp3OptionInitialNetworks -> initialNetworks,
						Mlp3OptionWeightsValueLimit -> weightsValueLimit,
						Mlp3OptionMinimizationPrecision -> minimizationPrecision,
						Mlp3OptionMaximumIterations -> maximumNumberOfIterations,
						Mlp3OptionIterationsToImprove -> numberOfIterationsToImprove,
			 			Mlp3OptionReportIteration -> 0,
						Mlp3OptionUnused11 -> unusedOptionParameter11,
						Mlp3OptionUnused12 -> unusedOptionParameter12,
						Mlp3OptionUnused13 -> unusedOptionParameter13,
						Mlp3OptionUnused21 -> unusedOptionParameter21,
						Mlp3OptionUnused22 -> unusedOptionParameter22,
						Mlp3OptionUnused23 -> unusedOptionParameter23,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			Mlp3OptionActivationAndScaling -> activationAndScaling,
		    			Mlp3OptionLambdaL2Regularization -> lambdaL2Regularization,
		    			Mlp3OptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					];
			];
			
			pureFunction = Function[inputs, CalculateMlp3Outputs[inputs, currentMlp3Info]];
			trainingSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTrainingSet, pureFunction]];
			testSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTestSet, pureFunction]];
			{
				{currentTrainingAndTestSet, currentMlp3Info},
				
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

ShowMlp3Output2D[

	(* Shows 2D mlp3 output.

	   Returns: Nothing *)


    (* Index of input neuron that receives argumentValue *)
    indexOfInput_?IntegerQ,

    (* Index of output neuron that returns function value *)
    indexOfFunctionValueOutput_?IntegerQ,
    
    (* Mlp3 input in original units: 
       inputsInOriginalUnits = {inputValue1, inputValue2, ...} 
       Value of input neuron with specified index (indexOfInput) is replaced by argumentValue *)
    input_/;VectorQ[input, NumberQ],
    
    (* Arguments to be displayed as points:
       arguments: {argumentValue1, argumentValue2, ...} *)
    arguments_/;VectorQ[arguments, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlp3Info_
      
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
						CalculateMlp3Value2D[arguments[[i]], indexOfInput, indexOfFunctionValueOutput, input, mlp3Info]
					},
						
					{i, Length[arguments]}
				],
				
			points = {}
		];
		
		dataSetScaleInfo = mlp3Info[[2]];
		inputsMinMaxList = dataSetScaleInfo[[1, 1]];
		xMin = inputsMinMaxList[[indexOfInput, 1]];
		xMax = inputsMinMaxList[[indexOfInput, 2]];
		
		labels = 
			{
				StringJoin["Argument Value of Input ", ToString[indexOfInput]],
				StringJoin["Value of Output ", ToString[indexOfFunctionValueOutput]],
				"Mlp3 Output"
			};
		Print[
			CIP`Graphics`PlotPoints2DAboveFunction[
				points, 
				Function[x, CalculateMlp3Value2D[x, indexOfInput, indexOfFunctionValueOutput, input, mlp3Info]], 
				labels,
				GraphicsOptionArgumentRange2D -> {xMin, xMax}
			]
		]
	];

ShowMlp3Output3D[

	(* Shows 3D mlp3 output.

	   Returns: Graphics3D *)


    (* Index of input neuron that receives argumentValue1 *)
    indexOfInput1_?IntegerQ,

    (* Index of input neuron that receives argumentValue2 *)
    indexOfInput2_?IntegerQ,

    (* Index of output neuron that returns function value *)
    indexOfFunctionValueOutput_?IntegerQ,
    
    (* Mlp3 input in original units: 
       inputsInOriginalUnits = {inputValue1, inputValue2, ...} 
       Value of input neuron with specified index (indexOfInput) is replaced by argumentValue *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlp3Info_,
    
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
	
		dataSetScaleInfo = mlp3Info[[2]];
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
				Function[{x1, x2}, CalculateMlp3Value3D[x1, x2, indexOfInput1, indexOfInput2, indexOfFunctionValueOutput, input, mlp3Info]], 
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

ShowMlp3ClassificationResult[

	(* Shows result of mlp3 classification for training and test set according to named property list.

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
    mlp3Info_,
    
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
		ShowMlp3SingleClassification[
			namedPropertyList,
			trainingSet, 
			mlp3Info,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionMinMaxIndex -> minMaxIndex
		];
		
		(* Analyze test set *)
		If[Length[testSet] > 0,
			Print["Test Set:"];
			ShowMlp3SingleClassification[
				namedPropertyList,
				testSet, 
				mlp3Info,
				GraphicsOptionImageSize -> imageSize,
				GraphicsOptionMinMaxIndex -> minMaxIndex
			];
		]
	];

ShowMlp3SingleClassification[

	(* Shows result of mlp3 classification for data set according to named property list.

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
    mlp3Info_,
    
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

   		pureFunction = Function[inputs, CalculateMlp3ClassNumbers[inputs, mlp3Info]];
		CIP`Graphics`ShowClassificationResult[
			namedPropertyList,
			classificationDataSet, 
			pureFunction,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionMinMaxIndex -> minMaxIndex
		]
	];

ShowMlp3ClassificationScan[

	(* Shows result of Mlp3 based classification scan of clustered training sets.

	   Returns: Nothing *)


	(* mlp3ClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlp3Info1}, {trainingAndTestSet2, mlp3Info2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, classification result in percent for training set}, {trainingFraction, classification result in percent for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)
	mlp3ClassificationScan_,
	
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
			mlp3ClassificationScan,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMlp3InputRelevanceClass[

	(* Shows mlp3InputComponentRelevanceListForClassification.

	   Returns: Nothing *)


	(* mlp3InputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlp3Info}
	   trainingSetResult: {numberOfRemovedInputs, (best) correct classification in percent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best correct classification in percent of test set} *)
	mlp3InputComponentRelevanceListForClassification_,
	
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
			mlp3InputComponentRelevanceListForClassification,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMlp3InputRelevanceRegress[

	(* Shows mlp3InputComponentRelevanceListForRegression.

	   Returns: Nothing *)


	(* mlp3InputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlp3Info}
	   trainingSetResult: {numberOfRemovedInputs, (best) RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)
	mlp3InputComponentRelevanceListForRegression_,
	
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
			mlp3InputComponentRelevanceListForRegression,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];
	
ShowMlp3RegressionResult[

	(* Shows result of mlp3 regression for training and test set according to named property list.
	
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
    mlp3Info_,
    
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
		ShowMlp3SingleRegression[
			namedPropertyList,
			trainingSet, 
			mlp3Info,
			GraphicsOptionPointSize -> pointSize,
			GraphicsOptionPointColor -> pointColor
		];
		
		(* Analyze test set *)
		If[Length[testSet] > 0,
			Print["Test Set:"];
			ShowMlp3SingleRegression[
				namedPropertyList,
				testSet, 
				mlp3Info,
				GraphicsOptionPointSize -> pointSize,
				GraphicsOptionPointColor -> pointColor
			]
		]
	];

ShowMlp3SingleRegression[
    
	(* Shows result of mlp3 regression for data set according to named property list.
	
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
    mlp3Info_,
    
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

		pureFunction = Function[inputs, CalculateMlp3Outputs[inputs, mlp3Info]];
		CIP`Graphics`ShowRegressionResult[
			namedPropertyList,
			dataSet, 
			pureFunction,
			GraphicsOptionPointSize -> pointSize,
			GraphicsOptionPointColor -> pointColor
		]
	];

ShowMlp3RegressionScan[

	(* Shows result of Mlp3 based regression scan of clustered training sets.

	   Returns: Nothing *)


	(* mlp3RegressionScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlp3Info1}, {trainingAndTestSet2, mlp3Info2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, RMSE for training set}, {trainingFraction, RMSE for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)
	mlp3RegressionScan_,
	
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
			mlp3RegressionScan,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMlp3SeriesClassificationResult[

	(* Shows result of Mlp3 series classifications for training and test set.

	   Returns: Nothing *)


	(* mlp3SeriesClassificationResult: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in mlp3InfoList, classification result in percent for training set}
	   testPoint[[i]]: {index i in mlp3InfoList, classification result in percent for test set} *)
	mlp3SeriesClassificationResult_,
    
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

		trainingPoints2D = mlp3SeriesClassificationResult[[1]];
		testPoints2D = mlp3SeriesClassificationResult[[2]];
		
		If[Length[testPoints2D] > 0,

			(* Training and test set *)
			labels = {"mlp3Info index", "Correct classifications [%]", "Training (green), Test (red)"};
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
			Print["Best test set classification with mlp3Info index = ", bestIndexList],
		
			(* Training set only *)
			labels = {"mlp3Info index", "Correct classifications [%]", "Training (green)"};
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
			Print["Best training set classification with mlp3Info index = ", bestIndexList]			
		]
	];

ShowMlp3SeriesRmse[

	(* Shows RMSE of Mlp3 series for training and test set.

	   Returns: Nothing *)


	(* mlp3SeriesRmse: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in mlp3InfoList, RMSE for training set}
	   testPoint[[i]]: {index i in mlp3InfoList, RMSE for test set} *)
	mlp3SeriesRmse_,
    
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

		trainingPoints2D = mlp3SeriesRmse[[1]];
		testPoints2D = mlp3SeriesRmse[[2]];

		If[Length[testPoints2D] > 0,
			
			(* Training and test set *)
			labels = {"mlp3Info index", "RMSE", "Training (green), Test (red)"};
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
			Print["Best test set regression with mlp3Info index = ", bestIndexList],

			(* Training set only *)
			labels = {"mlp3Info index", "RMSE", "Training (green)"};
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
			Print["Best training set regression with mlp3Info index = ", bestIndexList]			
		]
	];

ShowMlp3Training[

	(* Shows training of mlp3.

	   Returns: Nothing *)


  	(* See "Frequently used data structures" *)
    mlp3Info_
    
	] :=
  
	Module[
    
		{
			i,
			labels,
			mlp3TrainingResults,
			trainingSetMeanSquaredErrorList,
			testSetMeanSquaredErrorList
		},

		mlp3TrainingResults = mlp3Info[[3]];
		Do[

			If[Length[mlp3TrainingResults] == 1,
				
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
			
			trainingSetMeanSquaredErrorList = mlp3TrainingResults[[i, 1]];
			testSetMeanSquaredErrorList = mlp3TrainingResults[[i, 2]];
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
		
			{i, Length[mlp3TrainingResults]}
		]
	];

ShowMlp3TrainOptimization[

	(* Shows training set optimization result of mlp3.

	   Returns: Nothing *)


	(* mlp3TrainOptimization = {trainingSetRmseList, testSetRmseList, not interesting, not interesting}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set} *)
	mlp3TrainOptimization_,
    
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
			mlp3TrainOptimization, 
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
