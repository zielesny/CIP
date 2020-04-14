(*
-----------------------------------------------------------------------
Computational Intelligence Packages (CIP): Package MLP
(Multi-Layer Feed-Forward Neural Network or Perceptron with an 
arbitrary number of hidden-neuron layers)
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
mlpInfo: {networks, dataSetScaleInfo, mlpTrainingResults, normalizationInfo, activationAndScaling, optimizationMethod} 

	networks: {weights1, weights2, ...}
	weights: {hidden1Weights, hidden2Weights, hidden3Weights, ..., outputWeights}
	hidden1Weights: Weights from input to hidden1 units
	hidden2Weights: Weights from hidden1 to hidden2 units
	hidden3Weights: Weights from hidden2 to hidden3 units
	...
	outputWeights : Weights from hidden3 to output units
	dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs}, see GetDataSetScaleInfo
	mlpTrainingResults: {singleTrainingResult1, singleTrainingResult2, ...}
	singleTrainingResult[[i]] corresponds to weights[[i]]
	singleTrainingResult: {trainingMeanSquaredErrorList, testMeanSquaredErrorList}
	trainingMeanSquaredErrorList: {reportPair1, reportPair2, ...}
	reportPair: {reportIteration, mean squared error of report iteration}
	testMeanSquaredErrorList: Same structure as trainingMeanSquaredErrorList
	normalizationInfo: {normalizationType, meanAndStandardDeviationList}, see CIP`DataTransformation`GetDataMatrixNormalizationInfo
	activationAndScaling: See option MlpOptionActivationAndScaling
	optimizationMethod: Optimization method
-----------------------------------------------------------------------
*)

(* ::Section:: *)
(* Package and dependencies *)

BeginPackage["CIP`MLP`", {"CIP`Utility`", "CIP`Graphics`", "CIP`DataTransformation`", "CIP`Cluster`"}]

(* ::Section:: *)
(* Off settings *)

Off[General::"spell1"]
Off[General::shdw]
Off[FindMinimum::cvmit]
Off[General::compat]

(* ::Section:: *)
(* Options *)

Options[MlpOptionsTraining] = 
{
	(* True: Multiple mlps may be used (one mlp for every single output component), False: One mlp is used only *)
    MlpOptionMultipleMlps -> True,
	
	(* Optimization method: "FindMinimum", "NMinimize" *)
    MlpOptionOptimizationMethod -> "FindMinimum",
    
    (* Test set *)
    MlpOptionTestSet -> {},
    
	(* activationAndScaling: Definition of activation function and corresponding input/output scaling of data
       activationAndScaling: {activation, inputOutputTargetIntervals}
       e.g.
       {{"Sigmoid", "Sigmoid", "Sigmoid", "Sigmoid"}, {{-0.9, 0.9}, {0.1, 0.9}}}
       {{"Tanh", "Tanh", "Tanh", "Sigmoid"}, {{-0.9, 0.9}, {0.1, 0.9}}}
       {{"Tanh", "Tanh", "Tanh", "Tanh"}, {{-0.9, 0.9}, {-0.9, 0.9}}}
	   
	   activation: 
	   {<Activation function for hidden1 neurons>, <... for hidden2 neurons>, <... for hidden3 neurons>, ..., <... for output neurons>}
	   Activation function for hidden1/2/3/output neurons: "Sigmoid", "Tanh"
	   
	   inputOutputTargetIntervals: {inputTargetInterval, outputTargetInterval}
	   inputTargetInterval/outputTargetInterval contains the desired minimum and maximum value for each column of inputs and outputs
	   inputTargetInterval/outputTargetInterval: {targetMin, targetMax} 
	   targetMin: Minimum value for each column 
	   targetMax: Maximum value for each column *)
	MlpOptionActivationAndScaling -> {},
	
	(* Lambda parameter for L2 regularization: A value of 0.0 means NO L2 regularization *)
	MlpOptionLambdaL2Regularization -> 0.0,
	
	(* Cost function type: "SquaredError", "Cross-Entropy" *)
	MlpOptionCostFunctionType -> "SquaredError"
}

Options[MlpOptionsOptimization] = 
{
	(* Initial weights to be improved (may be empty list)
	   initialWeights: {hidden1Weights, hidden2Weights, hidden3Weights, ..., outputWeights}
	   hidden1Weights: Weights from input to hidden1 units
	   hidden2Weights: Weights from hidden1 to hidden2 units
	   hidden3Weights: Weights from hidden2 to hidden3 units
	   ...
	   outputWeights: Weights from hidden3 to output units *)
	MlpOptionInitialWeights -> {},

	(* Initial networks for multiple mlp2s training to be improved (may be empty list)
	   networks: {weights1, weights2, ...}
	   weights: {hidden1Weights, hidden2Weights, hidden3Weights, ..., outputWeights}
	   hidden1Weights: Weights from input to hidden1 units
	   hidden2Weights: Weights from hidden1 to hidden2 units
	   hidden3Weights: Weights from hidden2 to hidden3 units
	   ...
	   outputWeights: Weights from hidden3 to output units *)
	MlpOptionInitialNetworks -> {},
	
    (* Weights for NMinimize will be in interval 
       -Mlp2OptionWeightsValueLimit <= weight value <= +Mlp2OptionWeightsValueLimit*)
	MlpOptionWeightsValueLimit -> 1000.0,
	
    (* Number of digits for AccuracyGoal and PrecisionGoal (MUST be smaller than MachinePrecision) *)
    MlpOptionMinimizationPrecision -> 5,
    
    (* Maximum number of minimization steps *)
    MlpOptionMaximumIterations -> 10000,

    (* Number of iterations to improve *)
    MlpOptionIterationsToImprove -> 1000,
    
    (* The meanSquaredErrorLists (training protocol) will be filled every reportIteration steps.
       reportIteration <= 0 means no internal reports during training/minimization procedure. *)
    MlpOptionReportIteration -> 0
}

Options[MlpOptionsUnused1] = 
{
    (* Unused *)
    MlpOptionUnused11 -> 0.0,
    
    (* Unused *)
    MlpOptionUnused12 -> 0.0,
    
    (* Unused *)
    MlpOptionUnused13 -> 0.0
}

Options[MlpOptionsUnused2] =
{
    (* Unused *)
    MlpOptionUnused21 -> 0.0,
    
    (* Unused *)
    MlpOptionUnused22 -> 0.0,
    
    (* Unused *)
    MlpOptionUnused23 -> 0.0
}

(* ::Section:: *)
(* Declarations *)

BumpFunction::usage = 
	"BumpFunction[x, interval]"

BumpSum::usage = 
	"BumpSum[x, intervals]"

CalculateMlpValue2D::usage = 
	"CalculateMlpValue2D[argumentValue, indexOfInput, indexOfFunctionValueOutput, input, mlpInfo]"

CalculateMlpValue3D::usage = 
	"CalculateMlpValue3D[argumentValue1, argumentValue2, indexOfInput1, indexOfInput2, indexOfFunctionValueOutput, input, mlpInfo]"

CalculateMlpClassNumber::usage = 
	"CalculateMlpClassNumber[input, mlpInfo]"

CalculateMlpClassNumbers::usage = 
	"CalculateMlpClassNumbers[inputs, mlpInfo]"

CalculateMlpDataSetRmse::usage = 
	"CalculateMlpDataSetRmse[dataSet, mlpInfo]"

CalculateMlpOutput::usage = 
	"CalculateMlpOutput[input, mlpInfo]"

CalculateMlpOutputs::usage = 
	"CalculateMlpOutputs[inputs, mlpInfo]"

FitMlp::usage = 
	"FitMlp[dataSet, numberOfHiddenNeurons, options]"

FitMlpSeries::usage = 
	"FitMlpSeries[dataSet, numberOfHiddenNeuronsList, options]"

GetBestMlpClassOptimization::usage = 
	"GetBestMlpClassOptimization[mlpTrainOptimization, options]"

GetBestMlpRegressOptimization::usage = 
	"GetBestMlpRegressOptimization[mlpTrainOptimization, options]"

GetNumberOfHiddenNeurons::usage = 
	"GetNumberOfHiddenNeurons[mlpInfo]"

GetMlpInputInclusionClass::usage = 
	"GetMlpInputInclusionClass[trainingAndTestSet, numberOfHiddenNeurons, options]"

GetMlpInputInclusionRegress::usage = 
	"GetMlpInputInclusionRegress[trainingAndTestSet, numberOfHiddenNeurons, options]"

GetMlpInputRelevanceClass::usage = 
	"GetMlpInputRelevanceClass[trainingAndTestSet, numberOfHiddenNeurons, options]"

GetMlpClassRelevantComponents::usage = 
    "GetMlpClassRelevantComponents[mlpInputComponentRelevanceListForClassification, numberOfComponents]"

GetMlpInputRelevanceRegress::usage = 
	"GetMlpInputRelevanceRegress[trainingAndTestSet, numberOfHiddenNeurons, options]"

GetMlpRegressRelevantComponents::usage = 
    "GetMlpRegressRelevantComponents[mlpInputComponentRelevanceListForRegression, numberOfComponents]"

GetMlpRegressionResult::usage = 
	"GetMlpRegressionResult[namedProperty, dataSet, mlpInfo, options]"

GetMlpSeriesClassificationResult::usage = 
	"GetMlpSeriesClassificationResult[trainingAndTestSet, mlpInfoList]"

GetMlpSeriesRmse::usage = 
	"GetMlpSeriesRmse[trainingAndTestSet, mlpInfoList]"

GetMlpStructure::usage = 
	"GetMlpStructure[mlpInfo]"

GetMlpTrainOptimization::usage = 
	"GetMlpTrainOptimization[dataSet, numberOfHiddenNeurons, trainingFraction, numberOfTrainingSetOptimizationSteps, options]"

GetMlpWeights::usage = 
	"GetMlpWeights[mlpInfo, indexOfNetwork]"

ScanClassTrainingWithMlp::usage = 
	"ScanClassTrainingWithMlp[dataSet, numberOfHiddenNeurons, trainingFractionList, options]"

ScanRegressTrainingWithMlp::usage = 
	"ScanRegressTrainingWithMlp[dataSet, numberOfHiddenNeurons, trainingFractionList, options]"

ShowMlpOutput2D::usage = 
	"ShowMlpOutput2D[indexOfInput, indexOfFunctionValueOutput, input, arguments, mlpInfo]"

ShowMlpOutput3D::usage = 
	"ShowMlpOutput3D[indexOfInput1, indexOfInput2, indexOfFunctionValueOutput, input, mlpInfo, options]"

ShowMlpClassificationResult::usage = 
	"ShowMlpClassificationResult[namedPropertyList, trainingAndTestSet, mlpInfo]"

ShowMlpSingleClassification::usage = 
	"ShowMlpSingleClassification[namedPropertyList, classificationDataSet, mlpInfo]"

ShowMlpClassificationScan::usage = 
	"ShowMlpClassificationScan[mlpClassificationScan, options]"

ShowMlpInputRelevanceClass::usage = 
	"ShowMlpInputRelevanceClass[mlpInputComponentRelevanceListForClassification, options]"
	
ShowMlpInputRelevanceRegress::usage = 
	"ShowMlpInputRelevanceRegress[mlpInputComponentRelevanceListForRegression, options]"

ShowMlpRegressionResult::usage = 
	"ShowMlpRegressionResult[namedPropertyList, trainingAndTestSet, mlpInfo]"

ShowMlpSingleRegression::usage = 
	"ShowMlpSingleRegression[namedPropertyList, dataSet, mlpInfo]"

ShowMlpRegressionScan::usage = 
	"ShowMlpRegressionScan[mlpRegressionScan, options]"

ShowMlpSeriesClassificationResult::usage = 
	"ShowMlpSeriesClassificationResult[mlpSeriesClassificationResult, options]"

ShowMlpSeriesRmse::usage = 
	"ShowMlpSeriesRmse[mlpSeriesRmse, options]"

ShowMlpTraining::usage = 
	"ShowMlpTraining[mlpInfo]"

ShowMlpTrainOptimization::usage = 
	"ShowMlpTrainOptimization[mlpTrainOptimization, options]"

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

CalculateMlpValue2D[

	(* Calculates 2D output for specified argument and input for specified mlp.
	   This special method assumes an input and an output with one component only.

	   Returns:
	   Value of specified output neuron for argument *)

    (* Argument value for neuron with index indexOfInput *)
    argumentValue_?NumberQ,
    
  	(* See "Frequently used data structures" *)
    mlpInfo_
    
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
			CalculateMlpValue2D[argumentValue, indexOfInput, indexOfFunctionValueOutput, input, mlpInfo]
		]
	];

CalculateMlpValue2D[

	(* Calculates 2D output for specified argument and input for specified mlp.

	   Returns:
	   Value of specified output neuron for argument and input *)

    (* Argument value for neuron with index indexOfInput *)
    argumentValue_?NumberQ,
    
    (* Index of input neuron that receives argumentValue *)
    indexOfInput_?IntegerQ,

    (* Index of output neuron that returns function value *)
    indexOfFunctionValueOutput_?IntegerQ,
    
    (* Mlp input in original units: 
       inputsInOriginalUnits = {inputValue1, inputValue2, ...} 
       Value of input neuron with specified index (indexOfInput) is replaced by argumentValue *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlpInfo_
    
	] :=
  
	Module[
    
		{
			currentInput,
			output
		},
		
		currentInput = ReplacePart[input, {indexOfInput -> argumentValue}];
		output = CalculateMlpOutput[currentInput, mlpInfo];
		Return[output[[indexOfFunctionValueOutput]]];
	];

CalculateMlpValue3D[

	(* Calculates 3D output for specified arguments for specified mlp. 
	   This specific methods assumes a mlp with 2 input neurons and 1 output neuron.

	   Returns:
	   Value of the single output neuron for arguments *)


    (* Argument value for neuron with index indexOfInput1 *)
    argumentValue1_?NumberQ,
    
    (* Argument value for neuron with index indexOfInput2 *)
    argumentValue2_?NumberQ,
    
  	(* See "Frequently used data structures" *)
    mlpInfo_
    
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
			CalculateMlpValue3D[argumentValue1, argumentValue2, indexOfInput1, indexOfInput2, indexOfOutput, input, mlpInfo]
		]
	];

CalculateMlpValue3D[

	(* Calculates 3D output for specified arguments and input for specified mlp.

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
    
    (* Mlp input in original units: 
       inputsInOriginalUnits = {inputValue1, inputValue2, ...} 
       Value of input neurons with specified indices (indexOfInput1, indexOfInput2) are replaced by argument values *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlpInfo_
    
	] :=
  
	Module[
    
		{
			currentInput,
			output
		},
		
		currentInput = ReplacePart[input, {indexOfInput1 -> argumentValue1, indexOfInput2 -> argumentValue2}];
		output = CalculateMlpOutput[currentInput, mlpInfo];
		Return[output[[indexOfFunctionValueOutput]]];
	];

CalculateMlpClassNumber[

	(* Returns class number for specified input for classification mlp with specified weights.

	   Returns:
	   Class number of input *)

    
    (* input in original units: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    input_/;VectorQ[input, NumberQ],
        
  	(* See "Frequently used data structures" *)
    mlpInfo_
    
	] :=
  
	Module[
    
		{
			combinedOutputs,
			i,
			networks,
			scaledInputs,
			outputs,
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			weights
		},
    
    	networks = mlpInfo[[1]];
    	dataSetScaleInfo = mlpInfo[[2]];
    	normalizationInfo = mlpInfo[[4]];
    	activationAndScaling = mlpInfo[[5]];

		If[Length[networks] == 1,
	
			(* --------------------------------------------------------------------------------
			   One network
			   -------------------------------------------------------------------------------- *)		

			weights = networks[[1]];
			(* Transform original input *)
			scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo];
			outputs = GetInternalMlpOutputs[scaledInputs, weights, activationAndScaling];
			Return[CIP`Utility`GetPositionOfMaximumValue[outputs[[1]]]],
			
			(* --------------------------------------------------------------------------------
			   Multiple networks (with ONE output value each)
			   -------------------------------------------------------------------------------- *)		

			combinedOutputs =
				Table[
					weights = networks[[i]];
					(* Transform original input *)
					scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo];
					outputs = GetInternalMlpOutputs[scaledInputs, weights, activationAndScaling];
					outputs[[1, 1]],
					
					{i, Length[networks]}
				];
			Return[CIP`Utility`GetPositionOfMaximumValue[combinedOutputs]]
		]
	];

CalculateMlpClassNumbers[

	(* Returns class numbers for specified inputs for classification mlp with specified weights.

	   Returns:
	   {class number of input1, class number of input2, ...} *)

    
    (* {inputsInOriginalUnit1, inputsInOriginalUnit2, ...}
        inputsInOriginalUnit: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    inputs_/;MatrixQ[inputs, NumberQ],
        
  	(* See "Frequently used data structures" *)
    mlpInfo_
    
	] :=
  
	Module[
    
		{
			combinedOutputs,
			correspondingOutput,
			i,
			networks,
			scaledInputs,
			outputs,
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			weights
		},

    	networks = mlpInfo[[1]];
    	dataSetScaleInfo = mlpInfo[[2]];
    	normalizationInfo = mlpInfo[[4]];
    	activationAndScaling = mlpInfo[[5]];

		If[Length[networks] == 1,
	
			(* --------------------------------------------------------------------------------
			   One network
			   -------------------------------------------------------------------------------- *)		

			weights = networks[[1]];
			(* Transform original inputs *)
			scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
			outputs = GetInternalMlpOutputs[scaledInputs, weights, activationAndScaling];
			Return[
				Table[CIP`Utility`GetPositionOfMaximumValue[outputs[[i]]], {i, Length[outputs]}]
			],
			
			(* --------------------------------------------------------------------------------
			   Multiple networks (with ONE output value each)
			   -------------------------------------------------------------------------------- *)		

			combinedOutputs =
				Table[
					weights = networks[[i]];
					(* Transform original input *)
					scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
					outputs = GetInternalMlpOutputs[scaledInputs, weights, activationAndScaling];
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

CalculateMlpCorrectClassificationInPercent[

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
    mlpInfo_
    
	] :=
  
	Module[
    
		{
			pureFunction
		},

		pureFunction = Function[inputs, CalculateMlpClassNumbers[inputs, mlpInfo]];
		Return[CIP`Utility`GetCorrectClassificationInPercent[classificationDataSet, pureFunction]]
	];

CalculateMlpDataSetRmse[

	(* Returns RMSE of data set.

	   Returns: 
	   RMSE of data set *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

  	(* See "Frequently used data structures" *)
    mlpInfo_
    
	] :=
  
	Module[
    
		{
			pureFunction,
			rmse
		},

		pureFunction = Function[inputs, CalculateMlpOutputs[inputs, mlpInfo]];
		rmse = Sqrt[CIP`Utility`GetMeanSquaredError[dataSet, pureFunction]];
		Return[rmse]
	];

CalculateMlpOutput[

	(* Calculates output for specified input for specified mlp.

	   Returns:
	   output: {transformedValueOfOutput1, transformedValueOfOutput2, ...} *)

    
    (* Input in original units: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlpInfo_
    
	] :=
  
	Module[
    
		{
			combinedOutputs,
			dataMatrixScaleInfo,
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			i,
			networks,
			outputsInOriginalUnits,
			scaledOutputs,
			scaledInputs,
			weights
		},
    
    	networks = mlpInfo[[1]];
    	dataSetScaleInfo = mlpInfo[[2]];
    	normalizationInfo = mlpInfo[[4]];
    	activationAndScaling = mlpInfo[[5]];

		If[Length[networks] == 1,
	
			(* --------------------------------------------------------------------------------
			   One network (with multiple output values)
			   -------------------------------------------------------------------------------- *)		

			weights = networks[[1]];
			(* Transform original input *)
			scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo];
			scaledOutputs = GetInternalMlpOutputs[scaledInputs, weights, activationAndScaling];
			(* Transform outputs to original units *)
			outputsInOriginalUnits = CIP`DataTransformation`ScaleDataMatrixReverse[scaledOutputs, dataSetScaleInfo[[2]]];
			Return[First[outputsInOriginalUnits]],
			
			(* --------------------------------------------------------------------------------
			   Multiple networks (with ONE output value each)
			   -------------------------------------------------------------------------------- *)		

			combinedOutputs =
				Table[
					weights = networks[[i]];
					(* Transform original input *)
					scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo];
					scaledOutputs = GetInternalMlpOutputs[scaledInputs, weights, activationAndScaling];

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

CalculateMlpOutputs[

	(* Calculates outputs for specified inputs for specified mlp.

	   Returns:
	   outputs: {output1, output2, ...} 
	   output: {transformedValueOfOutput1, transformedValueOfOutput1, ...} *)

    
    (* {inputsInOriginalUnit1, inputsInOriginalUnit2, ...}
        inputsInOriginalUnit: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    inputs_/;MatrixQ[inputs, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlpInfo_
    
	] :=
  
	Module[
    
		{
			combinedOutputs,
			dataMatrixScaleInfo,
			dataSetScaleInfo,
			normalizationInfo,
			activationAndScaling,
			i,
			networks,
			outputsInOriginalUnits,
			scaledOutputs,
			scaledInputs,
			weights
		},
		
    	networks = mlpInfo[[1]];
    	dataSetScaleInfo = mlpInfo[[2]];
    	normalizationInfo = mlpInfo[[4]];
    	activationAndScaling = mlpInfo[[5]];

		If[Length[networks] == 1,
	
			(* --------------------------------------------------------------------------------
			   One network (with multiple output values)
			   -------------------------------------------------------------------------------- *)		

			weights = networks[[1]];
			(* Transform original inputs *)
			scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
			scaledOutputs = GetInternalMlpOutputs[scaledInputs, weights, activationAndScaling];
			(* Transform outputs to original units *)
			outputsInOriginalUnits = CIP`DataTransformation`ScaleDataMatrixReverse[scaledOutputs, dataSetScaleInfo[[2]]];
			Return[outputsInOriginalUnits],
			
			(* --------------------------------------------------------------------------------
			   Multiple networks (with ONE output value each)
			   -------------------------------------------------------------------------------- *)		

			combinedOutputs =
				Table[
					weights = networks[[i]];
					(* Transform original input *)
					scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
					scaledOutputs = GetInternalMlpOutputs[scaledInputs, weights, activationAndScaling];

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

FitMultipleMlpSC[

	(* Training of multiple (1 mlp per output component of data set) Mlp.
	
	   Returns:
	   mlpInfo (see "Frequently used data structures") *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],
	
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
			mlpInfo,
			mlpTrainingResults,
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
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		reportIteration = MlpOptionReportIteration/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
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
		mlpTrainingResults = {};
		Do[
			(* If initial networks are available overwrite initialWeights *)
			If[Length[initialNetworks] > 0 && Length[initialNetworks] == Length[multipleTrainingSet],
				initialWeights = initialNetworks[[i]];
			];
			mlpInfo = 
				FitSingleMlp[
					{multipleTrainingSet[[i]], multipleTestSet[[i]]},
					numberOfHiddenNeurons,
	    			MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
		 			MlpOptionReportIteration -> reportIteration,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			MlpOptionActivationAndScaling -> activationAndScaling,
	    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
	    			MlpOptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				];
			AppendTo[networks, mlpInfo[[1, 1]]];
			AppendTo[mlpTrainingResults, mlpInfo[[3, 1]]],
			
			{i, Length[multipleTrainingSet]}
		];

		(* ----------------------------------------------------------------------------------------------------
		   Return mlpInfo
		   ---------------------------------------------------------------------------------------------------- *)
		Return[{networks, dataSetScaleInfo, mlpTrainingResults, normalizationInfo, activationAndScaling}]		
	];
	
FitMultipleMlpPC[

	(* Training of multiple (1 mlp per output component of data set) Mlp.
	
	   Returns:
	   mlpInfo (see "Frequently used data structures") *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],
	
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
			mlpTrainingResults,
			unusedOptionParameter21,
			randomValueInitialization,
			reportIteration,
			testSet,
			optimizationMethod,
			trainingSet,
			weightsValueLimit,
			mlpInfoList,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		reportIteration = MlpOptionReportIteration/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
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
		
		ParallelNeeds[{"CIP`Mlp`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[initialNetworks, multipleTrainingSet, multipleTestSet, optimizationMethod, initialWeights, 
			weightsValueLimit, minimizationPrecision, maximumNumberOfIterations, numberOfIterationsToImprove, 
			reportIteration, unusedOptionParameter11, unusedOptionParameter12, unusedOptionParameter13, unusedOptionParameter21, 
			unusedOptionParameter22, unusedOptionParameter23, randomValueInitialization, activationAndScaling, normalizationType, 
			lambdaL2Regularization, costFunctionType];

		(* ----------------------------------------------------------------------------------------------------
		   Training
		   ---------------------------------------------------------------------------------------------------- *)
		mlpInfoList = ParallelTable[
			(* If initial networks are available overwrite initialWeights *)
			If[Length[initialNetworks] > 0 && Length[initialNetworks] == Length[multipleTrainingSet],
				initialWeights = initialNetworks[[i]]
			];
			
			FitSingleMlp[
				{multipleTrainingSet[[i]], multipleTestSet[[i]]},
				numberOfHiddenNeurons,
	    		MlpOptionOptimizationMethod -> optimizationMethod,
				MlpOptionInitialWeights -> initialWeights,
				MlpOptionWeightsValueLimit -> weightsValueLimit,
				MlpOptionMinimizationPrecision -> minimizationPrecision,
				MlpOptionMaximumIterations -> maximumNumberOfIterations,
				MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
	 			MlpOptionReportIteration -> reportIteration,
				MlpOptionUnused11 -> unusedOptionParameter11,
				MlpOptionUnused12 -> unusedOptionParameter12,
				MlpOptionUnused13 -> unusedOptionParameter13,
				MlpOptionUnused21 -> unusedOptionParameter21,
				MlpOptionUnused22 -> unusedOptionParameter22,
				MlpOptionUnused23 -> unusedOptionParameter23,
	    		UtilityOptionRandomInitializationMode -> randomValueInitialization,
	   			MlpOptionActivationAndScaling -> activationAndScaling,
	   			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
	   			MlpOptionCostFunctionType -> costFunctionType,
	   			DataTransformationOptionNormalizationType -> normalizationType
			],
			{i, Length[multipleTrainingSet]}
		];
		networks = Table[mlpInfoList[[i, 1, 1]], {i, Length[multipleTrainingSet]}];
		mlpTrainingResults = Table[mlpInfoList[[i, 3, 1]], {i, Length[multipleTrainingSet]}];
		(* ----------------------------------------------------------------------------------------------------
		   Return mlpInfo
		   ---------------------------------------------------------------------------------------------------- *)
		Return[{networks, dataSetScaleInfo, mlpTrainingResults, normalizationInfo, activationAndScaling}]		
	];

FitMlp[

	(* Training of single or multiple Mlp(s).

	   Returns:
	   mlpInfo (see "Frequently used data structures") *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],

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
			multipleMlps,
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
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
	    testSet = MlpOptionTestSet/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		reportIteration = MlpOptionReportIteration/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		(* ----------------------------------------------------------------------------------------------------
		   Switch training method
		   ---------------------------------------------------------------------------------------------------- *)
		trainingAndTestSet = {dataSet, testSet};
		
		If[multipleMlps,
			
			Switch[parallelization,
			
				(* ------------------------------------------------------------------------------- *)
				"ParallelCalculation",
				Return[
					FitMultipleMlpPC[
						trainingAndTestSet,
						numberOfHiddenNeurons,
		    			MlpOptionOptimizationMethod -> optimizationMethod,
						MlpOptionInitialWeights -> initialWeights,
						MlpOptionInitialNetworks -> initialNetworks,
						MlpOptionWeightsValueLimit -> weightsValueLimit,
						MlpOptionMinimizationPrecision -> minimizationPrecision,
						MlpOptionMaximumIterations -> maximumNumberOfIterations,
						MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
			 			MlpOptionReportIteration -> reportIteration,
						MlpOptionUnused11 -> unusedOptionParameter11,
						MlpOptionUnused12 -> unusedOptionParameter12,
						MlpOptionUnused13 -> unusedOptionParameter13,
						MlpOptionUnused21 -> unusedOptionParameter21,
						MlpOptionUnused22 -> unusedOptionParameter22,
						MlpOptionUnused23 -> unusedOptionParameter23,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			MlpOptionActivationAndScaling -> activationAndScaling,
		    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
		    			MlpOptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					]
				],

				(* ------------------------------------------------------------------------------- *)
				"SequentialCalculation",
				Return[
					FitMultipleMlpSC[
						trainingAndTestSet,
						numberOfHiddenNeurons,
		    			MlpOptionOptimizationMethod -> optimizationMethod,
						MlpOptionInitialWeights -> initialWeights,
						MlpOptionInitialNetworks -> initialNetworks,
						MlpOptionWeightsValueLimit -> weightsValueLimit,
						MlpOptionMinimizationPrecision -> minimizationPrecision,
						MlpOptionMaximumIterations -> maximumNumberOfIterations,
						MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
			 			MlpOptionReportIteration -> reportIteration,
						MlpOptionUnused11 -> unusedOptionParameter11,
						MlpOptionUnused12 -> unusedOptionParameter12,
						MlpOptionUnused13 -> unusedOptionParameter13,
						MlpOptionUnused21 -> unusedOptionParameter21,
						MlpOptionUnused22 -> unusedOptionParameter22,
						MlpOptionUnused23 -> unusedOptionParameter23,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			MlpOptionActivationAndScaling -> activationAndScaling,
		    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
		    			MlpOptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					]
				]
			],
			
			Return[
				FitSingleMlp[
					trainingAndTestSet,
					numberOfHiddenNeurons,
	    			MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
		 			MlpOptionReportIteration -> reportIteration,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			MlpOptionActivationAndScaling -> activationAndScaling,
	    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
	    			MlpOptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				]
			]
		]
	];

FitMlpSeries[

	(* Trains of a series of single or multiple Mlp(s).

	   Returns:
	   mlpInfoList: {mlpInfo1, mlpInfo2, ...}
	   mlpInfo[[i]] corresponds to numberOfHiddenNeuronsList[[i]]
	   (see "Frequently used data structures") *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* List with numbers of hidden neurons:
	   numberOfHiddenNeuronsList: {numberOfHiddenNeurons1, numberOfHiddenNeurons2, ...}
	   numberOfHiddenNeurons: {<number of neurons in hidden1>, <... in hidden2>, ...} *)
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
			multipleMlps,
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
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
	    testSet = MlpOptionTestSet/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		reportIteration = MlpOptionReportIteration/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
	    (* Parallelization options *)
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				FitMlpSeriesPC[
					dataSet,
					numberOfHiddenNeuronsList,
					MlpOptionMultipleMlps -> multipleMlps,
				    MlpOptionOptimizationMethod -> optimizationMethod,
				    MlpOptionTestSet -> testSet,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
					MlpOptionReportIteration -> reportIteration,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
				    MlpOptionActivationAndScaling -> activationAndScaling,
				    MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
				    MlpOptionCostFunctionType -> costFunctionType,
				    DataTransformationOptionNormalizationType -> normalizationType
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				FitMlpSeriesSC[
					dataSet,
					numberOfHiddenNeuronsList,
					MlpOptionMultipleMlps -> multipleMlps,
				    MlpOptionOptimizationMethod -> optimizationMethod,
				    MlpOptionTestSet -> testSet,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
					MlpOptionReportIteration -> reportIteration,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
				    MlpOptionActivationAndScaling -> activationAndScaling,
				    MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
				    MlpOptionCostFunctionType -> costFunctionType,
				    DataTransformationOptionNormalizationType -> normalizationType
				]
			]
		]
	];

FitMlpSeriesSC[

	(* Trains of a series of single or multiple Mlp(s).

	   Returns:
	   mlpInfoList: {mlpInfo1, mlpInfo2, ...}
	   mlpInfo[[i]] corresponds to numberOfHiddenNeuronsList[[i]]
	   (see "Frequently used data structures") *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* List with numbers of hidden neurons:
	   numberOfHiddenNeuronsList: {numberOfHiddenNeurons1, numberOfHiddenNeurons2, ...}
	   numberOfHiddenNeurons: {<number of neurons in hidden1>, <... in hidden2>, ...} *)
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
			multipleMlps,
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
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
	    testSet = MlpOptionTestSet/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		reportIteration = MlpOptionReportIteration/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		Return[
			Table[
				FitMlp[
					dataSet,
					numberOfHiddenNeuronsList[[i]],
					MlpOptionTestSet -> testSet,
					MlpOptionMultipleMlps -> multipleMlps,
	    			MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
		 			MlpOptionReportIteration -> reportIteration,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			MlpOptionActivationAndScaling -> activationAndScaling,
	    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
	    			MlpOptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				],
				
				{i, Length[numberOfHiddenNeuronsList]}
			]			
		]
	];
	
FitMlpSeriesPC[

	(* Trains of a series of single or multiple Mlp(s).

	   Returns:
	   mlpInfoList: {mlpInfo1, mlpInfo2, ...}
	   mlpInfo[[i]] corresponds to numberOfHiddenNeuronsList[[i]]
	   (see "Frequently used data structures") *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* List with numbers of hidden neurons:
	   numberOfHiddenNeuronsList: {numberOfHiddenNeurons1, numberOfHiddenNeurons2, ...}
	   numberOfHiddenNeurons: {<number of neurons in hidden1>, <... in hidden2>, ...} *)
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
			multipleMlps,
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
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
	    testSet = MlpOptionTestSet/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		reportIteration = MlpOptionReportIteration/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		
		ParallelNeeds[{"CIP`Mlp`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[testSet, multipleMlps, optimizationMethod, initialWeights, initialNetworks, 
			weightsValueLimit, minimizationPrecision, maximumNumberOfIterations, numberOfIterationsToImprove, 
			reportIteration, unusedOptionParameter11, unusedOptionParameter12, unusedOptionParameter13, unusedOptionParameter21, 
			unusedOptionParameter22, unusedOptionParameter23, randomValueInitialization, activationAndScaling, normalizationType, 
			lambdaL2Regularization, costFunctionType];

		Return[
			ParallelTable[
				FitMlp[
					dataSet,
					numberOfHiddenNeuronsList[[i]],
					MlpOptionTestSet -> testSet,
					MlpOptionMultipleMlps -> multipleMlps,
   	 				MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
	 				MlpOptionReportIteration -> reportIteration,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
    				UtilityOptionRandomInitializationMode -> randomValueInitialization,
    				MlpOptionActivationAndScaling -> activationAndScaling,
    				MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
    				MlpOptionCostFunctionType -> costFunctionType,
    				DataTransformationOptionNormalizationType -> normalizationType
				],
			
				{i, Length[numberOfHiddenNeuronsList]}
			]			
		]
	];

FitMlpWithFindMinimum[

	(* Training of mlp with FindMinimum and "ConjugateGradient" method.
	
	   Returns:
	   mlpInfo (see "Frequently used data structures") *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],
	
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
			hiddenLayerActivations,
			initialWeights,
			inputs,
			intermediateResult,
			i,
			j,
			k,
			lambdaL2Regularization,
			lastTrainingStep,
			maximumNumberOfIterations,
			costFunction,
			minimizationPrecision,
			minimizationStep,
			minimumInfo,
			numberOfHiddenLayers,
			numberOfInputs,
			numberOfIOPairs,
			numberOfOutputs,
			outputs,
			outputWeightsValueLimit,
			mlpOutputs,
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
			wHiddenToHidden,
			wHiddenToOutput,
			weightsRules,
			weightsVariables,
			weights,
			hiddenWeightsValueLimit,
			weightsVariablesWithoutTrueUnitBias
		},


		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
	    minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
	    maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
	    reportIteration = MlpOptionReportIteration/.{opts}/.Options[MlpOptionsOptimization];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		(* Set seed for random numbers if necessary *)
		If[randomValueInitialization == "Seed", SeedRandom[1], SeedRandom[]];

		numberOfHiddenLayers = Length[numberOfHiddenNeurons];

		(* Check activationAndScaling, e.g. {{"Sigmoid", "Sigmoid", "Sigmoid", "Sigmoid"}, {{-0.9, 0.9}, {0.1, 0.9}}} *)
		If[Length[activationAndScaling] == 0,
			hiddenLayerActivations =
				AppendTo[
					Table[
						"Tanh",
						
						{i, numberOfHiddenLayers}
					],
					"Sigmoid"
				];
			activationAndScaling = {hiddenLayerActivations, {{-0.9, 0.9}, {0.1, 0.9}}};
		];

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

	    (* Initialize hidden and output weights *)
	    If[Length[initialWeights] > 0,
      
			(* Use specified weights as initial weights *)
			weights = initialWeights,
      
			(* Use random weights as initial weights:
			   Y. Bengio, Practical Recommendations for Gradient-Based Training of Deep Architectures, https://arxiv.org/abs/1206.5533v2
			   Weight initialization for sigmoid activation neurons:
			   weightsValueLimit = 4.0 * Sqrt[6.0/(numberOfIn + 1 + numberOfOut)];
			   Weight initialization for tanh activation neurons:
			   weightsValueLimit = Sqrt[6.0/(numberOfIn + 1 + numberOfOut)];
			*)
			weights = 
				Table[
					{},
					
					{i, numberOfHiddenLayers + 1}
				];
			Switch[activationAndScaling[[1, 1]],
				
				"Sigmoid",
				hiddenWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons[[1]])],
				
				"Tanh",
				hiddenWeightsValueLimit = Sqrt[6.0/(numberOfInputs + 1 + numberOfHiddenNeurons[[1]])]
			];
			(* 'True unit' : 'numberOfInputs + 1' and 'numberOfHiddenNeurons + 1' *)
			weights[[1]] = Table[RandomReal[{-hiddenWeightsValueLimit, hiddenWeightsValueLimit}, numberOfInputs + 1], {numberOfHiddenNeurons[[1]]}];
			If[numberOfHiddenLayers > 1,
				Do[
					Switch[activationAndScaling[[1, k]],
						
						"Sigmoid",
						hiddenWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfHiddenNeurons[[k - 1]] + 1 + numberOfHiddenNeurons[[k]])],
						
						"Tanh",
						hiddenWeightsValueLimit = Sqrt[6.0/(numberOfHiddenNeurons[[k - 1]] + 1 + numberOfHiddenNeurons[[k]])];
					];
					(* 'True unit' : 'numberOfInputs + 1' and 'numberOfHiddenNeurons + 1' *)
					weights[[k]] = Table[RandomReal[{-hiddenWeightsValueLimit, hiddenWeightsValueLimit}, numberOfHiddenNeurons[[k - 1]] + 1], {numberOfHiddenNeurons[[k]]}],
					
					{k, 2, numberOfHiddenLayers}
				]
			];
			Switch[activationAndScaling[[1, numberOfHiddenLayers + 1]],
				
				"Sigmoid",
				outputWeightsValueLimit = 4.0 * Sqrt[6.0/(numberOfHiddenNeurons[[numberOfHiddenLayers]] + 1 + numberOfOutputs)],
				
				"Tanh",
				outputWeightsValueLimit = Sqrt[6.0/(numberOfHiddenNeurons[[numberOfHiddenLayers]] + 1 + numberOfOutputs)]
			];
			(* 'True unit' : 'numberOfInputs + 1' and 'numberOfHiddenNeurons + 1' *)
			weights[[numberOfHiddenLayers + 1]] = Table[RandomReal[{-outputWeightsValueLimit, outputWeightsValueLimit}, numberOfHiddenNeurons[[numberOfHiddenLayers]] + 1], {numberOfOutputs}]
		];

		(* Initialize training protocol *)
		trainingMeanSquaredErrorList = {{0, GetInternalMeanSquaredErrorOfMlp[scaledTrainingSet, weights, activationAndScaling]}};
		If[Length[scaledTestSet] > 0,
		
			(* Test set exists *)
			testMeanSquaredErrorList = {{0, GetInternalMeanSquaredErrorOfMlp[scaledTestSet, weights, activationAndScaling]}},
		
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
				wHiddenToHidden,
				wHiddenToOutput, 
				weights
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
	    		wHiddenToHidden, 
	    		wHiddenToOutput
    		];
	    
	    (* Map: Add 'true unit' *)
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			intermediateResult = SigmoidFunction[Map[Append[#, 1] &, inputs].Transpose[weightsVariables[[1]]]],
			
			"Tanh",
			intermediateResult = Tanh[Map[Append[#, 1] &, inputs].Transpose[weightsVariables[[1]]]]
		];

		If[numberOfHiddenLayers > 1,
			Do[
				Switch[activationAndScaling[[1, k]],
					
					"Sigmoid",
					intermediateResult = SigmoidFunction[Map[Append[#, 1] &, intermediateResult].Transpose[weightsVariables[[k]]]],
					
					"Tanh",
					intermediateResult = Tanh[Map[Append[#, 1] &, intermediateResult].Transpose[weightsVariables[[k]]]]
				],
				
				{k, 2, numberOfHiddenLayers}
			]
		];

		Switch[activationAndScaling[[1, numberOfHiddenLayers + 1]],
			
			"Sigmoid",
			mlpOutputs = SigmoidFunction[Map[Append[#, 1] &, intermediateResult].Transpose[weightsVariables[[numberOfHiddenLayers + 1]]]],
			
			"Tanh",
			Switch[costFunctionType,
			
				"SquaredError",				
				mlpOutputs = Tanh[Map[Append[#, 1] &, intermediateResult].Transpose[weightsVariables[[numberOfHiddenLayers + 1]]]],
				
				(* Cross-entropy cost function arguments MUST be in interval {0, 1} *)
				"Cross-Entropy",
				mlpOutputs = 0.5 * (Tanh[Map[Append[#, 1] &, intermediateResult].Transpose[weightsVariables[[numberOfHiddenLayers + 1]]]] + 1.0)
			]
		];

		weightsVariablesWithoutTrueUnitBias = 
			GetWeightsVariablesWithoutTrueUnitBias[
				numberOfInputs, 
				numberOfHiddenNeurons, 
				numberOfOutputs, 
				wInputToHidden1, 
				wHiddenToHidden, 
				wHiddenToOutput
			];
	    
	    Switch[costFunctionType,
	    	
	    	"SquaredError",
		    If[lambdaL2Regularization == 0.0,
	
				(* NO L2 regularization *)
				costFunction =
					Sum[
						Sum[
							(outputs[[i, k]] - mlpOutputs[[i, k]])^2,
							
							{k, numberOfOutputs}
						],
							    
						{i, numberOfIOPairs}	
					]/numberOfIOPairs,
		    	
				(* L2 regularization *)
				costFunction =
					Sum[
						Sum[
							(outputs[[i, k]] - mlpOutputs[[i, k]])^2,
							
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
							outputs[[i, k]] * Log[mlpOutputs[[i, k]]] + (1.0 - outputs[[i, k]]) * Log[1.0 - mlpOutputs[[i, k]]],
							
							{k, numberOfOutputs}
						],
							    
						{i, numberOfIOPairs}	
					]/numberOfIOPairs,
		    	
				(* L2 regularization *)
				costFunction =
					Sum[
						Sum[
							outputs[[i, k]] * Log[mlpOutputs[[i, k]]] + (1.0 - outputs[[i, k]]) * Log[1.0 - mlpOutputs[[i, k]]],
							
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
					{stepNumber, GetInternalMeanSquaredErrorOfMlp[scaledTrainingSet, weights, activationAndScaling]}
				];
				If[Length[scaledTestSet] > 0,
					(* Test set exists *)
					AppendTo[testMeanSquaredErrorList, 
						{stepNumber, GetInternalMeanSquaredErrorOfMlp[scaledTestSet, weights, activationAndScaling]}
					]
				],
				
				{i, Length[minimumInfo[[2, 1]]]}
			]
		];
			
		(* ----------------------------------------------------------------------------------------------------
		   Set results
		   ---------------------------------------------------------------------------------------------------- *)
		weights = 
			GetWeightsVariables[
				numberOfInputs, 
				numberOfHiddenNeurons, 
				numberOfOutputs, 
				wInputToHidden1, 
				wHiddenToHidden, 
				wHiddenToOutput
			]/.weightsRules;

		(* End of training protocol *)
		lastTrainingStep = Last[trainingMeanSquaredErrorList];
		If[lastTrainingStep[[1]] < steps,
			AppendTo[trainingMeanSquaredErrorList, 
				{steps, GetInternalMeanSquaredErrorOfMlp[scaledTrainingSet, weights, activationAndScaling]}
			];
			If[Length[scaledTestSet] > 0,
				(* Test set exists *)
				AppendTo[testMeanSquaredErrorList, 
					{steps, GetInternalMeanSquaredErrorOfMlp[scaledTestSet, weights, activationAndScaling]}
				]
			]
		];
		
		(* Return mlpInfo *)
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

FitMlpWithNMinimize[

	(* Training of mlp with NMinimize and "DifferentialEvolution".
	
	   Returns:
	   mlpInfo (see "Frequently used data structures") *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],

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
			numberOfHiddenLayers,
			hiddenLayerActivations,
			inputs,
			intermediateResult,
			i,
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
			mlpOutputs,
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
    		wHiddenToHidden, 
    		wHiddenToOutput,
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
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
	    minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
	    maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
	    reportIteration = MlpOptionReportIteration/.{opts}/.Options[MlpOptionsOptimization];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		(* Set seed for random numbers if necessary *)
		If[randomValueInitialization == "Seed", SeedRandom[1], SeedRandom[]];

		(* Check activationAndScaling, e.g. {{"Sigmoid", "Sigmoid", "Sigmoid", "Sigmoid"}, {{-0.9, 0.9}, {0.1, 0.9}}} *)
		If[Length[activationAndScaling] == 0,
			numberOfHiddenLayers = Length[numberOfHiddenNeurons];
			hiddenLayerActivations =
				AppendTo[
					Table[
						"Tanh",
						
						{i, numberOfHiddenLayers}
					],
					"Sigmoid"
				];
			activationAndScaling = {hiddenLayerActivations, {{-0.9, 0.9}, {0.1, 0.9}}};
		];

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
	    		wHiddenToHidden, 
	    		wHiddenToOutput
    		];

	    (* Map: Add 'true unit' *)
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			intermediateResult = SigmoidFunction[Map[Append[#, 1] &, inputs].Transpose[weightsVariables[[1]]]],
			
			"Tanh",
			intermediateResult = Tanh[Map[Append[#, 1] &, inputs].Transpose[weightsVariables[[1]]]]
		];

		If[numberOfHiddenLayers > 1,
			Do[
				Switch[activationAndScaling[[1, k]],
					
					"Sigmoid",
					intermediateResult = SigmoidFunction[Map[Append[#, 1] &, intermediateResult].Transpose[weightsVariables[[k]]]],
					
					"Tanh",
					intermediateResult = Tanh[Map[Append[#, 1] &, intermediateResult].Transpose[weightsVariables[[k]]]]
				],
				
				{k, 2, numberOfHiddenLayers}
			]
		];

		Switch[activationAndScaling[[1, numberOfHiddenLayers + 1]],
			
			"Sigmoid",
			mlpOutputs = SigmoidFunction[Map[Append[#, 1] &, intermediateResult].Transpose[weightsVariables[[numberOfHiddenLayers + 1]]]],
			
			"Tanh",
			Switch[costFunctionType,
			
				"SquaredError",				
				mlpOutputs = Tanh[Map[Append[#, 1] &, intermediateResult].Transpose[weightsVariables[[numberOfHiddenLayers + 1]]]],
				
				(* Cross-entropy cost function arguments MUST be in interval {0, 1} *)
				"Cross-Entropy",
				mlpOutputs = 0.5 * (Tanh[Map[Append[#, 1] &, intermediateResult].Transpose[weightsVariables[[numberOfHiddenLayers + 1]]]] + 1.0)
			]
		];

		weightsVariablesWithoutTrueUnitBias = 
			GetWeightsVariablesWithoutTrueUnitBias[
				numberOfInputs, 
				numberOfHiddenNeurons, 
				numberOfOutputs, 
				wInputToHidden1, 
				wHiddenToHidden, 
				wHiddenToOutput
			];
	    
	    Switch[costFunctionType,
	    	
	    	"SquaredError",
		    If[lambdaL2Regularization == 0.0,
	
				(* NO L2 regularization *)
				costFunction =
					Sum[
						Sum[
							(outputs[[i, k]] - mlpOutputs[[i, k]])^2,
							
							{k, numberOfOutputs}
						],
							    
						{i, numberOfIOPairs}	
					]/numberOfIOPairs,
		    	
				(* L2 regularization *)
				costFunction =
					Sum[
						Sum[
							(outputs[[i, k]] - mlpOutputs[[i, k]])^2,
							
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
							outputs[[i, k]] * Log[mlpOutputs[[i, k]]] + (1.0 - outputs[[i, k]]) * Log[1.0 - mlpOutputs[[i, k]]],
							
							{k, numberOfOutputs}
						],
							    
						{i, numberOfIOPairs}	
					]/numberOfIOPairs,
		    	
				(* L2 regularization *)
				costFunction =
					Sum[
						Sum[
							outputs[[i, k]] * Log[mlpOutputs[[i, k]]] + (1.0 - outputs[[i, k]]) * Log[1.0 - mlpOutputs[[i, k]]],
							
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
					{stepNumber, GetInternalMeanSquaredErrorOfMlp[scaledTrainingSet, weights, activationAndScaling]}
				];
				If[Length[scaledTestSet] > 0,
					(* Test set exists *)
					AppendTo[testMeanSquaredErrorList, 
						{stepNumber, GetInternalMeanSquaredErrorOfMlp[scaledTestSet, weights, activationAndScaling]}
					]
				],
				
				{i, Length[minimumInfo[[2, 1]]]}
			]
		];
			
		(* ----------------------------------------------------------------------------------------------------
		   Set results
		   ---------------------------------------------------------------------------------------------------- *)
		weights = 
			GetWeightsVariables[
				numberOfInputs, 
				numberOfHiddenNeurons, 
				numberOfOutputs, 
				wInputToHidden1, 
				wHiddenToHidden, 
				wHiddenToOutput
			]/.weightsRules;

		(* End of training protocol *)
		lastTrainingStep = Last[trainingMeanSquaredErrorList];
		If[lastTrainingStep[[1]] < steps,
			AppendTo[trainingMeanSquaredErrorList, 
				{steps, GetInternalMeanSquaredErrorOfMlp[scaledTrainingSet, weights, activationAndScaling]}
			];
			If[Length[scaledTestSet] > 0,
				(* Test set exists *)
				AppendTo[testMeanSquaredErrorList, 
					{steps, GetInternalMeanSquaredErrorOfMlp[scaledTestSet, weights, activationAndScaling]}
				]
			]
		];
		
		(* Return mlpInfo *)
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

FitSingleMlp[

	(* Training of single Mlp.

	   Returns:
	   mlpInfo (see "Frequently used data structures") *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],

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
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		reportIteration = MlpOptionReportIteration/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		
		(* ----------------------------------------------------------------------------------------------------
		   Switch training method
		   ---------------------------------------------------------------------------------------------------- *)
		Switch[optimizationMethod,
			
			"FindMinimum",
			Return[
				FitMlpWithFindMinimum[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					MlpOptionInitialWeights -> initialWeights,
	    			MlpOptionMinimizationPrecision -> minimizationPrecision,
	    			MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionReportIteration -> reportIteration,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			MlpOptionActivationAndScaling -> activationAndScaling,
	    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
	    			MlpOptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				]
			],
			
			"NMinimize",
			Return[
				FitMlpWithNMinimize[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
	    			MlpOptionMinimizationPrecision -> minimizationPrecision,
	    			MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionReportIteration -> reportIteration,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			MlpOptionActivationAndScaling -> activationAndScaling,
	    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
	    			MlpOptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				]
			]
		]
	];

GetBestMlpClassOptimization[

	(* Returns best training set optimization result of mlp for classification.

	   Returns: 
	   Best index for classification *)


	(* mlpTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlpInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlpInfoList: List with mlpInfo
	   mlpInfoList[[i]] refers to optimization step i *)
	mlpTrainOptimization_,
	
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
				GetBestMlpClassOptimizationPC[
					mlpTrainOptimization,
					UtilityOptionBestOptimization -> bestOptimization
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetBestMlpClassOptimizationSC[
					mlpTrainOptimization,
					UtilityOptionBestOptimization -> bestOptimization
				]
			]
		]
	];

GetBestMlpClassOptimizationSC[

	(* Returns best training set optimization result of mlp for classification.

	   Returns: 
	   Best index for classification *)


	(* mlpTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlpInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlpInfoList: List with mlpInfo
	   mlpInfoList[[i]] refers to optimization step i *)
	mlpTrainOptimization_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			k,
			trainingAndTestSetList,
			mlpInfoList,
			maximumCorrectClassificationInPercent,
			mlpInfo,
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
			trainingAndTestSetList = mlpTrainOptimization[[3]];
			mlpInfoList = mlpTrainOptimization[[4]];
			maximumCorrectClassificationInPercent = -1.0;
			Do[
				testSet = trainingAndTestSetList[[k, 2]];
				mlpInfo = mlpInfoList[[k]];
				correctClassificationInPercent = CalculateMlpCorrectClassificationInPercent[testSet, mlpInfo];
				If[correctClassificationInPercent > maximumCorrectClassificationInPercent,
					maximumCorrectClassificationInPercent = correctClassificationInPercent;
					bestIndex = k
				],
				
				{k, Length[mlpInfoList]}
			],
			
			(* ------------------------------------------------------------------------------- *)			
			"MinimumDeviation",
			trainingAndTestSetList = mlpTrainOptimization[[3]];
			mlpInfoList = mlpTrainOptimization[[4]];
			minimumDeviation = Infinity;
			Do[
				trainingSet = trainingAndTestSetList[[k, 1]];
				testSet = trainingAndTestSetList[[k, 2]];
				mlpInfo = mlpInfoList[[k]];
				trainingSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[trainingSet, mlpInfo];
				testSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[testSet, mlpInfo];
				deviation = Abs[testSetCorrectClassificationInPercent - trainingSetCorrectClassificationInPercent];
				If[deviation < minimumDeviation || (deviation == minimumDeviation && testSetCorrectClassificationInPercent < bestTestSetCorrectClassificationInPercent),
					minimumDeviation = deviation;
					bestTestSetCorrectClassificationInPercent = testSetCorrectClassificationInPercent;
					bestIndex = k
				],
				
				{k, Length[mlpInfoList]}
			]
		];

		Return[bestIndex]
	];

GetBestMlpClassOptimizationPC[

	(* Returns best training set optimization result of mlp for classification.

	   Returns: 
	   Best index for classification *)


	(* mlpTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlpInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlpInfoList: List with mlpInfo
	   mlpInfoList[[i]] refers to optimization step i *)
	mlpTrainOptimization_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			k,
			trainingAndTestSetList,
			mlpInfoList,
			maximumCorrectClassificationInPercent,
			mlpInfo,
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
			trainingAndTestSetList = mlpTrainOptimization[[3]];
			mlpInfoList = mlpTrainOptimization[[4]];
			maximumCorrectClassificationInPercent = -1.0;
			
			ParallelNeeds[{"CIP`Mlp`", "CIP`DataTransformation`", "CIP`Utility`"}];
			DistributeDefinitions[trainingAndTestSetList, mlpInfoList];
			
			correctClassificationInPercentList = ParallelTable[
				testSet = trainingAndTestSetList[[k, 2]];
				mlpInfo = mlpInfoList[[k]];
				
				CalculateMlpCorrectClassificationInPercent[testSet, mlpInfo],
				
				{k, Length[mlpInfoList]}
			];
			
			Do[
				If[correctClassificationInPercentList[[k]] > maximumCorrectClassificationInPercent,
					maximumCorrectClassificationInPercent = correctClassificationInPercentList[[k]];
					bestIndex = k
				],
				
				{k, Length[mlpInfoList]}
			],
			
			(* ------------------------------------------------------------------------------- *)			
			"MinimumDeviation",
			trainingAndTestSetList = mlpTrainOptimization[[3]];
			mlpInfoList = mlpTrainOptimization[[4]];
			minimumDeviation = Infinity;
			
			ParallelNeeds[{"CIP`Mlp`", "CIP`DataTransformation`", "CIP`Utility`"}];
			DistributeDefinitions[trainingAndTestSetList, mlpInfoList];
			
			listOfTestSetCorrectClassificationInPercentAndDeviation = ParallelTable[
				trainingSet = trainingAndTestSetList[[k, 1]];
				testSet = trainingAndTestSetList[[k, 2]];
				mlpInfo = mlpInfoList[[k]];
				trainingSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[trainingSet, mlpInfo];
				testSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[testSet, mlpInfo];
				deviation = Abs[testSetCorrectClassificationInPercent - trainingSetCorrectClassificationInPercent];
				
				{
					testSetCorrectClassificationInPercent,
					deviation
				},
				
				{k, Length[mlpInfoList]}
			];
			
			Do[
				If[listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]] < minimumDeviation || (listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]] == minimumDeviation && listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 1]] < bestTestSetCorrectClassificationInPercent),
					minimumDeviation = listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]];
					bestTestSetCorrectClassificationInPercent = listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 1]];
					bestIndex = k
				],
				
				{k, Length[mlpInfoList]}
			]
		];

		Return[bestIndex]
	];

GetBestMlpRegressOptimization[

	(* Returns best optimization result of mlp for regression.

	   Returns: 
	   Best index for regression *)


	(* mlpTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlpInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlpInfoList: List with mlpInfo
	   mlpInfoList[[i]] refers to optimization step i *)
	mlpTrainOptimization_,
	
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
				mlpTrainOptimization, 
				UtilityOptionBestOptimization -> bestOptimization
			]
		]
	];

GetInternalMeanSquaredErrorOfMlp[

	(* Calculates mean squared error of specified data set for mlp with specified weights

	   Returns:
	   Mean squared error of data set *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output: {outputComponent1, outputComponent2, ...})
	   NOTE: Each component must be in [0, 1] *)
    dataSet_,

    (* Weights *)
    weights_,
    
    (* Activation and scaling, see MlpOptionActivationAndScaling *)
    activationAndScaling_
    
	] :=
  
	Module[
    
		{
			errors,
			hiddenOutput,
			hiddenWeights,
			outputWeights,
			numberOfHiddenLayers,
			i,
			inputs,
			machineOutputs,
			outputs
		},
    
		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];
		outputs = CIP`Utility`GetOutputsOfDataSet[dataSet];

		numberOfHiddenLayers = Length[weights] - 1; 
		hiddenWeights = 
			Table[
				weights[[i]],
				
				{i, numberOfHiddenLayers}
			];
		outputWeights = weights[[numberOfHiddenLayers + 1]];

		hiddenOutput = 
			Table[
				{},
				
				{i, numberOfHiddenLayers}
			];

	    (* Add 'true unit' to inputs *)
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			hiddenOutput[[1]] = SigmoidFunction[Map[Append[#, 1.0] &, inputs].Transpose[hiddenWeights[[1]]]],
			
			"Tanh",
			hiddenOutput[[1]] = Tanh[Map[Append[#, 1.0] &, inputs].Transpose[hiddenWeights[[1]]]]
		];

		If[numberOfHiddenLayers > 1,
			Do[
			    (* Add 'true unit' to last hidden *)
				Switch[activationAndScaling[[1, i]],
			
					"Sigmoid",
					hiddenOutput[[i]] = SigmoidFunction[Map[Append[#, 1.0] &, hiddenOutput[[i - 1]]].Transpose[hiddenWeights[[i]]]],
					
					"Tanh",
					hiddenOutput[[i]] = Tanh[Map[Append[#, 1.0] &, hiddenOutput[[i - 1]]].Transpose[hiddenWeights[[i]]]]
				],
				
				{i, 2, numberOfHiddenLayers}
			]
		];

	    (* Add 'true unit' to last hidden *)
		Switch[activationAndScaling[[1, numberOfHiddenLayers + 1]],
			
			"Sigmoid",
			machineOutputs = SigmoidFunction[Map[Append[#, 1.0] &, hiddenOutput[[numberOfHiddenLayers]]].Transpose[outputWeights]],
			
			"Tanh",
			machineOutputs = Tanh[Map[Append[#, 1.0] &, hiddenOutput[[numberOfHiddenLayers]]].Transpose[outputWeights]]
		];

	    errors = outputs - machineOutputs;
        Return[Apply[Plus, errors^2, {0,1}]/Length[dataSet]]
	];

GetInternalMlpOutput[

	(* Calculates internal output for specified input of mlp with specified weights.

	   Returns:
	   output: {valueOfOutput1, valueOfOutput2, ...} *)

    
    (* input: {valueForInput1, valueForInput1, ...} *)
    input_/;VectorQ[input, NumberQ],

    (* Weights *)
    weights_,
    
    (* Activation and scaling, see MlpOptionActivationAndScaling *)
    activationAndScaling_
    
	] :=
  
	Module[
    
		{
			hiddenOutput,
			hiddenWeights,
			numberOfHiddenLayers,
			i,
			outputWeights,
			internalInputs,
			trueUnitHidden,
			outputs
		},

		numberOfHiddenLayers = Length[weights] - 1; 
		hiddenWeights = 
			Table[
				weights[[i]],
				
				{i, numberOfHiddenLayers}
			];
		outputWeights = weights[[numberOfHiddenLayers + 1]];

		hiddenOutput = 
			Table[
				{},
				
				{i, numberOfHiddenLayers}
			];

		(* Add 'true unit' to inputs *)
		internalInputs = Append[input, 1.0];
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			hiddenOutput[[1]] = SigmoidFunction[internalInputs.Transpose[hiddenWeights[[1]]]],
			
			"Tanh",
			hiddenOutput[[1]] = Tanh[internalInputs.Transpose[hiddenWeights[[1]] ]]
		];

		If[numberOfHiddenLayers > 1,
			Do[
			    (* Add 'true unit' to last hidden *)
				trueUnitHidden = Append[hiddenOutput[[i - 1]], 1.0];
				Switch[activationAndScaling[[1, i]],
					
					"Sigmoid",
					hiddenOutput[[i]] = SigmoidFunction[trueUnitHidden.Transpose[hiddenWeights[[i]]]],
					
					"Tanh",
					hiddenOutput[[i]] = Tanh[trueUnitHidden.Transpose[hiddenWeights[[i]]]]
				],
				
				{i, 2, numberOfHiddenLayers}
			]
		];

	    (* Add 'true unit' to last hidden *)
		trueUnitHidden = Append[hiddenOutput[[numberOfHiddenLayers]], 1.0];
		Switch[activationAndScaling[[1, numberOfHiddenLayers + 1]],
			
			"Sigmoid",
			outputs = SigmoidFunction[trueUnitHidden.Transpose[outputWeights]],
			
			"Tanh",
			outputs = Tanh[trueUnitHidden.Transpose[outputWeights]]
		];
		
		Return[outputs];
    ];

GetInternalMlpOutputs[

	(* Calculates internal outputs for specified inputs for mlp with specified weights.

	   Returns:
	   outputs: {output1, output2, ...} 
	   output: {valueOfOutput1, valueOfOutput2, ...} *)

    
    (* inputs: {input1, input2, ...} 
       input: {valueForInput1, valueForInput1, ...} *)
    inputs_/;MatrixQ[inputs, NumberQ],

    (* Weights *)
    weights_,
    
    (* Activation and scaling, see MlpOptionActivationAndScaling *)
    activationAndScaling_
    
	] :=
  
	Module[
    
		{
			hiddenOutput,
			hiddenWeights,
			numberOfHiddenLayers,
			i,
			outputWeights,
			internalInputs,
			trueUnitHidden,
			outputs
		},

		numberOfHiddenLayers = Length[weights] - 1; 
		hiddenWeights = 
			Table[
				weights[[i]],
				
				{i, numberOfHiddenLayers}
			];
		outputWeights = weights[[numberOfHiddenLayers + 1]];

		hiddenOutput = 
			Table[
				{},
				
				{i, numberOfHiddenLayers}
			];
    
		(* Add 'true unit' to inputs *)
		internalInputs = Map[Append[#, 1.0] &, inputs];
		Switch[activationAndScaling[[1, 1]],
			
			"Sigmoid",
			hiddenOutput[[1]] = SigmoidFunction[internalInputs.Transpose[hiddenWeights[[1]]]],
			
			"Tanh",
			hiddenOutput[[1]] = Tanh[internalInputs.Transpose[hiddenWeights[[1]]]]
		];

		If[numberOfHiddenLayers > 1,
			Do[
			    (* Add 'true unit' to last hidden *)
			    trueUnitHidden = Map[Append[#, 1.0] &, hiddenOutput[[i - 1]]];
				Switch[activationAndScaling[[1, i]],
					
					"Sigmoid",
					hiddenOutput[[i]] = SigmoidFunction[trueUnitHidden.Transpose[hiddenWeights[[i]]]],
					
					"Tanh",
					hiddenOutput[[i]] = Tanh[trueUnitHidden.Transpose[hiddenWeights[[i]]]]
				],
				
				{i, 2, numberOfHiddenLayers}
			]
		];
	    
	    (* Add 'true unit' to last hidden *)
		trueUnitHidden = Map[Append[#, 1.0] &, hiddenOutput[[numberOfHiddenLayers]]];
		Switch[activationAndScaling[[1, numberOfHiddenLayers + 1]],
			
			"Sigmoid",
			outputs = SigmoidFunction[trueUnitHidden.Transpose[outputWeights]],
			
			"Tanh",
			outputs = Tanh[trueUnitHidden.Transpose[outputWeights]]
		];
		
		Return[outputs];
    ];

GetNumberOfHiddenNeurons[

	(* Returns number of hidden neurons for specified mlpInfo.

	   Returns:
	   numberOfHiddenNeurons: {numberOfHidden1Neurons, numberOfHidden2Neurons, numberOfHidden3Neurons} *)

    
  	(* See "Frequently used data structures" *)
    mlpInfo_
    
	] :=
  
	Module[
    
		{},
		
		Return[
			GetMlpStructure[mlpInfo][[2]]
		]
	];

GetMlpInputInclusionClass[

	(* Analyzes relevance of input components by successive get-one-in for classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlpInputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mlpInfo}
	   trainingSetResult: {numberOfIncludedInputs, (best) classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, (best) classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlps,
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
		(* Mlp options *)   
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
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
				GetMlpInputInclusionCalculationPC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					MlpOptionMultipleMlps -> multipleMlps,
    				MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
	 				MlpOptionReportIteration -> 0,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
    				MlpOptionActivationAndScaling -> activationAndScaling,
    				MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
    				MlpOptionCostFunctionType -> costFunctionType,
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
				GetMlpInputInclusionCalculationSC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					MlpOptionMultipleMlps -> multipleMlps,
    				MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
	 				MlpOptionReportIteration -> 0,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
    				MlpOptionActivationAndScaling -> activationAndScaling,
    				MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
    				MlpOptionCostFunctionType -> costFunctionType,
    				DataTransformationOptionNormalizationType -> normalizationType,
    				UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			]
		]
	];

GetMlpInputInclusionRegress[

	(* Analyzes relevance of input components by successive get-one-in for regression.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlpInputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mlpInfo}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlps,
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
		(* Mlp options *)   
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
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
				GetMlpInputInclusionCalculationPC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					MlpOptionMultipleMlps -> multipleMlps,
    				MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
	 				MlpOptionReportIteration -> 0,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
    				MlpOptionActivationAndScaling -> activationAndScaling,
    				MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
    				MlpOptionCostFunctionType -> costFunctionType,
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
				GetMlpInputInclusionCalculationSC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					MlpOptionMultipleMlps -> multipleMlps,
    				MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
	 				MlpOptionReportIteration -> 0,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
    				MlpOptionActivationAndScaling -> activationAndScaling,
    				MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
    				MlpOptionCostFunctionType -> costFunctionType,
    				DataTransformationOptionNormalizationType -> normalizationType,
    				UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			]
		]
	];

GetMlpInputInclusionCalculationSC[

	(* Analyzes relevance of input components by successive get-one-in for regression and classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlpInputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mlpInfo}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlps,
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
			mlpInputComponentRelevanceList,
	        mlpInfo,
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
		(* Mlp options *)   
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
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
		mlpInputComponentRelevanceList = {};
    
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
					mlpInfo = 
						FitMlp[
							trainingSet,
							numberOfHiddenNeurons,
							MlpOptionMultipleMlps -> multipleMlps,
			    			MlpOptionOptimizationMethod -> optimizationMethod,
							MlpOptionInitialWeights -> initialWeights,
							MlpOptionInitialNetworks -> initialNetworks,
							MlpOptionWeightsValueLimit -> weightsValueLimit,
							MlpOptionMinimizationPrecision -> minimizationPrecision,
							MlpOptionMaximumIterations -> maximumNumberOfIterations,
							MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
				 			MlpOptionReportIteration -> 0,
							MlpOptionUnused11 -> unusedOptionParameter11,
							MlpOptionUnused12 -> unusedOptionParameter12,
							MlpOptionUnused13 -> unusedOptionParameter13,
							MlpOptionUnused21 -> unusedOptionParameter21,
							MlpOptionUnused22 -> unusedOptionParameter22,
							MlpOptionUnused23 -> unusedOptionParameter23,
			    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
			    			MlpOptionActivationAndScaling -> activationAndScaling,
			    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
			    			MlpOptionCostFunctionType -> costFunctionType,
			    			DataTransformationOptionNormalizationType -> normalizationType
						];
					If[Length[testSet] > 0,
            
						testSetRmse = CalculateMlpDataSetRmse[testSet, mlpInfo];
						AppendTo[rmseList,{testSetRmse, i}],
          
						trainingSetRmse = CalculateMlpDataSetRmse[trainingSet, mlpInfo];
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
			mlpInfo = 
				FitMlp[
					trainingSet,
					numberOfHiddenNeurons,
					MlpOptionMultipleMlps -> multipleMlps,
	    			MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
		 			MlpOptionReportIteration -> 0,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			MlpOptionActivationAndScaling -> activationAndScaling,
	    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
	    			MlpOptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMlpDataSetRmse[trainingSet, mlpInfo];
					currentTestSetRmse = CalculateMlpDataSetRmse[testSet, mlpInfo];
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
							mlpInfo
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMlpDataSetRmse[trainingSet, mlpInfo];
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
							mlpInfo
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[trainingSet, mlpInfo];
					currentTestSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[testSet, mlpInfo];
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
							mlpInfo
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[trainingSet, mlpInfo];
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
							mlpInfo
						}
				]
			];	

			AppendTo[mlpInputComponentRelevanceList, relevance];
			includedInputComponentList = currentIncludedInputComponentList,
			
			{k, Length[numberOfInclusionsPerStepList]}
		];
		
		Return[mlpInputComponentRelevanceList]
	];

GetMlpInputInclusionCalculationPC[

	(* Analyzes relevance of input components by successive get-one-in for regression and classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlpInputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mlpInfo}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlps,
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
			mlpInputComponentRelevanceList,
	        mlpInfo,
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
		(* Mlp options *)   
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
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
		mlpInputComponentRelevanceList = {};
    	
    	ParallelNeeds[{"CIP`Mlp`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[multipleMlps, optimizationMethod, initialWeights,
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
						
						mlpInfo = 
							FitMlp[
								trainingSet,
								numberOfHiddenNeurons,
								MlpOptionMultipleMlps -> multipleMlps,
				    			MlpOptionOptimizationMethod -> optimizationMethod,
								MlpOptionInitialWeights -> initialWeights,
								MlpOptionInitialNetworks -> initialNetworks,
								MlpOptionWeightsValueLimit -> weightsValueLimit,
								MlpOptionMinimizationPrecision -> minimizationPrecision,
								MlpOptionMaximumIterations -> maximumNumberOfIterations,
								MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
				 				MlpOptionReportIteration -> 0,
								MlpOptionUnused11 -> unusedOptionParameter11,
								MlpOptionUnused12 -> unusedOptionParameter12,
								MlpOptionUnused13 -> unusedOptionParameter13,
								MlpOptionUnused21 -> unusedOptionParameter21,
								MlpOptionUnused22 -> unusedOptionParameter22,
								MlpOptionUnused23 -> unusedOptionParameter23,
				    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
			    				MlpOptionActivationAndScaling -> activationAndScaling,
			    				MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
			    				MlpOptionCostFunctionType -> costFunctionType,
			    				DataTransformationOptionNormalizationType -> normalizationType
							];
						
						If[Length[testSet] > 0,
            
							testSetRmse = CalculateMlpDataSetRmse[testSet, mlpInfo];
							{testSetRmse, i},
          	
							trainingSetRmse = CalculateMlpDataSetRmse[trainingSet, mlpInfo];
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
			mlpInfo = 
				FitMlp[
					trainingSet,
					numberOfHiddenNeurons,
					MlpOptionMultipleMlps -> multipleMlps,
	    			MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
		 			MlpOptionReportIteration -> 0,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			MlpOptionActivationAndScaling -> activationAndScaling,
	    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
	    			MlpOptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionsParallelization -> "ParallelCalculation"
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMlpDataSetRmse[trainingSet, mlpInfo];
					currentTestSetRmse = CalculateMlpDataSetRmse[testSet, mlpInfo];
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
							mlpInfo
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMlpDataSetRmse[trainingSet, mlpInfo];
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
							mlpInfo
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[trainingSet, mlpInfo];
					currentTestSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[testSet, mlpInfo];
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
							mlpInfo
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[trainingSet, mlpInfo];
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
							mlpInfo
						}
				]
			];	

			AppendTo[mlpInputComponentRelevanceList, relevance];
			includedInputComponentList = currentIncludedInputComponentList,
			
			{k, Length[numberOfInclusionsPerStepList]}
		];
		
		Return[mlpInputComponentRelevanceList]
	];

GetMlpInputRelevanceClass[

	(* Analyzes relevance of input components by successive leave-one-out for classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlpInputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlpInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, (best) classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlps,
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
		(* Mlp options *)   
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
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
				GetMlpInputRelevanceCalculationPC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					MlpOptionMultipleMlps -> multipleMlps,
	    			MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
		 			MlpOptionReportIteration -> 0,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
	    			MlpOptionActivationAndScaling -> activationAndScaling,
	    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
	    			MlpOptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetMlpInputRelevanceCalculationSC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					MlpOptionMultipleMlps -> multipleMlps,
	    			MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
		 			MlpOptionReportIteration -> 0,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
	    			MlpOptionActivationAndScaling -> activationAndScaling,
	    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
	    			MlpOptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			]
		]
	];

GetMlpClassRelevantComponents[

	(* Returns most-to-least-relevance sorted components from mlpInputComponentRelevanceListForClassification.

	   Returns: Returns most-to-least-relevance sorted components *)


	(* mlpInputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlpInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) correct classification in percent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best correct classification in percent of test set} *)
	mlpInputComponentRelevanceListForClassification_,
	
	(* Number of components to return *)
	numberOfComponents_?IntegerQ
    
	] :=
  
	Module[
    
		{},

		Return[
			CIP`Graphics`GetClassRelevantComponents[mlpInputComponentRelevanceListForClassification, numberOfComponents]
		]
	];

GetMlpInputRelevanceRegress[

	(* Analyzes relevance of input components by successive leave-one-out for regression.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlpInputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlpInfo}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlps,
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
		(* Mlp options *)   
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
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
				GetMlpInputRelevanceCalculationPC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					MlpOptionMultipleMlps -> multipleMlps,
	    			MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
		 			MlpOptionReportIteration -> 0,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
	    			MlpOptionActivationAndScaling -> activationAndScaling,
	    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
	    			MlpOptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetMlpInputRelevanceCalculationSC[
					trainingAndTestSet,
					numberOfHiddenNeurons,
					isRegression,
					MlpOptionMultipleMlps -> multipleMlps,
	    			MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
		 			MlpOptionReportIteration -> 0,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
	    			MlpOptionActivationAndScaling -> activationAndScaling,
	    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
	    			MlpOptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			]
		]
	];

GetMlpRegressRelevantComponents[

	(* Returns most-to-least-relevance sorted components from mlpInputComponentRelevanceListForRegression.

	   Returns: Returns most-to-least-relevance sorted components *)


	(* mlpInputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlpInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)
	mlpInputComponentRelevanceListForRegression_,
	
	(* Number of components to return *)
	numberOfComponents_?IntegerQ
    
	] :=
  
	Module[
    
		{},

		Return[
			CIP`Graphics`GetRegressRelevantComponents[mlpInputComponentRelevanceListForRegression, numberOfComponents]
		]
	];

GetMlpInputRelevanceCalculationSC[

	(* Analyzes relevance of input components by successive leave-one-out for regression and classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlpInputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlpInfo}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlps,
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
			mlpInputComponentRelevanceList,
	        mlpInfo,
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
		(* Mlp options *)   
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
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
		mlpInputComponentRelevanceList = {};
    
		(* Result for no removal *)
		trainingSet = trainingAndTestSet[[1]];
		testSet = trainingAndTestSet[[2]];
		mlpInfo = 
			FitMlp[
				trainingSet,
				numberOfHiddenNeurons,
				MlpOptionMultipleMlps -> multipleMlps,
    			MlpOptionOptimizationMethod -> optimizationMethod,
				MlpOptionInitialWeights -> initialWeights,
				MlpOptionInitialNetworks -> initialNetworks,
				MlpOptionWeightsValueLimit -> weightsValueLimit,
				MlpOptionMinimizationPrecision -> minimizationPrecision,
				MlpOptionMaximumIterations -> maximumNumberOfIterations,
				MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
	 			MlpOptionReportIteration -> 0,
				MlpOptionUnused11 -> unusedOptionParameter11,
				MlpOptionUnused12 -> unusedOptionParameter12,
				MlpOptionUnused13 -> unusedOptionParameter13,
				MlpOptionUnused21 -> unusedOptionParameter21,
				MlpOptionUnused22 -> unusedOptionParameter22,
				MlpOptionUnused23 -> unusedOptionParameter23,
    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
    			MlpOptionActivationAndScaling -> activationAndScaling,
    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
    			MlpOptionCostFunctionType -> costFunctionType,
    			DataTransformationOptionNormalizationType -> normalizationType
			];
		
		initialTrainingSetRmse = CalculateMlpDataSetRmse[trainingSet, mlpInfo];
		If[isRegression,
			
			(* Regression*)
			If[Length[testSet] > 0,
				
				(* Regression WITH test set*)
				initialTestSetRmse = CalculateMlpDataSetRmse[testSet, mlpInfo];
				If[isIntermediateOutput,
					Print["initialTrainingSetRmse = ", initialTrainingSetRmse];
					Print["initialTestSetRmse     = ", initialTestSetRmse]
				];
				relevance = 
					{
						{0.0, initialTrainingSetRmse},
						{0.0, initialTestSetRmse},
						{}, 
						mlpInfo
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
						mlpInfo
					}
			],
			
			(* Classification *)
			initialTrainingSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[trainingSet, mlpInfo];
			If[Length[testSet] > 0,
				
				(* Classification WITH test set*)
				initialTestSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[testSet, mlpInfo];
				If[isIntermediateOutput,
					Print["initialTrainingSetCorrectClassificationInPercent = ", initialTrainingSetCorrectClassificationInPercent];
					Print["initialTestSetCorrectClassificationInPercent     = ", initialTestSetCorrectClassificationInPercent]
				];
				relevance = 
					{
						{0.0, initialTrainingSetCorrectClassificationInPercent},
						{0.0, initialTestSetCorrectClassificationInPercent},
						{}, 
						mlpInfo
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
						mlpInfo
					}
			]
		];	
		
		AppendTo[mlpInputComponentRelevanceList, relevance];
    
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
					mlpInfo = 
						FitMlp[
							trainingSet,
							numberOfHiddenNeurons,
							MlpOptionMultipleMlps -> multipleMlps,
			    			MlpOptionOptimizationMethod -> optimizationMethod,
							MlpOptionInitialWeights -> initialWeights,
							MlpOptionInitialNetworks -> initialNetworks,
							MlpOptionWeightsValueLimit -> weightsValueLimit,
							MlpOptionMinimizationPrecision -> minimizationPrecision,
							MlpOptionMaximumIterations -> maximumNumberOfIterations,
							MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
				 			MlpOptionReportIteration -> 0,
							MlpOptionUnused11 -> unusedOptionParameter11,
							MlpOptionUnused12 -> unusedOptionParameter12,
							MlpOptionUnused13 -> unusedOptionParameter13,
							MlpOptionUnused21 -> unusedOptionParameter21,
							MlpOptionUnused22 -> unusedOptionParameter22,
							MlpOptionUnused23 -> unusedOptionParameter23,
			    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
			    			MlpOptionActivationAndScaling -> activationAndScaling,
			    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
			    			MlpOptionCostFunctionType -> costFunctionType,
			    			DataTransformationOptionNormalizationType -> normalizationType
						];
					If[Length[testSet] > 0,
            
						testSetRmse = CalculateMlpDataSetRmse[testSet, mlpInfo];
						AppendTo[rmseList,{testSetRmse, i}],
          
						trainingSetRmse = CalculateMlpDataSetRmse[trainingSet, mlpInfo];
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
			mlpInfo = 
				FitMlp[
					trainingSet,
					numberOfHiddenNeurons,
					MlpOptionMultipleMlps -> multipleMlps,
	    			MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
		 			MlpOptionReportIteration -> 0,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			MlpOptionActivationAndScaling -> activationAndScaling,
	    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
	    			MlpOptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMlpDataSetRmse[trainingSet, mlpInfo];
					currentTestSetRmse = CalculateMlpDataSetRmse[testSet, mlpInfo];
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
							mlpInfo
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMlpDataSetRmse[trainingSet, mlpInfo];
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
							mlpInfo
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[trainingSet, mlpInfo];
					currentTestSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[testSet, mlpInfo];
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
							mlpInfo
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[trainingSet, mlpInfo];
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
							mlpInfo
						}
				]
			];	

			AppendTo[mlpInputComponentRelevanceList, relevance];
			removedInputComponentList = currentRemovedInputComponentList,
			
			{k, Length[numberOfExclusionsPerStepList]}
		];
		
		Return[mlpInputComponentRelevanceList]
	];

GetMlpInputRelevanceCalculationPC[

	(* Analyzes relevance of input components by successive leave-one-out for regression and classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlpInputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlpInfo}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			multipleMlps,
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
			mlpInputComponentRelevanceList,
	        mlpInfo,
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
		(* Mlp options *)   
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
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
		mlpInputComponentRelevanceList = {};
    
		(* Result for no removal *)
		trainingSet = trainingAndTestSet[[1]];
		testSet = trainingAndTestSet[[2]];
		mlpInfo = 
			FitMlp[
				trainingSet,
				numberOfHiddenNeurons,
				MlpOptionMultipleMlps -> multipleMlps,
    			MlpOptionOptimizationMethod -> optimizationMethod,
				MlpOptionInitialWeights -> initialWeights,
				MlpOptionInitialNetworks -> initialNetworks,
				MlpOptionWeightsValueLimit -> weightsValueLimit,
				MlpOptionMinimizationPrecision -> minimizationPrecision,
				MlpOptionMaximumIterations -> maximumNumberOfIterations,
				MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
	 			MlpOptionReportIteration -> 0,
				MlpOptionUnused11 -> unusedOptionParameter11,
				MlpOptionUnused12 -> unusedOptionParameter12,
				MlpOptionUnused13 -> unusedOptionParameter13,
				MlpOptionUnused21 -> unusedOptionParameter21,
				MlpOptionUnused22 -> unusedOptionParameter22,
				MlpOptionUnused23 -> unusedOptionParameter23,
    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
    			MlpOptionActivationAndScaling -> activationAndScaling,
    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
    			MlpOptionCostFunctionType -> costFunctionType,
    			DataTransformationOptionNormalizationType -> normalizationType,
    			UtilityOptionsParallelization -> "ParallelCalculation"
			];
			
		initialTrainingSetRmse = CalculateMlpDataSetRmse[trainingSet, mlpInfo];
		If[isRegression,
			
			(* Regression*)
			If[Length[testSet] > 0,
				
				(* Regression WITH test set*)
				initialTestSetRmse = CalculateMlpDataSetRmse[testSet, mlpInfo];
				If[isIntermediateOutput,
					Print["initialTrainingSetRmse = ", initialTrainingSetRmse];
					Print["initialTestSetRmse     = ", initialTestSetRmse]
				];
				relevance = 
					{
						{0.0, initialTrainingSetRmse},
						{0.0, initialTestSetRmse},
						{}, 
						mlpInfo
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
						mlpInfo
					}
			],
			
			(* Classification *)
			initialTrainingSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[trainingSet, mlpInfo];
			If[Length[testSet] > 0,
				
				(* Classification WITH test set*)
				initialTestSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[testSet, mlpInfo];
				If[isIntermediateOutput,
					Print["initialTrainingSetCorrectClassificationInPercent = ", initialTrainingSetCorrectClassificationInPercent];
					Print["initialTestSetCorrectClassificationInPercent     = ", initialTestSetCorrectClassificationInPercent]
				];
				relevance = 
					{
						{0.0, initialTrainingSetCorrectClassificationInPercent},
						{0.0, initialTestSetCorrectClassificationInPercent},
						{}, 
						mlpInfo
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
						mlpInfo
					}
			]
		];	
		
		AppendTo[mlpInputComponentRelevanceList, relevance];
    
		ParallelNeeds[{"CIP`Mlp`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[multipleMlps, optimizationMethod, initialWeights,
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
						
						mlpInfo = 
							FitMlp[
								trainingSet,
								numberOfHiddenNeurons,
								MlpOptionMultipleMlps -> multipleMlps,
				    			MlpOptionOptimizationMethod -> optimizationMethod,
								MlpOptionInitialWeights -> initialWeights,
								MlpOptionInitialNetworks -> initialNetworks,
								MlpOptionWeightsValueLimit -> weightsValueLimit,
								MlpOptionMinimizationPrecision -> minimizationPrecision,
								MlpOptionMaximumIterations -> maximumNumberOfIterations,
								MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
					 			MlpOptionReportIteration -> 0,
								MlpOptionUnused11 -> unusedOptionParameter11,
								MlpOptionUnused12 -> unusedOptionParameter12,
								MlpOptionUnused13 -> unusedOptionParameter13,
								MlpOptionUnused21 -> unusedOptionParameter21,
								MlpOptionUnused22 -> unusedOptionParameter22,
								MlpOptionUnused23 -> unusedOptionParameter23,
				    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
				    			MlpOptionActivationAndScaling -> activationAndScaling,
				    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
				    			MlpOptionCostFunctionType -> costFunctionType,
				    			DataTransformationOptionNormalizationType -> normalizationType
							];
								
						If[Length[testSet] > 0,
	            
							testSetRmse = CalculateMlpDataSetRmse[testSet, mlpInfo];
							{testSetRmse, i},
	          
							trainingSetRmse = CalculateMlpDataSetRmse[trainingSet, mlpInfo];
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
			mlpInfo = 
				FitMlp[
					trainingSet,
					numberOfHiddenNeurons,
					MlpOptionMultipleMlps -> multipleMlps,
	    			MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
		 			MlpOptionReportIteration -> 0,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			MlpOptionActivationAndScaling -> activationAndScaling,
	    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
	    			MlpOptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionsParallelization -> "ParallelCalculation"
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMlpDataSetRmse[trainingSet, mlpInfo];
					currentTestSetRmse = CalculateMlpDataSetRmse[testSet, mlpInfo];
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
							mlpInfo
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMlpDataSetRmse[trainingSet, mlpInfo];
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
							mlpInfo
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[trainingSet, mlpInfo];
					currentTestSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[testSet, mlpInfo];
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
							mlpInfo
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlpCorrectClassificationInPercent[trainingSet, mlpInfo];
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
							mlpInfo
						}
				]
			];	

			AppendTo[mlpInputComponentRelevanceList, relevance];
			removedInputComponentList = currentRemovedInputComponentList,
			
			{k, Length[numberOfExclusionsPerStepList]}
		];
		
		Return[mlpInputComponentRelevanceList]
	];
	
GetMlpRegressionResult[
	
	(* Returns mlp regression result according to named property list.

	   Returns :
	   Mlp regression result according to named property *)

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
    mlpInfo_,
	
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
	    
		pureFunction = Function[inputs, CalculateMlpOutputs[inputs, mlpInfo]];
	    Return[
	    	CIP`Graphics`GetSingleRegressionResult[
		    	namedProperty, 
		    	dataSet, 
		    	pureFunction,
		    	GraphicsOptionNumberOfIntervals -> numberOfIntervals
			]
		]
	];

GetMlpSeriesClassificationResult[

	(* Shows result of Mlp series classifications for training and test set.

	   Returns: 
	   mlpSeriesClassificationResult: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in mlpInfoList, classification result in percent for training set}
	   testPoint[[i]]: {index i in mlpInfoList, classification result in percent for test set} *)


    (* {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...}
	   testSet has the same structure and restrictions as trainingSet 
	   NOTE: Training and test set MUST be in original units *)
    trainingAndTestSet_,

	(* {mlpInfo1, mlpInfo2, ...}
	   mlpInfo (see "Frequently used data structures") *)
    mlpInfoList_
    
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
			pureFunction = Function[inputs, CalculateMlpClassNumbers[inputs, mlpInfoList[[i]]]];
			correctClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[trainingSet, pureFunction];
			AppendTo[trainingPoints2D, {N[i], correctClassificationInPercent}];
			If[Length[testSet] > 0,
				correctClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[testSet, pureFunction];
				AppendTo[testPoints2D, {N[i], correctClassificationInPercent}]
			],
			
			{i, Length[mlpInfoList]}
		];
		
		Return[{trainingPoints2D, testPoints2D}]
	];

GetMlpSeriesRmse[

	(* Shows RMSE of Mlp series for training and test set.

	   Returns: 
	   mlpSeriesRmse: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in mlpInfoList, RMSE for training set}
	   testPoint[[i]]: {index i in mlpInfoList, RMSE for test set} *)


    (* {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...}
	   testSet has the same structure and restrictions as trainingSet 
	   NOTE: Training and test set MUST be in original units *)
    trainingAndTestSet_,

	(* {mlpInfo1, mlpInfo2, ...}
	   mlpInfo (see "Frequently used data structures") *)
    mlpInfoList_
    
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
			pureFunction = Function[inputs, CalculateMlpOutputs[inputs, mlpInfoList[[i]]]];
			rmse = Sqrt[CIP`Utility`GetMeanSquaredError[trainingSet, pureFunction]];
			AppendTo[trainingPoints2D, {N[i], rmse}];
			If[Length[testSet] > 0,
				rmse = Sqrt[CIP`Utility`GetMeanSquaredError[testSet, pureFunction]];
				AppendTo[testPoints2D, {N[i], rmse}]
			],
			
			{i, Length[mlpInfoList]}
		];
		
		Return[{trainingPoints2D, testPoints2D}]
	];

GetMlpStructure[

	(* Returns mlp structure for specified mlpInfo.

	   Returns:
	   {numberOfInputs, numberOfHiddenNeurons, numberOfOutputs} 
	   
	   numberOfHiddenNeurons: {numberOfHidden1Neurons, numberOfHidden2Neurons, ..., numberOfHidden3Neurons} *)

    
  	(* See "Frequently used data structures" *)
    mlpInfo_
    
	] :=
  
	Module[
    
		{
			i,
			numberOfHiddenLayers,
			numberOfHiddenNeurons,
			numberOfInputs,
			numberOfOutputs,
			networks,
			weights
		},
    
    	networks = mlpInfo[[1]];
    	
		If[Length[networks] == 1,
	
			(* --------------------------------------------------------------------------------
			   One network
			   -------------------------------------------------------------------------------- *)		

	    	weights = networks[[1]];
	    	numberOfInputs = Length[weights[[1, 1]]] - 1;
			numberOfHiddenLayers = Length[weights] - 1; 
	    	numberOfHiddenNeurons = 
	    		Table[
	    			Length[weights[[i]]],
	    			
	    			{i, numberOfHiddenLayers}
	    		];
	    	numberOfOutputs = Length[weights[[numberOfHiddenLayers + 1]]];
	    	Return[{numberOfInputs, numberOfHiddenNeurons, numberOfOutputs}],
			
			(* --------------------------------------------------------------------------------
			   Multiple networks (with ONE output value each)
			   -------------------------------------------------------------------------------- *)		

	    	weights = networks[[1]];
	    	numberOfInputs = Length[weights[[1, 1]]] - 1;
			numberOfHiddenLayers = Length[weights] - 1; 
	    	numberOfHiddenNeurons = 
	    		Table[
	    			Length[weights[[i]]],
	    			
	    			{i, numberOfHiddenLayers}
	    		];
	    	numberOfOutputs = Length[networks];;
	    	Return[{numberOfInputs, numberOfHiddenNeurons, numberOfOutputs}]
		]
	];

GetMlpTrainOptimization[

	(* Returns training set optimization result for mlp training.

	   Returns:
	   mlpTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlpInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlpInfoList: List with mlpInfo
	   mlpInfoList[[i]] refers to optimization step i *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],

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
			multipleMlps,
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
			mlpInfo,
			trainingSetRMSE,
			testSetRMSE,
			pureOutputFunction,
			trainingSetRmseList,
			testSetRmseList,
			trainingAndTestSetList,
			mlpInfoList,
			selectionResult,
			blackList,
			parallelization,
			lambdaL2Regularization,
			costFunctionType
		},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp options *)
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
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
				MlpOptionActivationAndScaling -> activationAndScaling
			];
		trainingSetIndexList = clusterRepresentativesRelatedIndexLists[[1]];
		testSetIndexList = clusterRepresentativesRelatedIndexLists[[2]];
		indexLists = clusterRepresentativesRelatedIndexLists[[3]];

		trainingSetRmseList = {};
		testSetRmseList = {};
		trainingAndTestSetList = {};
		mlpInfoList = {};
		blackList = {};
		Do[
			(* Fit training set and evaluate RMSE *)
			trainingSet = CIP`DataTransformation`GetDataSetPart[dataSet, trainingSetIndexList];
			testSet = CIP`DataTransformation`GetDataSetPart[dataSet, testSetIndexList];
			
			mlpInfo = 
				FitMlp[
					trainingSet,
					numberOfHiddenNeurons,
					MlpOptionMultipleMlps -> multipleMlps,
	    			MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
		 			MlpOptionReportIteration -> 0,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
	    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
	    			MlpOptionActivationAndScaling -> activationAndScaling,
	    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
	    			MlpOptionCostFunctionType -> costFunctionType,
	    			DataTransformationOptionNormalizationType -> normalizationType,
	    			UtilityOptionsParallelization -> parallelization
				];
				
			trainingSetRMSE = CalculateMlpDataSetRmse[trainingSet, mlpInfo];
			testSetRMSE = CalculateMlpDataSetRmse[testSet, mlpInfo];

			(* Set iteration results *)
			AppendTo[trainingSetRmseList, {N[i], trainingSetRMSE}];
			AppendTo[testSetRmseList, {N[i], testSetRMSE}];
			AppendTo[trainingAndTestSetList, {trainingSet, testSet}];
			AppendTo[mlpInfoList, mlpInfo];
			
			(* Break if necessary *)
			If[i == numberOfTrainingSetOptimizationSteps,
				Break[]
			];

			(* Select new training and test set index lists *)
			pureOutputFunction = Function[input, CalculateMlpOutput[input, mlpInfo]];
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
				mlpInfoList
			}
		]
	];

GetMlpWeights[

	(* Returns weights of specified network of mlpInfo.

	   Returns:
	   weights: {hidden1Weights, hidden2Weights, outputWeights}
	   hidden1Weights: Weights from input to hidden units
	   hidden2Weights: Weights from hidden1 to hidden2 units
	   hidden3Weights: Weights from hidden2 to hidden3 units
	   outputWeights : Weights from hidden3 to output units *)

    
  	(* See "Frequently used data structures" *)
    mlpInfo_,
    
	(* Index of network in mlpInfo *)
    indexOfNetwork_?IntegerQ
    
	] :=
  
	Module[
    
		{
			networks
		},
    
    	networks = mlpInfo[[1]];
   		Return[networks[[indexOfNetwork]]]
	];

GetWeightsStartVariables[

	(* Returns weights start variables, see code. *)

    
	(* Number of hidden units *)
    numberOfInputs_?IntegerQ,

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],

	(* Number of hidden units *)
    numberOfOutputs_?IntegerQ,

    (* Variable for hidden1 weights *)
    wInputToHidden1_,

    (* Variable for hidden2 and higher weights *)
    wHiddenToHidden_,
    
    (* Variable for output weights *)
    wHiddenToOutput_,

    (* Weights *)
    weights_

	] :=
  
	Module[
    
		{
			factor,
			hiddenWeigthsStartVariables,
			numberOfHiddenLayers,
			hiddenWeights,
			outputWeights,
			i,
			j,
			k,
			outputWeightsStartVariables
		},

		numberOfHiddenLayers = Length[weights] - 1; 
		hiddenWeights = 
			Table[
				weights[[i]],
				
				{i, numberOfHiddenLayers}
			];
		outputWeights = weights[[numberOfHiddenLayers + 1]];

		hiddenWeigthsStartVariables = 
			Table[
				{},
				
				{i, numberOfHiddenLayers}
			];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfInputs + 1];
		Do[
	    	Do[
				AppendTo[hiddenWeigthsStartVariables[[1]], {Subscript[wInputToHidden1, j*factor + i], hiddenWeights[[1, j, i]]}],
	    		
	    		{i, numberOfInputs + 1}
	    	],
	    
	    	{j, numberOfHiddenNeurons[[1]]}	
		];

		If[numberOfHiddenLayers > 1,
			Do[
				factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons[[k - 1]] + 1];
				Do[
			    	Do[
						AppendTo[hiddenWeigthsStartVariables[[k]], {Subscript[wHiddenToHidden, k, j * factor + i], hiddenWeights[[k, j, i]]}],
			    		
			    		{i, numberOfHiddenNeurons[[k - 1]] + 1}
			    	],
			    
			    	{j, numberOfHiddenNeurons[[k]]}	
				],
				
				{k, 2, numberOfHiddenLayers}
			]
		];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons[[numberOfHiddenLayers]] + 1];
		outputWeightsStartVariables = {};
		Do[	
			Do[
				AppendTo[outputWeightsStartVariables, {Subscript[wHiddenToOutput, j * factor + i], outputWeights[[j, i]]}],
		    
		    	{i, numberOfHiddenNeurons[[numberOfHiddenLayers]] + 1}	
			],
				
			{j, numberOfOutputs}
		];
		
		Return[Join[Flatten[hiddenWeigthsStartVariables, 1], outputWeightsStartVariables]]
	];

GetWeightsVariablesWithoutTrueUnitBias[

	(* Returns weights variables without true unit bias weights, see code. *)

    
	(* Number of hidden units *)
    numberOfInputs_?IntegerQ,

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],

	(* Number of hidden units *)
    numberOfOutputs_?IntegerQ,

    (* Variable for hidden weights *)
    wInputToHidden1_,

    (* Variable for hidden2 and higher weights *)
    wHiddenToHidden_,
    
    (* Variable for output weights *)
    wHiddenToOutput_
    
	] :=
  
	Module[
    
		{
			factor,
			numberOfHiddenLayers,
			weigthVariables,
			i,
			j,
			k
		},

		(* NO true unit bias: Do NOT add 1 to numberOfInputs or numberOfHiddenNeurons *)

		weigthVariables = {};
		
		numberOfHiddenLayers = Length[numberOfHiddenNeurons];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfInputs + 1];
		Do[
	    	Do[
				AppendTo[weigthVariables, Subscript[wInputToHidden1, j * factor + i]],
	    		
	    		{i, numberOfInputs}
	    	],
	    
	    	{j, numberOfHiddenNeurons[[1]]}	
		];

		If[numberOfHiddenLayers > 1,
			Do[
				factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons[[k - 1]] + 1];
				Do[
			    	Do[
						AppendTo[weigthVariables, Subscript[wHiddenToHidden, k, j * factor + i]],
			    		
			    		{i,  numberOfHiddenNeurons[[k - 1]]}
			    	],
			    
			    	{j,  numberOfHiddenNeurons[[k]]}	
				],
				
				{k, 2, numberOfHiddenLayers}
			]
		];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons[[numberOfHiddenLayers]] + 1];
		Do[	
			Do[
				AppendTo[weigthVariables, Subscript[wHiddenToOutput, j * factor + i]],
		    
		    	{i, numberOfHiddenNeurons[[numberOfHiddenLayers]]}	
			],
				
			{j, numberOfOutputs}
		];
		
		Return[weigthVariables]
	];

GetWeightsVariables[

	(* Returns weights variables, see code. *)

    
	(* Number of hidden units *)
    numberOfInputs_?IntegerQ,

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],

	(* Number of hidden units *)
    numberOfOutputs_?IntegerQ,

    (* Variable for hidden1 weights *)
    wInputToHidden1_,

    (* Variable for hidden2 and higher weights *)
    wHiddenToHidden_,
    
    (* Variable for output weights *)
    wHiddenToOutput_

	] :=
  
	Module[
    
		{
			factor,
			hiddenWeigthsVariables,
			numberOfHiddenLayers,
			i,
			j,
			k,
			outputWeightsVariables
		},

		numberOfHiddenLayers = Length[numberOfHiddenNeurons]; 

		hiddenWeigthsVariables = 
			Table[
				{},
				
				{i, numberOfHiddenLayers}
			];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfInputs + 1];
		hiddenWeigthsVariables[[1]] = 
			Table[
				Table[
					Subscript[wInputToHidden1, j * factor + i], 
					
					{i, numberOfInputs + 1}
				], 
				
				{j, numberOfHiddenNeurons[[1]]}
			];

		If[numberOfHiddenLayers > 1,
			Do[
				factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons[[k - 1]] + 1];
				hiddenWeigthsVariables[[k]] = 
					Table[
						Table[
							Subscript[wHiddenToHidden, k, j * factor + i], 
							
							{i, numberOfHiddenNeurons[[k - 1]] + 1}
						], 
						
						{j, numberOfHiddenNeurons[[k]]}
					],
				
				{k, 2, numberOfHiddenLayers}
			]
		];

		factor = CIP`Utility`GetNextHigherMultipleOfTen[numberOfHiddenNeurons[[numberOfHiddenLayers]] + 1];
		outputWeightsVariables = 
			Table[
				Table[
					Subscript[wHiddenToOutput, j * factor + i], 
					
					{i, numberOfHiddenNeurons[[numberOfHiddenLayers]] + 1}
				], 
				
				{j, numberOfOutputs}
			];
			
		Return[AppendTo[hiddenWeigthsVariables, outputWeightsVariables]]		
	];

ScanClassTrainingWithMlp[

	(* Scans training and test set for different training fractions based on method FitMlp, see code.
	
	   Returns:
	   mlpClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlpInfo1}, {trainingAndTestSet2, mlpInfo2}, ...}
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

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],

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
			multipleMlps,
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
		(* Mlp options *)
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
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
				ScanClassTrainingWithMlpPC[
					classificationDataSet,
					numberOfHiddenNeurons,
					trainingFractionList,
					MlpOptionMultipleMlps -> multipleMlps,
				    MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
					ClusterOptionMethod -> clusterMethod,
				    ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				    ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				    ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
			   	    MlpOptionActivationAndScaling -> activationAndScaling,
			   	    MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
			   	    MlpOptionCostFunctionType -> costFunctionType,
			   	    DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
				    UtilityOptionBlackListLength -> blackListLength
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				ScanClassTrainingWithMlpSC[
					classificationDataSet,
					numberOfHiddenNeurons,
					trainingFractionList,
					MlpOptionMultipleMlps -> multipleMlps,
				    MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
					ClusterOptionMethod -> clusterMethod,
				    ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				    ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				    ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
			   	    MlpOptionActivationAndScaling -> activationAndScaling,
			   	    MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
			   	    MlpOptionCostFunctionType -> costFunctionType,
			   	    DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
				    UtilityOptionBlackListLength -> blackListLength
				]
			]
		]
	];

ScanClassTrainingWithMlpSC[

	(* Scans training and test set for different training fractions based on method FitMlp, see code.
	
	   Returns:
	   mlpClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlpInfo1}, {trainingAndTestSet2, mlpInfo2}, ...}
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

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],

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
			multipleMlps,
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
			currentMlpInfo,
			pureFunction,
			trainingSetCorrectClassificationInPercent,
			testSetCorrectClassificationInPercent,
			mlpTrainOptimization,
			mlpInfoList,
			trainingAndTestSetList,
			bestIndex,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp options *)
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
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
				mlpTrainOptimization = 
					GetMlpTrainOptimization[
						classificationDataSet, 
						numberOfHiddenNeurons, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						MlpOptionMultipleMlps -> multipleMlps,
		    			MlpOptionOptimizationMethod -> optimizationMethod,
						MlpOptionInitialWeights -> initialWeights,
						MlpOptionInitialNetworks -> initialNetworks,
						MlpOptionWeightsValueLimit -> weightsValueLimit,
						MlpOptionMinimizationPrecision -> minimizationPrecision,
						MlpOptionMaximumIterations -> maximumNumberOfIterations,
						MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
						MlpOptionUnused11 -> unusedOptionParameter11,
						MlpOptionUnused12 -> unusedOptionParameter12,
						MlpOptionUnused13 -> unusedOptionParameter13,
						MlpOptionUnused21 -> unusedOptionParameter21,
						MlpOptionUnused22 -> unusedOptionParameter22,
						MlpOptionUnused23 -> unusedOptionParameter23,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						MlpOptionActivationAndScaling -> activationAndScaling,
						MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
						MlpOptionCostFunctionType -> costFunctionType,
						DataTransformationOptionNormalizationType -> normalizationType,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestMlpClassOptimization[mlpTrainOptimization];
				trainingAndTestSetList = mlpTrainOptimization[[3]];
				mlpInfoList = mlpTrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlpInfo = mlpInfoList[[bestIndex]],
				
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
				currentMlpInfo = 
					FitMlp[
						currentTrainingSet,
						numberOfHiddenNeurons,
						MlpOptionMultipleMlps -> multipleMlps,
		    			MlpOptionOptimizationMethod -> optimizationMethod,
						MlpOptionInitialWeights -> initialWeights,
						MlpOptionInitialNetworks -> initialNetworks,
						MlpOptionWeightsValueLimit -> weightsValueLimit,
						MlpOptionMinimizationPrecision -> minimizationPrecision,
						MlpOptionMaximumIterations -> maximumNumberOfIterations,
						MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
			 			MlpOptionReportIteration -> 0,
						MlpOptionUnused11 -> unusedOptionParameter11,
						MlpOptionUnused12 -> unusedOptionParameter12,
						MlpOptionUnused13 -> unusedOptionParameter13,
						MlpOptionUnused21 -> unusedOptionParameter21,
						MlpOptionUnused22 -> unusedOptionParameter22,
						MlpOptionUnused23 -> unusedOptionParameter23,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			MlpOptionActivationAndScaling -> activationAndScaling,
		    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
		    			MlpOptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					]
			];
			
			pureFunction = Function[inputs, CalculateMlpClassNumbers[inputs, currentMlpInfo]];
			trainingSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTrainingSet, pureFunction];
			testSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTestSet, pureFunction];
			AppendTo[trainingAndTestSetsInfo, {currentTrainingAndTestSet, currentMlpInfo}];
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

ScanClassTrainingWithMlpPC[

	(* Scans training and test set for different training fractions based on method FitMlp, see code.
	
	   Returns:
	   mlpClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlpInfo1}, {trainingAndTestSet2, mlpInfo2}, ...}
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

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],

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
			multipleMlps,
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
			currentMlpInfo,
			pureFunction,
			trainingSetCorrectClassificationInPercent,
			testSetCorrectClassificationInPercent,
			mlpTrainOptimization,
			mlpInfoList,
			trainingAndTestSetList,
			bestIndex,
			listOfTrainingAndTestSetsInfoAndScanReport,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp options *)
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* Training set optimization options *)
		numberOfTrainingSetOptimizationSteps = UtilityOptionOptimizationSteps/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];

		ParallelNeeds[{"CIP`Mlp`", "CIP`Cluster`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[numberOfTrainingSetOptimizationSteps, multipleMlps, optimizationMethod, initialWeights,
						initialNetworks, weightsValueLimit, minimizationPrecision, maximumNumberOfIterations, numberOfIterationsToImprove, 
						unusedOptionParameter11, unusedOptionParameter12, unusedOptionParameter13, unusedOptionParameter21, unusedOptionParameter22, unusedOptionParameter23,
						clusterMethod, maximumNumberOfEpochs, scalarProductMinimumTreshold, maximumNumberOfTrialSteps, activationAndScaling, 
						normalizationType, randomValueInitialization, deviationCalculationMethod, blackListLength, lambdaL2Regularization, 
						costFunctionType];
		
		listOfTrainingAndTestSetsInfoAndScanReport = ParallelTable[
			If[numberOfTrainingSetOptimizationSteps > 0,
				
				(* ------------------------------------------------------------------------------------------------------ *)
				(* Training set optimization *)
				
				mlpTrainOptimization = 
					GetMlpTrainOptimization[
						classificationDataSet, 
						numberOfHiddenNeurons, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						MlpOptionMultipleMlps -> multipleMlps,
		    			MlpOptionOptimizationMethod -> optimizationMethod,
						MlpOptionInitialWeights -> initialWeights,
						MlpOptionInitialNetworks -> initialNetworks,
						MlpOptionWeightsValueLimit -> weightsValueLimit,
						MlpOptionMinimizationPrecision -> minimizationPrecision,
						MlpOptionMaximumIterations -> maximumNumberOfIterations,
						MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
						MlpOptionUnused11 -> unusedOptionParameter11,
						MlpOptionUnused12 -> unusedOptionParameter12,
						MlpOptionUnused13 -> unusedOptionParameter13,
						MlpOptionUnused21 -> unusedOptionParameter21,
						MlpOptionUnused22 -> unusedOptionParameter22,
						MlpOptionUnused23 -> unusedOptionParameter23,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						MlpOptionActivationAndScaling -> activationAndScaling,
						MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
						MlpOptionCostFunctionType -> costFunctionType,
						DataTransformationOptionNormalizationType -> normalizationType,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestMlpClassOptimization[mlpTrainOptimization];				
				trainingAndTestSetList = mlpTrainOptimization[[3]];
				mlpInfoList = mlpTrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlpInfo = mlpInfoList[[bestIndex]],
				
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
				
				currentMlpInfo = 
					FitMlp[
						currentTrainingSet,
						numberOfHiddenNeurons,
						MlpOptionMultipleMlps -> multipleMlps,
		    			MlpOptionOptimizationMethod -> optimizationMethod,
						MlpOptionInitialWeights -> initialWeights,
						MlpOptionInitialNetworks -> initialNetworks,
						MlpOptionWeightsValueLimit -> weightsValueLimit,
						MlpOptionMinimizationPrecision -> minimizationPrecision,
						MlpOptionMaximumIterations -> maximumNumberOfIterations,
						MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
			 			MlpOptionReportIteration -> 0,
						MlpOptionUnused11 -> unusedOptionParameter11,
						MlpOptionUnused12 -> unusedOptionParameter12,
						MlpOptionUnused13 -> unusedOptionParameter13,
						MlpOptionUnused21 -> unusedOptionParameter21,
						MlpOptionUnused22 -> unusedOptionParameter22,
						MlpOptionUnused23 -> unusedOptionParameter23,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			MlpOptionActivationAndScaling -> activationAndScaling,
		    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
		    			MlpOptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					];
			];
			
			pureFunction = Function[inputs, CalculateMlpClassNumbers[inputs, currentMlpInfo]];
			trainingSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTrainingSet, pureFunction];
			testSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTestSet, pureFunction];
			{
				{currentTrainingAndTestSet, currentMlpInfo},
				
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

ScanRegressTrainingWithMlp[

	(* Scans training and test set for different training fractions based on method FitMlp, see code.
	
	   Returns:
	   mlpRegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlpInfo1}, {trainingAndTestSet2, mlpInfo2}, ...}
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

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],

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
			multipleMlps,
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
		(* Mlp options *)
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
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
				ScanRegressTrainingWithMlpPC[
					dataSet,
					numberOfHiddenNeurons,
					trainingFractionList,
					MlpOptionMultipleMlps -> multipleMlps,
				    MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
					ClusterOptionMethod -> clusterMethod,
				    ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				    ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				    ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
			   	    MlpOptionActivationAndScaling -> activationAndScaling,
			   	    MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
			   	    MlpOptionCostFunctionType -> costFunctionType,
			   	    DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
				    UtilityOptionBlackListLength -> blackListLength
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				ScanRegressTrainingWithMlpSC[
					dataSet,
					numberOfHiddenNeurons,
					trainingFractionList,
					MlpOptionMultipleMlps -> multipleMlps,
				    MlpOptionOptimizationMethod -> optimizationMethod,
					MlpOptionInitialWeights -> initialWeights,
					MlpOptionInitialNetworks -> initialNetworks,
					MlpOptionWeightsValueLimit -> weightsValueLimit,
					MlpOptionMinimizationPrecision -> minimizationPrecision,
					MlpOptionMaximumIterations -> maximumNumberOfIterations,
					MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
					MlpOptionUnused11 -> unusedOptionParameter11,
					MlpOptionUnused12 -> unusedOptionParameter12,
					MlpOptionUnused13 -> unusedOptionParameter13,
					MlpOptionUnused21 -> unusedOptionParameter21,
					MlpOptionUnused22 -> unusedOptionParameter22,
					MlpOptionUnused23 -> unusedOptionParameter23,
					ClusterOptionMethod -> clusterMethod,
				    ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				    ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				    ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
			   	    MlpOptionActivationAndScaling -> activationAndScaling,
			   	    MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
			   	    MlpOptionCostFunctionType -> costFunctionType,
			   	    DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
				    UtilityOptionBlackListLength -> blackListLength
				]
			]
		]
	];

ScanRegressTrainingWithMlpSC[

	(* Scans training and test set for different training fractions based on method FitMlp, see code.
	
	   Returns:
	   mlpRegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlpInfo1}, {trainingAndTestSet2, mlpInfo2}, ...}
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

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],

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
			multipleMlps,
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
			currentMlpInfo,
			pureFunction,
			trainingSetRMSE,
			testSetRMSE,
			mlpTrainOptimization,
			trainingAndTestSetList,
			mlpInfoList,
			bestIndex,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp options *)
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
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
				mlpTrainOptimization = 
					GetMlpTrainOptimization[
						dataSet, 
						numberOfHiddenNeurons, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						MlpOptionMultipleMlps -> multipleMlps,
		    			MlpOptionOptimizationMethod -> optimizationMethod,
						MlpOptionInitialWeights -> initialWeights,
						MlpOptionInitialNetworks -> initialNetworks,
						MlpOptionWeightsValueLimit -> weightsValueLimit,
						MlpOptionMinimizationPrecision -> minimizationPrecision,
						MlpOptionMaximumIterations -> maximumNumberOfIterations,
						MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
						MlpOptionUnused11 -> unusedOptionParameter11,
						MlpOptionUnused12 -> unusedOptionParameter12,
						MlpOptionUnused13 -> unusedOptionParameter13,
						MlpOptionUnused21 -> unusedOptionParameter21,
						MlpOptionUnused22 -> unusedOptionParameter22,
						MlpOptionUnused23 -> unusedOptionParameter23,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						MlpOptionActivationAndScaling -> activationAndScaling,
						MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
						MlpOptionCostFunctionType -> costFunctionType,
						DataTransformationOptionNormalizationType -> normalizationType,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestMlpRegressOptimization[mlpTrainOptimization];
				trainingAndTestSetList = mlpTrainOptimization[[3]];
				mlpInfoList = mlpTrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlpInfo = mlpInfoList[[bestIndex]],
				
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
				currentMlpInfo = 
					FitMlp[
						currentTrainingSet,
						numberOfHiddenNeurons,
						MlpOptionMultipleMlps -> multipleMlps,
		    			MlpOptionOptimizationMethod -> optimizationMethod,
						MlpOptionInitialWeights -> initialWeights,
						MlpOptionInitialNetworks -> initialNetworks,
						MlpOptionWeightsValueLimit -> weightsValueLimit,
						MlpOptionMinimizationPrecision -> minimizationPrecision,
						MlpOptionMaximumIterations -> maximumNumberOfIterations,
						MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
			 			MlpOptionReportIteration -> 0,
						MlpOptionUnused11 -> unusedOptionParameter11,
						MlpOptionUnused12 -> unusedOptionParameter12,
						MlpOptionUnused13 -> unusedOptionParameter13,
						MlpOptionUnused21 -> unusedOptionParameter21,
						MlpOptionUnused22 -> unusedOptionParameter22,
						MlpOptionUnused23 -> unusedOptionParameter23,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			MlpOptionActivationAndScaling -> activationAndScaling,
		    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
		    			MlpOptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					]
			];
			
			pureFunction = Function[inputs, CalculateMlpOutputs[inputs, currentMlpInfo]];
			trainingSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTrainingSet, pureFunction]];
			testSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTestSet, pureFunction]];
			AppendTo[trainingAndTestSetsInfo, {currentTrainingAndTestSet, currentMlpInfo}];
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

ScanRegressTrainingWithMlpPC[

	(* Scans training and test set for different training fractions based on method FitMlp, see code.
	
	   Returns:
	   mlpRegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlpInfo1}, {trainingAndTestSet2, mlpInfo2}, ...}
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

	(* Number of hidden neurons: {<number of neurons in hidden1>, <... in hidden2>, <... in hidden3>, ...} *)
	numberOfHiddenNeurons_/;VectorQ[numberOfHiddenNeurons, IntegerQ],

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
			multipleMlps,
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
			currentMlpInfo,
			pureFunction,
			trainingSetRMSE,
			testSetRMSE,
			mlpTrainOptimization,
			trainingAndTestSetList,
			mlpInfoList,
			bestIndex,
			listOfTrainingAndTestSetsInfoAndScanReport,
			lambdaL2Regularization,
			costFunctionType
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Mlp options *)
		multipleMlps = MlpOptionMultipleMlps/.{opts}/.Options[MlpOptionsTraining];
	    optimizationMethod = MlpOptionOptimizationMethod/.{opts}/.Options[MlpOptionsTraining];
		initialWeights = MlpOptionInitialWeights/.{opts}/.Options[MlpOptionsOptimization];
		initialNetworks = MlpOptionInitialNetworks/.{opts}/.Options[MlpOptionsOptimization];
		weightsValueLimit = MlpOptionWeightsValueLimit/.{opts}/.Options[MlpOptionsOptimization];
		minimizationPrecision = MlpOptionMinimizationPrecision/.{opts}/.Options[MlpOptionsOptimization];
		maximumNumberOfIterations = MlpOptionMaximumIterations/.{opts}/.Options[MlpOptionsOptimization];
		numberOfIterationsToImprove = MlpOptionIterationsToImprove/.{opts}/.Options[MlpOptionsOptimization];
		unusedOptionParameter11 = MlpOptionUnused11/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter12 = MlpOptionUnused12/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter13 = MlpOptionUnused13/.{opts}/.Options[MlpOptionsUnused1];
		unusedOptionParameter21 = MlpOptionUnused21/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter22 = MlpOptionUnused22/.{opts}/.Options[MlpOptionsUnused2];
		unusedOptionParameter23 = MlpOptionUnused23/.{opts}/.Options[MlpOptionsUnused2];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    activationAndScaling = MlpOptionActivationAndScaling/.{opts}/.Options[MlpOptionsTraining];
	    lambdaL2Regularization = MlpOptionLambdaL2Regularization/.{opts}/.Options[MlpOptionsTraining];
	    costFunctionType = MlpOptionCostFunctionType/.{opts}/.Options[MlpOptionsTraining];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* Training set optimization options *)
		numberOfTrainingSetOptimizationSteps = UtilityOptionOptimizationSteps/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		
		ParallelNeeds[{"CIP`Mlp`", "CIP`Cluster`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[numberOfTrainingSetOptimizationSteps, multipleMlps, optimizationMethod, initialWeights,
						initialNetworks, weightsValueLimit, minimizationPrecision, maximumNumberOfIterations, numberOfIterationsToImprove, 
						unusedOptionParameter11, unusedOptionParameter12, unusedOptionParameter13, unusedOptionParameter21, unusedOptionParameter22, unusedOptionParameter23,
						clusterMethod, maximumNumberOfEpochs, scalarProductMinimumTreshold, maximumNumberOfTrialSteps, activationAndScaling, 
						normalizationType, randomValueInitialization, deviationCalculationMethod, blackListLength, lambdaL2Regularization];
			
		listOfTrainingAndTestSetsInfoAndScanReport = ParallelTable[
			If[numberOfTrainingSetOptimizationSteps > 0,
				
				(* ------------------------------------------------------------------------------------------------------ *)
				(* Training set optimization *)
				
				mlpTrainOptimization = 
					GetMlpTrainOptimization[
						dataSet, 
						numberOfHiddenNeurons, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						MlpOptionMultipleMlps -> multipleMlps,
		    			MlpOptionOptimizationMethod -> optimizationMethod,
						MlpOptionInitialWeights -> initialWeights,
						MlpOptionInitialNetworks -> initialNetworks,
						MlpOptionWeightsValueLimit -> weightsValueLimit,
						MlpOptionMinimizationPrecision -> minimizationPrecision,
						MlpOptionMaximumIterations -> maximumNumberOfIterations,
						MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
						MlpOptionUnused11 -> unusedOptionParameter11,
						MlpOptionUnused12 -> unusedOptionParameter12,
						MlpOptionUnused13 -> unusedOptionParameter13,
						MlpOptionUnused21 -> unusedOptionParameter21,
						MlpOptionUnused22 -> unusedOptionParameter22,
						MlpOptionUnused23 -> unusedOptionParameter23,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						MlpOptionActivationAndScaling -> activationAndScaling,
						MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
						MlpOptionCostFunctionType -> costFunctionType,
						DataTransformationOptionNormalizationType -> normalizationType,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
					
				bestIndex = GetBestMlpRegressOptimization[mlpTrainOptimization];
				trainingAndTestSetList = mlpTrainOptimization[[3]];
				mlpInfoList = mlpTrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlpInfo = mlpInfoList[[bestIndex]],
				
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
				
				currentMlpInfo = 
					FitMlp[
						currentTrainingSet,
						numberOfHiddenNeurons,
						MlpOptionMultipleMlps -> multipleMlps,
	   		 			MlpOptionOptimizationMethod -> optimizationMethod,
						MlpOptionInitialWeights -> initialWeights,
						MlpOptionInitialNetworks -> initialNetworks,
						MlpOptionWeightsValueLimit -> weightsValueLimit,
						MlpOptionMinimizationPrecision -> minimizationPrecision,
						MlpOptionMaximumIterations -> maximumNumberOfIterations,
						MlpOptionIterationsToImprove -> numberOfIterationsToImprove,
			 			MlpOptionReportIteration -> 0,
						MlpOptionUnused11 -> unusedOptionParameter11,
						MlpOptionUnused12 -> unusedOptionParameter12,
						MlpOptionUnused13 -> unusedOptionParameter13,
						MlpOptionUnused21 -> unusedOptionParameter21,
						MlpOptionUnused22 -> unusedOptionParameter22,
						MlpOptionUnused23 -> unusedOptionParameter23,
		    			UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    			MlpOptionActivationAndScaling -> activationAndScaling,
		    			MlpOptionLambdaL2Regularization -> lambdaL2Regularization,
		    			MlpOptionCostFunctionType -> costFunctionType,
		    			DataTransformationOptionNormalizationType -> normalizationType
					];
			];
			
			pureFunction = Function[inputs, CalculateMlpOutputs[inputs, currentMlpInfo]];
			trainingSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTrainingSet, pureFunction]];
			testSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTestSet, pureFunction]];
			{
				{currentTrainingAndTestSet, currentMlpInfo},
				
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

ShowMlpOutput2D[

	(* Shows 2D mlp output.

	   Returns: Nothing *)


    (* Index of input neuron that receives argumentValue *)
    indexOfInput_?IntegerQ,

    (* Index of output neuron that returns function value *)
    indexOfFunctionValueOutput_?IntegerQ,
    
    (* Mlp input in original units: 
       inputsInOriginalUnits = {inputValue1, inputValue2, ...} 
       Value of input neuron with specified index (indexOfInput) is replaced by argumentValue *)
    input_/;VectorQ[input, NumberQ],
    
    (* Arguments to be displayed as points:
       arguments: {argumentValue1, argumentValue2, ...} *)
    arguments_/;VectorQ[arguments, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlpInfo_
      
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
						CalculateMlpValue2D[arguments[[i]], indexOfInput, indexOfFunctionValueOutput, input, mlpInfo]
					},
						
					{i, Length[arguments]}
				],
				
			points = {}
		];
		
		dataSetScaleInfo = mlpInfo[[2]];
		inputsMinMaxList = dataSetScaleInfo[[1, 1]];
		xMin = inputsMinMaxList[[indexOfInput, 1]];
		xMax = inputsMinMaxList[[indexOfInput, 2]];
		
		labels = 
			{
				StringJoin["Argument Value of Input ", ToString[indexOfInput]],
				StringJoin["Value of Output ", ToString[indexOfFunctionValueOutput]],
				"Mlp Output"
			};
		Print[
			CIP`Graphics`PlotPoints2DAboveFunction[
				points, 
				Function[x, CalculateMlpValue2D[x, indexOfInput, indexOfFunctionValueOutput, input, mlpInfo]], 
				labels,
				GraphicsOptionArgumentRange2D -> {xMin, xMax}
			]
		]
	];

ShowMlpOutput3D[

	(* Shows 3D mlp output.

	   Returns: Graphics3D *)


    (* Index of input neuron that receives argumentValue1 *)
    indexOfInput1_?IntegerQ,

    (* Index of input neuron that receives argumentValue2 *)
    indexOfInput2_?IntegerQ,

    (* Index of output neuron that returns function value *)
    indexOfFunctionValueOutput_?IntegerQ,
    
    (* Mlp input in original units: 
       inputsInOriginalUnits = {inputValue1, inputValue2, ...} 
       Value of input neuron with specified index (indexOfInput) is replaced by argumentValue *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlpInfo_,
    
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
	
		dataSetScaleInfo = mlpInfo[[2]];
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
				Function[{x1, x2}, CalculateMlpValue3D[x1, x2, indexOfInput1, indexOfInput2, indexOfFunctionValueOutput, input, mlpInfo]], 
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

ShowMlpClassificationResult[

	(* Shows result of mlp classification for training and test set according to named property list.

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
    mlpInfo_,
    
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
		ShowMlpSingleClassification[
			namedPropertyList,
			trainingSet, 
			mlpInfo,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionMinMaxIndex -> minMaxIndex
		];
		
		(* Analyze test set *)
		If[Length[testSet] > 0,
			Print["Test Set:"];
			ShowMlpSingleClassification[
				namedPropertyList,
				testSet, 
				mlpInfo,
				GraphicsOptionImageSize -> imageSize,
				GraphicsOptionMinMaxIndex -> minMaxIndex
			];
		]
	];

ShowMlpSingleClassification[

	(* Shows result of mlp classification for data set according to named property list.

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
    mlpInfo_,
    
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

   		pureFunction = Function[inputs, CalculateMlpClassNumbers[inputs, mlpInfo]];
		CIP`Graphics`ShowClassificationResult[
			namedPropertyList,
			classificationDataSet, 
			pureFunction,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionMinMaxIndex -> minMaxIndex
		]
	];

ShowMlpClassificationScan[

	(* Shows result of Mlp based classification scan of clustered training sets.

	   Returns: Nothing *)


	(* mlpClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlpInfo1}, {trainingAndTestSet2, mlpInfo2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, classification result in percent for training set}, {trainingFraction, classification result in percent for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)
	mlpClassificationScan_,
	
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
			mlpClassificationScan,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMlpInputRelevanceClass[

	(* Shows mlpInputComponentRelevanceListForClassification.

	   Returns: Nothing *)


	(* mlpInputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlpInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) correct classification in percent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best correct classification in percent of test set} *)
	mlpInputComponentRelevanceListForClassification_,
	
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
			mlpInputComponentRelevanceListForClassification,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMlpInputRelevanceRegress[

	(* Shows mlpInputComponentRelevanceListForRegression.

	   Returns: Nothing *)


	(* mlpInputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlpInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)
	mlpInputComponentRelevanceListForRegression_,
	
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
			mlpInputComponentRelevanceListForRegression,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];
	
ShowMlpRegressionResult[

	(* Shows result of mlp regression for training and test set according to named property list.
	
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
    mlpInfo_,
    
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
		ShowMlpSingleRegression[
			namedPropertyList,
			trainingSet, 
			mlpInfo,
			GraphicsOptionPointSize -> pointSize,
			GraphicsOptionPointColor -> pointColor
		];
		
		(* Analyze test set *)
		If[Length[testSet] > 0,
			Print["Test Set:"];
			ShowMlpSingleRegression[
				namedPropertyList,
				testSet, 
				mlpInfo,
				GraphicsOptionPointSize -> pointSize,
				GraphicsOptionPointColor -> pointColor
			]
		]
	];

ShowMlpSingleRegression[
    
	(* Shows result of mlp regression for data set according to named property list.
	
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
    mlpInfo_,
    
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

		pureFunction = Function[inputs, CalculateMlpOutputs[inputs, mlpInfo]];
		CIP`Graphics`ShowRegressionResult[
			namedPropertyList,
			dataSet, 
			pureFunction,
			GraphicsOptionPointSize -> pointSize,
			GraphicsOptionPointColor -> pointColor
		]
	];

ShowMlpRegressionScan[

	(* Shows result of Mlp based regression scan of clustered training sets.

	   Returns: Nothing *)


	(* mlpRegressionScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlpInfo1}, {trainingAndTestSet2, mlpInfo2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, RMSE for training set}, {trainingFraction, RMSE for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)
	mlpRegressionScan_,
	
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
			mlpRegressionScan,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMlpSeriesClassificationResult[

	(* Shows result of Mlp series classifications for training and test set.

	   Returns: Nothing *)


	(* mlpSeriesClassificationResult: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in mlpInfoList, classification result in percent for training set}
	   testPoint[[i]]: {index i in mlpInfoList, classification result in percent for test set} *)
	mlpSeriesClassificationResult_,
    
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

		trainingPoints2D = mlpSeriesClassificationResult[[1]];
		testPoints2D = mlpSeriesClassificationResult[[2]];
		
		If[Length[testPoints2D] > 0,

			(* Training and test set *)
			labels = {"mlpInfo index", "Correct classifications [%]", "Training (green), Test (red)"};
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
			Print["Best test set classification with mlpInfo index = ", bestIndexList],
		
			(* Training set only *)
			labels = {"mlpInfo index", "Correct classifications [%]", "Training (green)"};
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
			Print["Best training set classification with mlpInfo index = ", bestIndexList]			
		]
	];

ShowMlpSeriesRmse[

	(* Shows RMSE of Mlp series for training and test set.

	   Returns: Nothing *)


	(* mlpSeriesRmse: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in mlpInfoList, RMSE for training set}
	   testPoint[[i]]: {index i in mlpInfoList, RMSE for test set} *)
	mlpSeriesRmse_,
    
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

		trainingPoints2D = mlpSeriesRmse[[1]];
		testPoints2D = mlpSeriesRmse[[2]];

		If[Length[testPoints2D] > 0,
			
			(* Training and test set *)
			labels = {"mlpInfo index", "RMSE", "Training (green), Test (red)"};
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
			Print["Best test set regression with mlpInfo index = ", bestIndexList],

			(* Training set only *)
			labels = {"mlpInfo index", "RMSE", "Training (green)"};
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
			Print["Best training set regression with mlpInfo index = ", bestIndexList]			
		]
	];

ShowMlpTraining[

	(* Shows training of mlp.

	   Returns: Nothing *)


  	(* See "Frequently used data structures" *)
    mlpInfo_
    
	] :=
  
	Module[
    
		{
			i,
			labels,
			mlpTrainingResults,
			trainingSetMeanSquaredErrorList,
			testSetMeanSquaredErrorList
		},

		mlpTrainingResults = mlpInfo[[3]];
		Do[

			If[Length[mlpTrainingResults] == 1,
				
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
			
			trainingSetMeanSquaredErrorList = mlpTrainingResults[[i, 1]];
			testSetMeanSquaredErrorList = mlpTrainingResults[[i, 2]];
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
		
			{i, Length[mlpTrainingResults]}
		]
	];

ShowMlpTrainOptimization[

	(* Shows training set optimization result of mlp.

	   Returns: Nothing *)


	(* mlpTrainOptimization = {trainingSetRmseList, testSetRmseList, not interesting, not interesting}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set} *)
	mlpTrainOptimization_,
    
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
			mlpTrainOptimization, 
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
