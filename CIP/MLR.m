(*
----------------------------------------------------------------------------------------
Computational Intelligence Packages (CIP): Package Multiple Linear Regression (MLR)
Version 3.1 for Mathematica 11 or higher
----------------------------------------------------------------------------------------

Authors: Kolja Berger (parallelization for CIP 2.0), Achim Zielesny 

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
----------------------------------------------------------------------------------------
*)

(* ::Section:: *)
(* Frequently used data structures *)

(*
-----------------------------------------------------------------------
Frequently used data structures
-----------------------------------------------------------------------
mlrInfo: {mlrFitResult, dataSetScaleInfo, dataTransformationMode, outputOffsets, normalizationInfo}

	mlrFitResult: See code of FitMlr[] and signature of Fit[]
	dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs}, see CIP`DataTransformation`GetDataSetScaleInfo 
	dataTransformationMode <> "None": All values of dataSet are internally transformed by Log/Sqrt operation, "None": Otherwise 
	outputOffsets: Offset value for transformations of outputs
	normalizationInfo: {normalizationType, meanAndStandardDeviationList}, see CIP`DataTransformation`GetDataMatrixNormalizationInfo
-----------------------------------------------------------------------
*)

(* ::Section:: *)
(* Package and dependencies *)

BeginPackage["CIP`MLR`", {"CIP`Utility`", "CIP`Graphics`", "CIP`DataTransformation`", "CIP`Cluster`"}]

(* ::Section:: *)
(* Off settings *)

Off[General::"spell1"]
Off[General::shdw]
Off[General::compat]

(* ::Section:: *)
(* Options *)

Options[MlrOptionsDataTransformation] = 
{
	(* Data transformation mode: "None", "Log", "Sqrt" *)
    MlrOptionDataTransformationMode -> "None"
}

(* ::Section:: *)
(* Declarations *)

CalculateMlrValue2D::usage = 
	"CalculateMlrValue2D[argumentValue, indexOfInput, indexOfOutput, input, mlrInfo]"

CalculateMlrValue3D::usage = 
	"CalculateMlrValue3D[argumentValue1, argumentValue2, indexOfInput1, indexOfInput2, indexOfOutput, input, mlrInfo]"

CalculateMlrClassNumber::usage = 
	"CalculateMlrClassNumber[input, mlrInfo]"

CalculateMlrClassNumbers::usage = 
	"CalculateMlrClassNumbers[inputs, mlrInfo]"

CalculateMlrDataSetRmse::usage = 
	"CalculateMlrDataSetRmse[dataSet, mlrInfo]"

CalculateMlrOutput::usage = 
	"CalculateMlrOutput[input, mlrInfo]"

CalculateMlrOutputs::usage = 
	"CalculateMlrOutputs[inputs, mlrInfo]"

FitMlr::usage = 
	"FitMlr[dataSet,options]"

GetBestMlrClassOptimization::usage = 
	"GetBestMlrClassOptimization[mlrTrainOptimization, options]"

GetBestMlrRegressOptimization::usage = 
	"GetBestMlrRegressOptimization[mlrTrainOptimization, options]"

GetMlrInputInclusionClass::usage = 
	"GetMlrInputInclusionClass[trainingAndTestSet, options]"

GetMlrInputInclusionRegress::usage = 
	"GetMlrInputInclusionRegress[trainingAndTestSet, options]"

GetMlrInputRelevanceClass::usage = 
	"GetMlrInputRelevanceClass[trainingAndTestSet, options]"

GetMlrClassRelevantComponents::usage = 
    "GetMlrClassRelevantComponents[mlrInputComponentRelevanceListForClassification, numberOfComponents]"

GetMlrInputRelevanceRegress::usage = 
	"GetMlrInputRelevanceRegress[trainingAndTestSet, options]"

GetMlrRegressRelevantComponents::usage = 
    "GetMlrRegressRelevantComponents[mlrInputComponentRelevanceListForRegression, numberOfComponents]"

GetMlrRegressionResult::usage = 
	"GetMlrRegressionResult[namedProperty, dataSet, mlrInfo, options]"

GetMlrTrainOptimization::usage = 
	"GetMlrTrainOptimization[dataSet, trainingFraction, numberOfTrainingSetOptimizationSteps, options]"

ScanClassTrainingWithMlr::usage = 
	"ScanClassTrainingWithMlr[dataSet, trainingFractionList, options]"

ScanRegressTrainingWithMlr::usage = 
	"ScanRegressTrainingWithMlr[dataSet, trainingFractionList, options]"

ShowMlrOutput3D::usage = 
	"ShowMlrOutput3D[indexOfInput1, indexOfInput2, indexOfOutput, input, mlrInfo, graphicsOptions, displayFunction]"

ShowMlrClassificationResult::usage = 
	"ShowMlrClassificationResult[namedPropertyList, trainingAndTestSet, mlrInfo]"

ShowMlrSingleClassification::usage = 
	"ShowMlrSingleClassification[namedPropertyList, dataSet, mlrInfo]"

ShowMlrClassificationScan::usage = 
	"ShowMlrClassificationScan[mlrClassificationScan, options]"

ShowMlrInputRelevanceClass::usage = 
	"ShowMlrInputRelevanceClass[mlrInputComponentRelevanceListForClassification, options]"
	
ShowMlrInputRelevanceRegress::usage = 
	"ShowMlrInputRelevanceRegress[mlrInputComponentRelevanceListForRegression, options]"

ShowMlrRegressionResult::usage = 
	"ShowMlrRegressionResult[namedPropertyList, trainingAndTestSet, mlrInfo, options]"

ShowMlrSingleRegression::usage = 
	"ShowMlrSingleRegression[namedPropertyList, dataSet, mlrInfo, options]"

ShowMlrRegressionScan::usage = 
	"ShowMlrRegressionScan[mlrRegressionScan, options]"

ShowMlrTrainOptimization::usage = 
	"ShowMlrTrainOptimization[mlrTrainOptimization, options]"
	
(* ::Section:: *)
(* Functions *)

Begin["`Private`"]

CalculateMlrValue2D[

	(* Calculates 2D output for specified argument and input for specified MLR.
	   This special method assumes an input and an output with one component only.

	   Returns:
	   Value of specified output component for argument *)


    (* Argument value for input component with index indexOfInput *)
    argumentValue_?NumberQ,
    
  	(* See "Frequently used data structures" *)
    mlrInfo_
    
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
			CalculateMlrValue2D[argumentValue, indexOfInput, indexOfFunctionValueOutput, input, mlrInfo]
		]
	];

CalculateMlrValue2D[

	(* Calculates 2D output for specified argument and input for specified MLR.

	   Returns:
	   Value of specified output component for argument and input *)


    (* Argument value for input component with index indexOfInput *)
    argumentValue_?NumberQ,
    
    (* Index of input component that receives argumentValue *)
    indexOfInput_?IntegerQ,

    (* Index of output component that returns function value *)
    indexOfOutput_?IntegerQ,
    
    (* Input in original units: 
       inputsInOriginalUnits = {inputValue1, inputValue2, ...} 
       Values of input components with specified indices (indexOfInput1, indexOfInput2) are replaced by argument values *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlrInfo_
    
	] :=
  
	Module[
    
		{
			currentInput,
			output
		},
		
		currentInput = ReplacePart[input, {indexOfInput -> argumentValue}];
		output = CalculateMlrOutput[currentInput, mlrInfo];
		Return[output[[indexOfOutput]]];
	];

CalculateMlrValue3D[

	(* Calculates 3D output for specified arguments for specified MLR. 
	   This specific methods assumes a MLR with input vector of length 2 and an output vector of length 1.

	   Returns:
	   Value of specified output component for input *)


    (* Argument value for input component with index indexOfInput1 *)
    argumentValue1_?NumberQ,
    
    (* Argument value for input component with index indexOfInput2 *)
    argumentValue2_?NumberQ,
    
  	(* See "Frequently used data structures" *)
    mlrInfo_
    
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
		input = {0.0,0.0};
		Return[
			CalculateMlrValue3D[argumentValue1, argumentValue2, indexOfInput1, indexOfInput2, indexOfOutput, input, mlrInfo]
		];
	];

CalculateMlrValue3D[

	(* Calculates 3D output for specified arguments and input for specified MLR.

	   Returns:
	   Value of specified output component for arguments and input *)


    (* Argument value for input component with index indexOfInput1 *)
    argumentValue1_?NumberQ,
    
    (* Argument value for input component with index indexOfInput2 *)
    argumentValue2_?NumberQ,
    
    (* Index of input component that receives argumentValue1 *)
    indexOfInput1_?IntegerQ,

    (* Index of input component that receives argumentValue2 *)
    indexOfInput2_?IntegerQ,

    (* Index of output component that returns function value *)
    indexOfOutput_?IntegerQ,
    
    (* Input in original units: 
       inputsInOriginalUnits = {inputValue1, inputValue2, ...} 
       Values of input components with specified indices (indexOfInput1, indexOfInput2) are replaced by argument values *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlrInfo_
    
	] :=
  
	Module[
    
		{
			currentInput,
			output
		},
		
		currentInput = ReplacePart[input, {indexOfInput1 -> argumentValue1, indexOfInput2 -> argumentValue2}];
		output = CalculateMlrOutput[currentInput, mlrInfo];
		Return[output[[indexOfOutput]]];
	];

CalculateMlrClassNumber[

	(* Returns class number for specified input for classification MLR.

	   Returns:
	   Class number of input *)

    
    (* Input in original units: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlrInfo_
    
	] :=
  
	Module[
    
		{
			dataSetScaleInfo,
			mlrFitResult,
			normalizationInfo,
			scaledInput,
			scaledOutput
		},
    
		mlrFitResult = mlrInfo[[1]];
    	dataSetScaleInfo = mlrInfo[[2]];
    	normalizationInfo = mlrInfo[[5]];

		scaledInput = First[CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo]];
		scaledOutput = GetInternalMlrOutput[scaledInput, mlrFitResult];
		Return[CIP`Utility`GetPositionOfMaximumValue[scaledOutput]]
	];

CalculateMlrClassNumbers[

	(* Returns class numbers for specified inputs for classification MLR.

	   Returns:
	   {class number of input1, class number of input2, ...} *)

    
    (* {inputsInOriginalUnit1, inputsInOriginalUnit2, ...}
        inputsInOriginalUnit: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    inputs_/;MatrixQ[inputs, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlrInfo_
    
	] :=
  
	Module[
    
		{
			i,
			mlrFitResult,
			dataSetScaleInfo,
			normalizationInfo,
			scaledInputs,
			scaledOutputs
		},

		mlrFitResult = mlrInfo[[1]];
    	dataSetScaleInfo = mlrInfo[[2]];
    	normalizationInfo = mlrInfo[[5]];

		scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
		scaledOutputs = GetInternalMlrOutputs[scaledInputs, mlrFitResult];
		Return[
			Table[
				CIP`Utility`GetPositionOfMaximumValue[scaledOutputs[[i]]],
				
				{i, Length[scaledOutputs]}
			]
		]
	];

CalculateMlrCorrectClassificationInPercent[

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
    mlrInfo_
    
	] :=
  
	Module[
    
		{
			pureFunction
		},

		pureFunction = Function[inputs, CalculateMlrClassNumbers[inputs, mlrInfo]];
		Return[CIP`Utility`GetCorrectClassificationInPercent[classificationDataSet, pureFunction]]
	];

CalculateMlrDataSetRmse[

	(* Returns RMSE of data set.

	   Returns: 
	   RMSE of data set *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

  	(* See "Frequently used data structures" *)
    mlrInfo_
    
	] :=
  
	Module[
    
		{
			pureFunction,
			rmse
		},

		pureFunction = Function[inputs, CalculateMlrOutputs[inputs, mlrInfo]];
		rmse = Sqrt[CIP`Utility`GetMeanSquaredError[dataSet, pureFunction]];
		Return[rmse]
	];

CalculateMlrOutput[

	(* Calculates output for specified input for MLR.

	   Returns:
	   output: {transformedOutputValue1, transformedOutputValue2, ...} *)

    
    (* Input in original units: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlrInfo_
    
	] :=
  
	Module[
    
		{
			dataSetScaleInfo,
			i,
			k,
			dataTransformationMode,
			mlrFitResult,
			normalizationInfo,
			outputOffsets,
			outputsInOriginalUnits,
			scaledInput,
			scaledOutput,
			unscaledOutputs
		},
    
		mlrFitResult = mlrInfo[[1]];
    	dataSetScaleInfo = mlrInfo[[2]];
    	dataTransformationMode = mlrInfo[[3]];
    	outputOffsets = mlrInfo[[4]];
    	normalizationInfo = mlrInfo[[5]];

		scaledInput = First[CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo]];
		scaledOutput = GetInternalMlrOutput[scaledInput, mlrFitResult];

		Switch[dataTransformationMode,
			
			"None",
			outputsInOriginalUnits = CIP`DataTransformation`ScaleDataMatrixReverse[{scaledOutput}, dataSetScaleInfo[[2]]],

			"Log",
			(* All values are internally transformed by Log operation *)
			unscaledOutputs = CIP`DataTransformation`ScaleDataMatrixReverse[{scaledOutput}, dataSetScaleInfo[[2]]];
			outputsInOriginalUnits = 
				Table[
					Exp[unscaledOutputs[[i]]] - outputOffsets,
					
					{i, Length[unscaledOutputs]}
				],

			"Sqrt",
			(* All values are internally transformed by Sqrt operation *)
			unscaledOutputs = CIP`DataTransformation`ScaleDataMatrixReverse[{scaledOutput}, dataSetScaleInfo[[2]]];
			outputsInOriginalUnits = 
				Table[
					Table[
						unscaledOutputs[[i, k]]*unscaledOutputs[[i, k]],
						
						{k, Length[unscaledOutputs[[i]]]}
					] - outputOffsets,
					
					{i, Length[unscaledOutputs]}
				]
		];

		Return[First[outputsInOriginalUnits]]
	];

CalculateMlrOutputs[

	(* Calculates outputs for specified inputs for MLR.

	   Returns:
	   outputs = {output1, ..., output<Length[inputs]>}
	   output: {transformedOutputValue1, transformedOutputValue2, ...} *)

    
    (* {inputsInOriginalUnit1, inputsInOriginalUnit2, ...}
        inputsInOriginalUnit: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    inputs_/;MatrixQ[inputs, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlrInfo_
    
	] :=
  
	Module[
    
		{
			dataSetScaleInfo,
			i,
			dataTransformationMode,
			mlrFitResult,
			normalizationInfo,
			outputOffsets,
			outputsInOriginalUnits,
			scaledInputs,
			scaledOutputs,
			unscaledOutputs
		},
    
		mlrFitResult = mlrInfo[[1]];
    	dataSetScaleInfo = mlrInfo[[2]];
    	dataTransformationMode = mlrInfo[[3]];
    	outputOffsets = mlrInfo[[4]];
    	normalizationInfo = mlrInfo[[5]];

		scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
		scaledOutputs = GetInternalMlrOutputs[scaledInputs, mlrFitResult];

		Switch[dataTransformationMode,
			
			"None",
			outputsInOriginalUnits = CIP`DataTransformation`ScaleDataMatrixReverse[scaledOutputs, dataSetScaleInfo[[2]]],

			"Log",
			(* All values are internally transformed by Log operation *)
			unscaledOutputs = CIP`DataTransformation`ScaleDataMatrixReverse[scaledOutputs, dataSetScaleInfo[[2]]];
			outputsInOriginalUnits = 
				Table[
					Exp[unscaledOutputs[[i]]] - outputOffsets,
					
					{i, Length[unscaledOutputs]}
				],

			"Sqrt",
			(* All values are internally transformed by Sqrt operation *)
			unscaledOutputs = CIP`DataTransformation`ScaleDataMatrixReverse[scaledOutputs, dataSetScaleInfo[[2]]];
			outputsInOriginalUnits = 
				Table[
					Table[
						unscaledOutputs[[i, k]]*unscaledOutputs[[i, k]],
						
						{k, Length[unscaledOutputs[[i]]]}
					] - outputOffsets,
					
					{i, Length[unscaledOutputs]}
				]
		];

		Return[outputsInOriginalUnits];
	];

FitMlr[

	(* Trains with MLR

	   Returns: 
	   mlrInfo (see "Frequently used data structures") *)
	   
	   
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,
	
	(* Options *)
	opts___
		
	] :=
  
	Module[
    
		{
			applicationResult,
			correctedDataSet,
			correctedDataSetScaleInfo,
			ioPair,
			dataTransformationMode,
			dataSetScaleInfo,
			mlrFitResult,
			functionVector,
			i,
			k,
			mlrInputDataList,
			normalizationInfo,
			normalizationType,
			numberOfInputVariables,
			numberOfOutputVariables,
			outputOffsets,
			scaledDataSet,
			variableVector,
			targetInterval
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    dataTransformationMode = MlrOptionDataTransformationMode/.{opts}/.Options[MlrOptionsDataTransformation];
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		dataSetScaleInfo = CIP`DataTransformation`GetDataSetScaleInfo[dataSet, targetInterval, targetInterval];
		normalizationInfo = CIP`DataTransformation`GetDataSetNormalizationInfo[dataSet, normalizationType, dataSetScaleInfo];

		(* ----------------------------------------------------------------------------------------------------
		   Training
		   ---------------------------------------------------------------------------------------------------- *)
		Switch[dataTransformationMode,
			
			"None",
			correctedDataSetScaleInfo = dataSetScaleInfo;
			outputOffsets = Table[0.0, {Length[dataSet[[1, 2]]]}];
			scaledDataSet = CIP`DataTransformation`ScaleAndNormalizeDataSet[dataSet, dataSetScaleInfo, normalizationInfo],

			"Log",
			(* All values are internally transformed by Log operation *)
			applicationResult = CIP`DataTransformation`ApplyLogToDataSetOutputs[dataSet]; 
			correctedDataSet = applicationResult[[1]];
			outputOffsets = applicationResult[[2]];
			correctedDataSetScaleInfo = CIP`DataTransformation`CorrectDataSetScaleInfoForLogApplication[dataSetScaleInfo];
			scaledDataSet = CIP`DataTransformation`ScaleAndNormalizeDataSet[correctedDataSet, correctedDataSetScaleInfo, normalizationInfo],

			"Sqrt",
			(* All values are internally transformed by Sqrt operation *)
			applicationResult = CIP`DataTransformation`ApplySqrtToDataSetOutputs[dataSet]; 
			correctedDataSet = applicationResult[[1]];
			outputOffsets = applicationResult[[2]];
			correctedDataSetScaleInfo = CIP`DataTransformation`CorrectDataSetScaleInfoForSqrtApplication[dataSetScaleInfo];
			scaledDataSet = CIP`DataTransformation`ScaleAndNormalizeDataSet[correctedDataSet, correctedDataSetScaleInfo, normalizationInfo]
		];
		
		(* Initialization *)
		numberOfInputVariables = Length[scaledDataSet[[1, 1]]];
		numberOfOutputVariables = Length[scaledDataSet[[1, 2]]];
		
		(* Clear mlrVariable variable. NOTE: This is NOT a local variable *)
		Clear[mlrVariable];

		(* Create subscripted mlrVariable variable vector and function vector. NOTE : These are NOT local variables *)
		variableVector = Table[Subscript[mlrVariable, i], {i, numberOfInputVariables}];
		functionVector = Flatten[{1.0, variableVector}];

		(* Transform data for Fit[] (see signature of Fit) *)
		mlrInputDataList = 
			Table[
				Table[
					ioPair = scaledDataSet[[k]];
					AppendTo[ioPair[[1]], ioPair[[2, i]]],
					
					{k, Length[scaledDataSet]}
				],
				
				{i, numberOfOutputVariables}
			];

		(* Fit data *)
		mlrFitResult = 
			Table[
				Fit[mlrInputDataList[[i]], functionVector, variableVector], 
					
				{i, numberOfOutputVariables}
			];
		
		(* ----------------------------------------------------------------------------------------------------
		   Return mlrInfo
		   ---------------------------------------------------------------------------------------------------- *)
    	Return[{mlrFitResult, correctedDataSetScaleInfo, dataTransformationMode, outputOffsets, normalizationInfo}]
	];

GetBestMlrClassOptimization[

	(* Returns best training set optimization result of MLR for classification.

	   Returns: 
	   Best index for classification *)


	(* mlrTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlrInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlrInfoList: List with mlrInfo
	   mlrInfoList[[i]] refers to optimization step i *)
	mlrTrainOptimization_,
	
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
				GetBestMlrClassOptimizationPC[
					mlrTrainOptimization,
					UtilityOptionBestOptimization -> bestOptimization
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetBestMlrClassOptimizationSC[
					mlrTrainOptimization,
					UtilityOptionBestOptimization -> bestOptimization
				]
			]
		]
	];
	
GetBestMlrClassOptimizationSC[

	(* Returns best training set optimization result of MLR for classification.

	   Returns: 
	   Best index for classification *)


	(* mlrTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlrInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlrInfoList: List with mlrInfo
	   mlrInfoList[[i]] refers to optimization step i *)
	mlrTrainOptimization_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			k,
			trainingAndTestSetList,
			mlrInfoList,
			maximumCorrectClassificationInPercent,
			mlrInfo,
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
			trainingAndTestSetList = mlrTrainOptimization[[3]];
			mlrInfoList = mlrTrainOptimization[[4]];
			maximumCorrectClassificationInPercent = -1.0;
			Do[
				testSet = trainingAndTestSetList[[k, 2]];
				mlrInfo = mlrInfoList[[k]];
				correctClassificationInPercent = CalculateMlrCorrectClassificationInPercent[testSet, mlrInfo];
				If[correctClassificationInPercent > maximumCorrectClassificationInPercent,
					maximumCorrectClassificationInPercent = correctClassificationInPercent;
					bestIndex = k
				],
				
				{k, Length[mlrInfoList]}
			],
			
			(* ------------------------------------------------------------------------------- *)			
			"MinimumDeviation",
			trainingAndTestSetList = mlrTrainOptimization[[3]];
			mlrInfoList = mlrTrainOptimization[[4]];
			minimumDeviation = Infinity;
			Do[
				trainingSet = trainingAndTestSetList[[k, 1]];
				testSet = trainingAndTestSetList[[k, 2]];
				mlrInfo = mlrInfoList[[k]];
				trainingSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[trainingSet, mlrInfo];
				testSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[testSet, mlrInfo];
				deviation = Abs[testSetCorrectClassificationInPercent - trainingSetCorrectClassificationInPercent];
				If[deviation < minimumDeviation || (deviation == minimumDeviation && testSetCorrectClassificationInPercent < bestTestSetCorrectClassificationInPercent),
					minimumDeviation = deviation;
					bestTestSetCorrectClassificationInPercent = testSetCorrectClassificationInPercent;
					bestIndex = k
				],
				
				{k, Length[mlrInfoList]}
			]
		];

		Return[bestIndex]
	];
	
GetBestMlrClassOptimizationPC[

	(* Returns best training set optimization result of MLR for classification.

	   Returns: 
	   Best index for classification *)


	(* mlrTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlrInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlrInfoList: List with mlrInfo
	   mlrInfoList[[i]] refers to optimization step i *)
	mlrTrainOptimization_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			k,
			trainingAndTestSetList,
			mlrInfoList,
			maximumCorrectClassificationInPercent,
			mlrInfo,
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
			trainingAndTestSetList = mlrTrainOptimization[[3]];
			mlrInfoList = mlrTrainOptimization[[4]];
			maximumCorrectClassificationInPercent = -1.0;
			
			ParallelNeeds[{"CIP`MLR`", "CIP`DataTransformation`", "CIP`Utility`"}];
			DistributeDefinitions[trainingAndTestSetList, mlrInfoList];
			
			correctClassificationInPercentList = ParallelTable[
				testSet = trainingAndTestSetList[[k, 2]];
				mlrInfo = mlrInfoList[[k]];
				
				CalculateMlrCorrectClassificationInPercent[testSet, mlrInfo],
				
				{k, Length[mlrInfoList]}
			];
				
			Do[	
				If[correctClassificationInPercentList[[k]] > maximumCorrectClassificationInPercent,
					maximumCorrectClassificationInPercent = correctClassificationInPercentList[[k]];
					bestIndex = k
				],
				
				{k, Length[mlrInfoList]}
			],
			
			(* ------------------------------------------------------------------------------- *)			
			"MinimumDeviation",
			trainingAndTestSetList = mlrTrainOptimization[[3]];
			mlrInfoList = mlrTrainOptimization[[4]];
			minimumDeviation = Infinity;
			
			ParallelNeeds[{"CIP`MLR`", "CIP`DataTransformation`", "CIP`Utility`"}];
			DistributeDefinitions[trainingAndTestSetList, mlrInfoList];
			
			listOfTestSetCorrectClassificationInPercentAndDeviation = ParallelTable[
				trainingSet = trainingAndTestSetList[[k, 1]];
				testSet = trainingAndTestSetList[[k, 2]];
				mlrInfo = mlrInfoList[[k]];
				trainingSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[trainingSet, mlrInfo];
				testSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[testSet, mlrInfo];
				deviation = Abs[testSetCorrectClassificationInPercent - trainingSetCorrectClassificationInPercent];
				
				{
					testSetCorrectClassificationInPercent,
					deviation
				},
				
				{k, Length[mlrInfoList]}
			];
				
			Do[	
				If[listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]] < minimumDeviation || (listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]] == minimumDeviation && listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 1]] < bestTestSetCorrectClassificationInPercent),
					minimumDeviation = listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]];
					bestTestSetCorrectClassificationInPercent = listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 1]];
					bestIndex = k
				],
				
				{k, Length[mlrInfoList]}
			]
		];

		Return[bestIndex]
	];

GetBestMlrRegressOptimization[

	(* Returns best optimization result of MLR for regression.

	   Returns: 
	   Best index for regression *)


	(* mlrTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlrInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlrInfoList: List with mlrInfo
	   mlrInfoList[[i]] refers to optimization step i *)
	mlrTrainOptimization_,
	
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
				mlrTrainOptimization, 
				UtilityOptionBestOptimization -> bestOptimization
			]
		]
	];

GetInternalMlrOutput[

	(* Returns output of MLR according to specified input.

	   Returns: output: {outputValue1, outputValue2, ..., outputValue<<numberOfoutputValues>} *)


    (* {inputValue1, inputValue2, ..., inputValue<<numberOfInputValues>} *)
    input_/;VectorQ[input, NumberQ],

	(* mlrFitResult: See code of FitMlr[] and signature of Fit[] *)
    mlrFitResult_
    
	] :=
  
	Module[
    
		{
			j,
			k,
			output,
			replacementList
		},
    
		(* NOTE: Subscripted mlrVariable variables are NOT local *)
		replacementList = 
			Table[
				Subscript[mlrVariable, j] -> input[[j]], 
					
				{j, Length[input]}
			];
        output = 
        	Table[
        		mlrFitResult[[k]] /. replacementList, 
        			
        		{k, Length[mlrFitResult]}
        	];
		Return[output]
	];

GetInternalMlrOutputs[

	(* Returns outputs of MLR according to specified inputs.

	   Returns: 
	   outputs: {output1, ..., output<Length[inputs]>} 
	   output[[i]] corresponds to inputs[[i]] *)

    (* inputs: {input1, input2, ...} 
       input: {inputValue1, inputValue2, ..., inputValue<<numberOfInputValues>} *)
    inputs_/;MatrixQ[inputs, NumberQ],

	(* mlrFitResult: See code of FitMlr[] and signature of Fit[] *)
    mlrFitResult_
	
	] :=
  
	Module[
    
		{
			i,
			j,
			k,
			outputs,
			replacementList,
			singleInput
		},
    
		(* NOTE: Subscripted mlrVariable variables are NOT local *)
		outputs = 
			Table[
		    	singleInput = inputs[[i]];
				replacementList = 
					Table[
						Subscript[mlrVariable, j] -> singleInput[[j]], 
							
						{j, Length[singleInput]}
					];
	    		Table[
	    			mlrFitResult[[k]] /. replacementList, 
	    				
	    			{k, Length[mlrFitResult]}
	    		],
	    
				{i, Length[inputs]}
		    ];
		Return[outputs]
	];

GetMlrInputInclusionClass[

	(* Analyzes relevance of input components by successive get-one-in for classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlrInputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mlrInfo}
	   trainingSetResult: {numberOfIncludedInputs, (best) classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, (best) classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			dataTransformationMode,
			targetInterval,
			isIntermediateOutput,
			numberOfInclusionsPerStepList,
			isRegression,
			inclusionStartList,
			parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MLR options *)   
		dataTransformationMode = MlrOptionDataTransformationMode/.{opts}/.Options[MlrOptionsDataTransformation];
	    (* DataTransformation options *)   
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    (* Utility options *)   
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfInclusionsPerStepList = UtilityOptionInclusionsPerStep/.{opts}/.Options[UtilityOptionsInclusion];
	    inclusionStartList = UtilityOptionInclusionStartList/.{opts}/.Options[UtilityOptionsInclusion];
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		isRegression = False;
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				GetMlrInputInclusionCalculationPC[
					trainingAndTestSet,
					isRegression,
					MlrOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetMlrInputInclusionCalculationSC[
					trainingAndTestSet,
					isRegression,
					MlrOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			]
		]
	];

GetMlrInputInclusionRegress[

	(* Analyzes relevance of input components by successive get-one-in for regression.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlrInputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mlrInfo}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			dataTransformationMode,
			targetInterval,
			isIntermediateOutput,
			numberOfInclusionsPerStepList,
			isRegression,
			inclusionStartList,
			parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MLR options *)   
		dataTransformationMode = MlrOptionDataTransformationMode/.{opts}/.Options[MlrOptionsDataTransformation];
	    (* DataTransformation options *)   
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    (* Utility options *)   
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfInclusionsPerStepList = UtilityOptionInclusionsPerStep/.{opts}/.Options[UtilityOptionsInclusion];
	    inclusionStartList = UtilityOptionInclusionStartList/.{opts}/.Options[UtilityOptionsInclusion];
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		isRegression = True;
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				GetMlrInputInclusionCalculationPC[
					trainingAndTestSet,
					isRegression,
					MlrOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList				
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
				Return[
				GetMlrInputInclusionCalculationSC[
					trainingAndTestSet,
					isRegression,
					MlrOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList				
				]
			]
		]
	];

GetMlrInputInclusionCalculationSC[

	(* Analyzes relevance of input components by successive get-one-in for regression and classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlrInputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mlrInfo}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			dataTransformationMode,
			targetInterval,
			currentIncludedInputComponentList,
			i,
			k,
			numberOfInputs,
			numberOfIncludedInputs,
			mlrInputComponentRelevanceList,
	        mlrInfo,
			normalizationType,
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
			inclusionStartList
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MLR options *)   
		dataTransformationMode = MlrOptionDataTransformationMode/.{opts}/.Options[MlrOptionsDataTransformation];
	    (* DataTransformation options *)   
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
	    (* Utility options *)   
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfInclusionsPerStepList = UtilityOptionInclusionsPerStep/.{opts}/.Options[UtilityOptionsInclusion];
	    inclusionStartList = UtilityOptionInclusionStartList/.{opts}/.Options[UtilityOptionsInclusion];

		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		numberOfInputs = First[Dimensions[trainingAndTestSet[[1, 1, 1]] ]];
		If[Length[numberOfInclusionsPerStepList] == 0,
			numberOfInclusionsPerStepList = Table[1, {numberOfInputs}];
		];				   
		includedInputComponentList = inclusionStartList;
		numberOfIncludedInputs = Length[includedInputComponentList];
		mlrInputComponentRelevanceList = {};
    
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
					mlrInfo = 
						FitMlr[
							trainingSet,
							MlrOptionDataTransformationMode -> dataTransformationMode,
							DataTransformationOptionTargetInterval -> targetInterval,
							DataTransformationOptionNormalizationType -> normalizationType
						];
					If[Length[testSet] > 0,
            
						testSetRmse = CalculateMlrDataSetRmse[testSet, mlrInfo];
						AppendTo[rmseList,{testSetRmse, i}],
          
						trainingSetRmse = CalculateMlrDataSetRmse[trainingSet, mlrInfo];
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
			mlrInfo = 
				FitMlr[
					trainingSet,
					MlrOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMlrDataSetRmse[trainingSet, mlrInfo];
					currentTestSetRmse = CalculateMlrDataSetRmse[testSet, mlrInfo];
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
							mlrInfo
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMlrDataSetRmse[trainingSet, mlrInfo];
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
							mlrInfo
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[trainingSet, mlrInfo];
					currentTestSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[testSet, mlrInfo];
					If[isIntermediateOutput,
						Print["numberOfIncludedInputs            = ", numberOfIncludedInputs];
						Print["currentIncludedInputComponentList                = ", currentIncludedInputComponentList];
						Print["currentTrainingSetCorrectClassificationInPercent = ", currentTrainingSetCorrectClassificationInPercent];
						Print["currentTestSetCorrectClassificationInPercent     = ", currentTestSetCorrectClassificationInPercent]
					];
					relevance = 
						{
							{N[numberOfIncludedInputs], currentTrainingSetCorrectClassificationInPercent}, 
							{N[numberOfIncludedInputs], currentTestSetCorrectClassificationInPercent}, 
							currentIncludedInputComponentList, 
							mlrInfo
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[trainingSet, mlrInfo];
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
							mlrInfo
						}
				]
			];	

			AppendTo[mlrInputComponentRelevanceList, relevance];
			includedInputComponentList = currentIncludedInputComponentList,
			
			{k, Length[numberOfInclusionsPerStepList]}
		];
		
		Return[mlrInputComponentRelevanceList]
	];
	
GetMlrInputInclusionCalculationPC[

	(* Analyzes relevance of input components by successive get-one-in for regression and classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlrInputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mlrInfo}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			dataTransformationMode,
			targetInterval,
			currentIncludedInputComponentList,
			i,
			k,
			numberOfInputs,
			numberOfIncludedInputs,
			mlrInputComponentRelevanceList,
	        mlrInfo,
			normalizationType,
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
			inclusionStartList
		},
		

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MLR options *)   
		dataTransformationMode = MlrOptionDataTransformationMode/.{opts}/.Options[MlrOptionsDataTransformation];
	    (* DataTransformation options *)   
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
	    (* Utility options *)   
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfInclusionsPerStepList = UtilityOptionInclusionsPerStep/.{opts}/.Options[UtilityOptionsInclusion];
	    inclusionStartList = UtilityOptionInclusionStartList/.{opts}/.Options[UtilityOptionsInclusion];

		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		numberOfInputs = First[Dimensions[trainingAndTestSet[[1, 1, 1]] ]];
		If[Length[numberOfInclusionsPerStepList] == 0,
			numberOfInclusionsPerStepList = Table[1, {numberOfInputs}];
		];				   
		includedInputComponentList = inclusionStartList;
		numberOfIncludedInputs = Length[includedInputComponentList];
		mlrInputComponentRelevanceList = {};
		
		ParallelNeeds[{"CIP`MLR`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[dataTransformationMode, targetInterval];
    
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
						mlrInfo = 
							FitMlr[
								trainingSet,
								MlrOptionDataTransformationMode -> dataTransformationMode,
								DataTransformationOptionTargetInterval -> targetInterval,
								DataTransformationOptionNormalizationType -> normalizationType
							];
						If[Length[testSet] > 0,
            
							testSetRmse = CalculateMlrDataSetRmse[testSet, mlrInfo];
							{testSetRmse, i},
          
							trainingSetRmse = CalculateMlrDataSetRmse[trainingSet, mlrInfo];
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
			mlrInfo = 
				FitMlr[
					trainingSet,
					MlrOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMlrDataSetRmse[trainingSet, mlrInfo];
					currentTestSetRmse = CalculateMlrDataSetRmse[testSet, mlrInfo];
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
							mlrInfo
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMlrDataSetRmse[trainingSet, mlrInfo];
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
							mlrInfo
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[trainingSet, mlrInfo];
					currentTestSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[testSet, mlrInfo];
					If[isIntermediateOutput,
						Print["numberOfIncludedInputs            = ", numberOfIncludedInputs];
						Print["currentIncludedInputComponentList                = ", currentIncludedInputComponentList];
						Print["currentTrainingSetCorrectClassificationInPercent = ", currentTrainingSetCorrectClassificationInPercent];
						Print["currentTestSetCorrectClassificationInPercent     = ", currentTestSetCorrectClassificationInPercent]
					];
					relevance = 
						{
							{N[numberOfIncludedInputs], currentTrainingSetCorrectClassificationInPercent}, 
							{N[numberOfIncludedInputs], currentTestSetCorrectClassificationInPercent}, 
							currentIncludedInputComponentList, 
							mlrInfo
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[trainingSet, mlrInfo];
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
							mlrInfo
						}
				]
			];	

			AppendTo[mlrInputComponentRelevanceList, relevance];
			includedInputComponentList = currentIncludedInputComponentList,
			
			
			
			{k, Length[numberOfInclusionsPerStepList]}
		];
		
		Return[mlrInputComponentRelevanceList]
	];
	
GetMlrInputRelevanceClass[

	(* Analyzes relevance of input components by successive leave-one-out for classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlrInputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlrInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, (best) classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			dataTransformationMode,
			targetInterval,
			isIntermediateOutput,
			numberOfExclusionsPerStepList,
			isRegression,
			parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MLR options *)   
		dataTransformationMode = MlrOptionDataTransformationMode/.{opts}/.Options[MlrOptionsDataTransformation];
	    (* DataTransformation options *)   
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    (* Utility options *)   
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfExclusionsPerStepList = UtilityOptionExclusionsPerStep/.{opts}/.Options[UtilityOptionsExclusion];
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		isRegression = False;
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				GetMlrInputRelevanceCalculationPC[
					trainingAndTestSet,
					isRegression,
					MlrOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			],
		
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetMlrInputRelevanceCalculationSC[
					trainingAndTestSet,
					isRegression,
					MlrOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			]
		]
	];

GetMlrClassRelevantComponents[

	(* Returns most-to-least-relevance sorted components from mlrInputComponentRelevanceListForClassification.

	   Returns: Returns most-to-least-relevance sorted components *)


	(* mlrInputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlrInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) correct classification in percent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best correct classification in percent of test set} *)
	mlrInputComponentRelevanceListForClassification_,
	
	(* Number of components to return *)
	numberOfComponents_?IntegerQ
    
	] :=
  
	Module[
    
		{},

		Return[
			CIP`Graphics`GetClassRelevantComponents[mlrInputComponentRelevanceListForClassification, numberOfComponents]
		]
	];

GetMlrInputRelevanceRegress[

	(* Analyzes relevance of input components by successive leave-one-out for regression.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlrInputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlrInfo}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			dataTransformationMode,
			targetInterval,
			isIntermediateOutput,
			numberOfExclusionsPerStepList,
			isRegression,
			parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MLR options *)   
		dataTransformationMode = MlrOptionDataTransformationMode/.{opts}/.Options[MlrOptionsDataTransformation];
	    (* DataTransformation options *)   
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    (* Utility options *)   
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfExclusionsPerStepList = UtilityOptionExclusionsPerStep/.{opts}/.Options[UtilityOptionsExclusion];
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		isRegression = True;
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				GetMlrInputRelevanceCalculationPC[
					trainingAndTestSet,
					isRegression,
					MlrOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetMlrInputRelevanceCalculationSC[
					trainingAndTestSet,
					isRegression,
					MlrOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			]
		]
	];

GetMlrRegressRelevantComponents[

	(* Returns most-to-least-relevance sorted components from mlrInputComponentRelevanceListForRegression.

	   Returns: Returns most-to-least-relevance sorted components *)


	(* mlrInputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlrInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)
	mlrInputComponentRelevanceListForRegression_,
	
	(* Number of components to return *)
	numberOfComponents_?IntegerQ
    
	] :=
  
	Module[
    
		{},

		Return[
			CIP`Graphics`GetRegressRelevantComponents[mlrInputComponentRelevanceListForRegression, numberOfComponents]
		]
	];

GetMlrInputRelevanceCalculationSC[

	(* Analyzes relevance of input components by successive leave-one-out for regression and classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlrInputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlrInfo}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			dataTransformationMode,
			targetInterval,
			currentRemovedInputComponentList,
			i,
			k,
			numberOfInputs,
			numberOfRemovedInputs,
			mlrInputComponentRelevanceList,
	        mlrInfo,
			normalizationType,
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
			currentTrainingSetCorrectClassificationInPercent
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MLR options *)   
		dataTransformationMode = MlrOptionDataTransformationMode/.{opts}/.Options[MlrOptionsDataTransformation];
	    (* DataTransformation options *)   
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
	    (* Utility options *)   
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfExclusionsPerStepList = UtilityOptionExclusionsPerStep/.{opts}/.Options[UtilityOptionsExclusion];

		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		numberOfInputs = First[Dimensions[trainingAndTestSet[[1, 1, 1]] ]];
		If[Length[numberOfExclusionsPerStepList] == 0,
			numberOfExclusionsPerStepList = Table[1, {numberOfInputs - 1}];
		];				   
		removedInputComponentList = {};
		mlrInputComponentRelevanceList = {};
    
		(* Result for no removal *)
		trainingSet = trainingAndTestSet[[1]];
		testSet = trainingAndTestSet[[2]];
		mlrInfo = 
			FitMlr[
				trainingSet,
				MlrOptionDataTransformationMode -> dataTransformationMode,
				DataTransformationOptionTargetInterval -> targetInterval,
				DataTransformationOptionNormalizationType -> normalizationType
			];
		
		initialTrainingSetRmse = CalculateMlrDataSetRmse[trainingSet, mlrInfo];
		If[isRegression,
			
			(* Regression*)
			If[Length[testSet] > 0,
				
				(* Regression WITH test set*)
				initialTestSetRmse = CalculateMlrDataSetRmse[testSet, mlrInfo];
				If[isIntermediateOutput,
					Print["initialTrainingSetRmse = ", initialTrainingSetRmse];
					Print["initialTestSetRmse     = ", initialTestSetRmse]
				];
				relevance = 
					{
						{0.0, initialTrainingSetRmse},
						{0.0, initialTestSetRmse},
						{}, 
						mlrInfo
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
						mlrInfo
					}
			],
			
			(* Classification *)
			initialTrainingSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[trainingSet, mlrInfo];
			If[Length[testSet] > 0,
				
				(* Classification WITH test set*)
				initialTestSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[testSet, mlrInfo];
				If[isIntermediateOutput,
					Print["initialTrainingSetCorrectClassificationInPercent = ", initialTrainingSetCorrectClassificationInPercent];
					Print["initialTestSetCorrectClassificationInPercent     = ", initialTestSetCorrectClassificationInPercent]
				];
				relevance = 
					{
						{0.0, initialTrainingSetCorrectClassificationInPercent},
						{0.0, initialTestSetCorrectClassificationInPercent},
						{}, 
						mlrInfo
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
						mlrInfo
					}
			]
		];	
		
		AppendTo[mlrInputComponentRelevanceList, relevance];
    
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
					mlrInfo = 
						FitMlr[
							trainingSet,
							MlrOptionDataTransformationMode -> dataTransformationMode,
							DataTransformationOptionTargetInterval -> targetInterval,
							DataTransformationOptionNormalizationType -> normalizationType
						];
					If[Length[testSet] > 0,
            
						testSetRmse = CalculateMlrDataSetRmse[testSet, mlrInfo];
						AppendTo[rmseList,{testSetRmse, i}],
          
						trainingSetRmse = CalculateMlrDataSetRmse[trainingSet, mlrInfo];
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
			mlrInfo = 
				FitMlr[
					trainingSet,
					MlrOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMlrDataSetRmse[trainingSet, mlrInfo];
					currentTestSetRmse = CalculateMlrDataSetRmse[testSet, mlrInfo];
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
							mlrInfo
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMlrDataSetRmse[trainingSet, mlrInfo];
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
							mlrInfo
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[trainingSet, mlrInfo];
					currentTestSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[testSet, mlrInfo];
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
							mlrInfo
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[trainingSet, mlrInfo];
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
							mlrInfo
						}
				]
			];	

			AppendTo[mlrInputComponentRelevanceList, relevance];
			removedInputComponentList = currentRemovedInputComponentList,
			
			{k, Length[numberOfExclusionsPerStepList]}
		];
		
		Return[mlrInputComponentRelevanceList]
	];
	
GetMlrInputRelevanceCalculationPC[

	(* Analyzes relevance of input components by successive leave-one-out for regression and classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mlrInputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlrInfo}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			dataTransformationMode,
			targetInterval,
			currentRemovedInputComponentList,
			i,
			k,
			numberOfInputs,
			numberOfRemovedInputs,
			mlrInputComponentRelevanceList,
	        mlrInfo,
			normalizationType,
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
			currentTrainingSetCorrectClassificationInPercent
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MLR options *)   
		dataTransformationMode = MlrOptionDataTransformationMode/.{opts}/.Options[MlrOptionsDataTransformation];
	    (* DataTransformation options *)   
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
	    (* Utility options *)   
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfExclusionsPerStepList = UtilityOptionExclusionsPerStep/.{opts}/.Options[UtilityOptionsExclusion];

		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		numberOfInputs = First[Dimensions[trainingAndTestSet[[1, 1, 1]] ]];
		If[Length[numberOfExclusionsPerStepList] == 0,
			numberOfExclusionsPerStepList = Table[1, {numberOfInputs - 1}];
		];				   
		removedInputComponentList = {};
		mlrInputComponentRelevanceList = {};
    
		(* Result for no removal *)
		trainingSet = trainingAndTestSet[[1]];
		testSet = trainingAndTestSet[[2]];
		mlrInfo = 
			FitMlr[
				trainingSet,
				MlrOptionDataTransformationMode -> dataTransformationMode,
				DataTransformationOptionTargetInterval -> targetInterval,
				DataTransformationOptionNormalizationType -> normalizationType
			];
		
		initialTrainingSetRmse = CalculateMlrDataSetRmse[trainingSet, mlrInfo];
		If[isRegression,
			
			(* Regression*)
			If[Length[testSet] > 0,
				
				(* Regression WITH test set*)
				initialTestSetRmse = CalculateMlrDataSetRmse[testSet, mlrInfo];
				If[isIntermediateOutput,
					Print["initialTrainingSetRmse = ", initialTrainingSetRmse];
					Print["initialTestSetRmse     = ", initialTestSetRmse]
				];
				relevance = 
					{
						{0.0, initialTrainingSetRmse},
						{0.0, initialTestSetRmse},
						{}, 
						mlrInfo
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
						mlrInfo
					}
			],
			
			(* Classification *)
			initialTrainingSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[trainingSet, mlrInfo];
			If[Length[testSet] > 0,
				
				(* Classification WITH test set*)
				initialTestSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[testSet, mlrInfo];
				If[isIntermediateOutput,
					Print["initialTrainingSetCorrectClassificationInPercent = ", initialTrainingSetCorrectClassificationInPercent];
					Print["initialTestSetCorrectClassificationInPercent     = ", initialTestSetCorrectClassificationInPercent]
				];
				relevance = 
					{
						{0.0, initialTrainingSetCorrectClassificationInPercent},
						{0.0, initialTestSetCorrectClassificationInPercent},
						{}, 
						mlrInfo
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
						mlrInfo
					}
			]
		];	
		
		AppendTo[mlrInputComponentRelevanceList, relevance];
    
   		ParallelNeeds[{"CIP`MLR`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[dataTransformationMode, targetInterval];
    
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
						mlrInfo = 
							FitMlr[
								trainingSet,
								MlrOptionDataTransformationMode -> dataTransformationMode,
								DataTransformationOptionTargetInterval -> targetInterval,
								DataTransformationOptionNormalizationType -> normalizationType
							];
						If[Length[testSet] > 0,
	            
							testSetRmse = CalculateMlrDataSetRmse[testSet, mlrInfo];
							{testSetRmse, i},
	          
							trainingSetRmse = CalculateMlrDataSetRmse[trainingSet, mlrInfo];
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
			mlrInfo = 
				FitMlr[
					trainingSet,
					MlrOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMlrDataSetRmse[trainingSet, mlrInfo];
					currentTestSetRmse = CalculateMlrDataSetRmse[testSet, mlrInfo];
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
							mlrInfo
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMlrDataSetRmse[trainingSet, mlrInfo];
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
							mlrInfo
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[trainingSet, mlrInfo];
					currentTestSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[testSet, mlrInfo];
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
							mlrInfo
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMlrCorrectClassificationInPercent[trainingSet, mlrInfo];
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
							mlrInfo
						}
				]
			];	

			AppendTo[mlrInputComponentRelevanceList, relevance];
			removedInputComponentList = currentRemovedInputComponentList,
			
			{k, Length[numberOfExclusionsPerStepList]}
		];
		
		Return[mlrInputComponentRelevanceList]
	];
	
GetMlrRegressionResult[
	
	(* Returns MLR regression result according to named property list.

	   Returns :
	   MLR regression result according to named property *)

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
    mlrInfo_,
	
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
	    
		pureFunction = Function[inputs, CalculateMlrOutputs[inputs, mlrInfo]];
	    Return[
	    	CIP`Graphics`GetSingleRegressionResult[
		    	namedProperty, 
		    	dataSet, 
		    	pureFunction,
		    	GraphicsOptionNumberOfIntervals -> numberOfIntervals
			]
		]
	];

GetMlrTrainOptimization[

	(* Returns training set optimization result for MLR training.

	   Returns:
	   mlrTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mlrInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mlrInfoList: List with mlrInfo
	   mlrInfoList[[i]] refers to optimization step i *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

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
			dataTransformationMode,
			targetInterval,

			clusterMethod,
			maximumNumberOfEpochs,
			scalarProductMinimumTreshold,
			maximumNumberOfTrialSteps,
			randomValueInitialization,

			deviationCalculationMethod,
			blackListLength,
			
			i,
			testSet,
			trainingSet,
			clusterRepresentativesRelatedIndexLists,
			trainingSetIndexList,
			testSetIndexList,
			indexLists,
			mlrInfo,
			normalizationType,
			trainingSetRMSE,
			testSetRMSE,
			pureOutputFunction,
			trainingSetRmseList,
			testSetRmseList,
			trainingAndTestSetList,
			mlrInfoList,
			selectionResult,
			blackList
		},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MLR options *)
	    dataTransformationMode = MlrOptionDataTransformationMode/.{opts}/.Options[MlrOptionsDataTransformation];
	    (* DataTransformation options *)   
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];

		(* Training set optimization options *)
	    deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];

		clusterRepresentativesRelatedIndexLists = 
			CIP`Cluster`GetClusterRepresentativesRelatedIndexLists[
				dataSet, 
				trainingFraction, 
				ClusterOptionMethod -> clusterMethod,
				ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				UtilityOptionRandomInitializationMode -> randomValueInitialization,
				DataTransformationOptionTargetInterval -> targetInterval
			];
		trainingSetIndexList = clusterRepresentativesRelatedIndexLists[[1]];
		testSetIndexList = clusterRepresentativesRelatedIndexLists[[2]];
		indexLists = clusterRepresentativesRelatedIndexLists[[3]];

		trainingSetRmseList = {};
		testSetRmseList = {};
		trainingAndTestSetList = {};
		mlrInfoList = {};
		blackList = {};
		Do[
			(* Fit training set and evaluate RMSE *)
			trainingSet = CIP`DataTransformation`GetDataSetPart[dataSet, trainingSetIndexList];
			testSet = CIP`DataTransformation`GetDataSetPart[dataSet, testSetIndexList];
			mlrInfo = 
				FitMlr[
					trainingSet,
					MlrOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType
				];
			trainingSetRMSE = CalculateMlrDataSetRmse[trainingSet, mlrInfo];
			testSetRMSE = CalculateMlrDataSetRmse[testSet, mlrInfo];

			(* Set iteration results *)
			AppendTo[trainingSetRmseList, {N[i], trainingSetRMSE}];
			AppendTo[testSetRmseList, {N[i], testSetRMSE}];
			AppendTo[trainingAndTestSetList, {trainingSet, testSet}];
			AppendTo[mlrInfoList, mlrInfo];
			
			(* Break if necessary *)
			If[i == numberOfTrainingSetOptimizationSteps,
				Break[]
			];

			(* Select new training and test set index lists *)
			pureOutputFunction = Function[input, CalculateMlrOutput[input, mlrInfo]];
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
				mlrInfoList
			}
		]
	];

ScanClassTrainingWithMlr[

	(* Scans training and test set for different training fractions based on method FitMlr, see code.
	
	   Returns:
	   mlrClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlrInfo1}, {trainingAndTestSet2, mlrInfo2}, ...}
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

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
	    	dataTransformationMode,
	    	targetInterval,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,

			numberOfTrainingSetOptimizationSteps,
			deviationCalculationMethod,
			blackListLength,
			
			parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MLR options *)
		dataTransformationMode = MlrOptionDataTransformationMode/.{opts}/.Options[MlrOptionsDataTransformation];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];

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
				ScanClassTrainingWithMlrPC[
					classificationDataSet,
					trainingFractionList,
					MlrOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					ClusterOptionMethod -> clusterMethod,
					ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
					ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
					ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
					UtilityOptionBlackListLength -> blackListLength
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				ScanClassTrainingWithMlrSC[
					classificationDataSet,
					trainingFractionList,
					MlrOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					ClusterOptionMethod -> clusterMethod,
					ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
					ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
					ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
					UtilityOptionBlackListLength -> blackListLength
				]
			]
		]
	];

ScanClassTrainingWithMlrSC[

	(* Scans training and test set for different training fractions based on method FitMlr, see code.
	
	   Returns:
	   mlrClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlrInfo1}, {trainingAndTestSet2, mlrInfo2}, ...}
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

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
	    	dataTransformationMode,
	    	targetInterval,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,

			numberOfTrainingSetOptimizationSteps,
			deviationCalculationMethod,
			blackListLength,

			i,
			scanReport,
			trainingAndTestSetsInfo,
			currentTrainingAndTestSet,
			currentTrainingSet,
			currentTestSet,
			currentMlrInfo,
			pureFunction,
			trainingSetCorrectClassificationInPercent,
			testSetCorrectClassificationInPercent,
			mlrTrainOptimization,
			trainingAndTestSetList,
			mlrInfoList,
			normalizationType,
			bestIndex
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MLR options *)
		dataTransformationMode = MlrOptionDataTransformationMode/.{opts}/.Options[MlrOptionsDataTransformation];
	    (* DataTransformation options *)   
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];

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
				mlrTrainOptimization = 
					GetMlrTrainOptimization[
						classificationDataSet, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						MlrOptionDataTransformationMode -> dataTransformationMode,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						DataTransformationOptionTargetInterval -> targetInterval,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestMlrClassOptimization[mlrTrainOptimization];
				trainingAndTestSetList = mlrTrainOptimization[[3]];
				mlrInfoList = mlrTrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlrInfo = mlrInfoList[[bestIndex]],
				
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
						DataTransformationOptionTargetInterval -> targetInterval
				];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlrInfo = 
					FitMlr[
						currentTrainingSet,
						MlrOptionDataTransformationMode -> dataTransformationMode,
						DataTransformationOptionTargetInterval -> targetInterval,
						DataTransformationOptionNormalizationType -> normalizationType
					]
			];
			
			pureFunction = Function[inputs, CalculateMlrClassNumbers[inputs, currentMlrInfo]];
			trainingSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTrainingSet, pureFunction];
			testSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTestSet, pureFunction];
			AppendTo[trainingAndTestSetsInfo, {currentTrainingAndTestSet, currentMlrInfo}];
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
	
ScanClassTrainingWithMlrPC[

	(* Scans training and test set for different training fractions based on method FitMlr, see code.
	
	   Returns:
	   mlrClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlrInfo1}, {trainingAndTestSet2, mlrInfo2}, ...}
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

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
	    	dataTransformationMode,
	    	targetInterval,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,

			numberOfTrainingSetOptimizationSteps,
			deviationCalculationMethod,
			blackListLength,
			
			i,
			scanReport,
			trainingAndTestSetsInfo,
			currentTrainingAndTestSet,
			currentTrainingSet,
			currentTestSet,
			currentMlrInfo,
			pureFunction,
			trainingSetCorrectClassificationInPercent,
			testSetCorrectClassificationInPercent,
			mlrTrainOptimization,
			trainingAndTestSetList,
			mlrInfoList,
			normalizationType,
			bestIndex,
			listOfTrainingAndTestSetsInfoAndScanReport
			
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MLR options *)
		dataTransformationMode = MlrOptionDataTransformationMode/.{opts}/.Options[MlrOptionsDataTransformation];
	    (* DataTransformation options *)   
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];

		(* Training set optimization options *)
		numberOfTrainingSetOptimizationSteps = UtilityOptionOptimizationSteps/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];

		ParallelNeeds[{"CIP`MLR`", "CIP`Cluster`", "CIP`DataTransformation`", "CIP`Utility`", "Combinatorica`"}];
		DistributeDefinitions[numberOfTrainingSetOptimizationSteps, dataTransformationMode, clusterMethod, maximumNumberOfEpochs, scalarProductMinimumTreshold, 
			maximumNumberOfTrialSteps, targetInterval, randomValueInitialization, deviationCalculationMethod, blackListLength];
		
		listOfTrainingAndTestSetsInfoAndScanReport = ParallelTable[
			If[numberOfTrainingSetOptimizationSteps > 0,
			
				(* ------------------------------------------------------------------------------------------------------ *)
				(* Training set optimization *)
				mlrTrainOptimization = 
					GetMlrTrainOptimization[
						classificationDataSet, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						MlrOptionDataTransformationMode -> dataTransformationMode,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						DataTransformationOptionTargetInterval -> targetInterval,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				
				bestIndex = GetBestMlrClassOptimization[mlrTrainOptimization];				
				trainingAndTestSetList = mlrTrainOptimization[[3]];
				mlrInfoList = mlrTrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlrInfo = mlrInfoList[[bestIndex]],
				
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
						DataTransformationOptionTargetInterval -> targetInterval
				];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlrInfo = 
					FitMlr[
						currentTrainingSet,
						MlrOptionDataTransformationMode -> dataTransformationMode,
						DataTransformationOptionTargetInterval -> targetInterval,
						DataTransformationOptionNormalizationType -> normalizationType
					]
				];
			
			pureFunction = Function[inputs, CalculateMlrClassNumbers[inputs, currentMlrInfo]];
			trainingSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTrainingSet, pureFunction];
			testSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTestSet, pureFunction];
			{
			 	{currentTrainingAndTestSet, currentMlrInfo},
			
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

ScanRegressTrainingWithMlr[

	(* Scans training and test set for different training fractions based on method FitMlr, see code.
	
	   Returns:
	   mlrRegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlrInfo1}, {trainingAndTestSet2, mlrInfo2}, ...}
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

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
	    	dataTransformationMode,
	    	targetInterval,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,

			numberOfTrainingSetOptimizationSteps,
			deviationCalculationMethod,
			blackListLength,
			
			parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MLR options *)
		dataTransformationMode = MlrOptionDataTransformationMode/.{opts}/.Options[MlrOptionsDataTransformation];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];

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
			    ScanRegressTrainingWithMlrPC[
		    		dataSet,
		    		trainingFractionList,
		    		MlrOptionDataTransformationMode -> dataTransformationMode,
		    		DataTransformationOptionTargetInterval -> targetInterval,
		    		ClusterOptionMethod -> clusterMethod,
		    		ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
		    		ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
		    		ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
		    		UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    		UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
		    		UtilityOptionDeviationCalculation -> deviationCalculationMethod,
		    		UtilityOptionBlackListLength -> blackListLength
			    ]
			],
			
	    	(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
			    ScanRegressTrainingWithMlrSC[
		    		dataSet,
		    		trainingFractionList,
		    		MlrOptionDataTransformationMode -> dataTransformationMode,
		    		DataTransformationOptionTargetInterval -> targetInterval,
		    		ClusterOptionMethod -> clusterMethod,
		    		ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
		    		ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
		    		ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
		    		UtilityOptionRandomInitializationMode -> randomValueInitialization,
		    		UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
		    		UtilityOptionDeviationCalculation -> deviationCalculationMethod,
		    		UtilityOptionBlackListLength -> blackListLength
			    ]
			]
	    ]
	];

ScanRegressTrainingWithMlrSC[

	(* Scans training and test set for different training fractions based on method FitMlr, see code.
	
	   Returns:
	   mlrRegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlrInfo1}, {trainingAndTestSet2, mlrInfo2}, ...}
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

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
	    	dataTransformationMode,
	    	targetInterval,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,

			numberOfTrainingSetOptimizationSteps,
			deviationCalculationMethod,
			blackListLength,

			i,
			scanReport,
			trainingAndTestSetsInfo,
			currentTrainingAndTestSet,
			currentTrainingSet,
			currentTestSet,
			currentMlrInfo,
			pureFunction,
			trainingSetRMSE,
			testSetRMSE,
			mlrTrainOptimization,
			trainingAndTestSetList,
			mlrInfoList,
			normalizationType,
			bestIndex
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MLR options *)
		dataTransformationMode = MlrOptionDataTransformationMode/.{opts}/.Options[MlrOptionsDataTransformation];
	    (* DataTransformation options *)   
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];

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
				mlrTrainOptimization = 
					GetMlrTrainOptimization[
						dataSet, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						MlrOptionDataTransformationMode -> dataTransformationMode,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						DataTransformationOptionTargetInterval -> targetInterval,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestMlrRegressOptimization[mlrTrainOptimization];
				trainingAndTestSetList = mlrTrainOptimization[[3]];
				mlrInfoList = mlrTrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlrInfo = mlrInfoList[[bestIndex]],
				
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
						DataTransformationOptionTargetInterval -> targetInterval
				];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlrInfo = 
					FitMlr[
						currentTrainingSet,
						MlrOptionDataTransformationMode -> dataTransformationMode,
						DataTransformationOptionTargetInterval -> targetInterval,
						DataTransformationOptionNormalizationType -> normalizationType
					]
			];
			
			pureFunction = Function[inputs, CalculateMlrOutputs[inputs, currentMlrInfo]];
			trainingSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTrainingSet, pureFunction]];
			testSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTestSet, pureFunction]];
			AppendTo[trainingAndTestSetsInfo, {currentTrainingAndTestSet, currentMlrInfo}];
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

ScanRegressTrainingWithMlrPC[

	(* Scans training and test set for different training fractions based on method FitMlr, see code.
	
	   Returns:
	   mlrRegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlrInfo1}, {trainingAndTestSet2, mlrInfo2}, ...}
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

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
	    	dataTransformationMode,
	    	targetInterval,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,

			numberOfTrainingSetOptimizationSteps,
			deviationCalculationMethod,
			blackListLength,

			i,
			scanReport,
			trainingAndTestSetsInfo,
			currentTrainingAndTestSet,
			currentTrainingSet,
			currentTestSet,
			currentMlrInfo,
			pureFunction,
			trainingSetRMSE,
			testSetRMSE,
			mlrTrainOptimization,
			trainingAndTestSetList,
			mlrInfoList,
			normalizationType,
			bestIndex,
			listOfTrainingAndTestSetsInfoAndScanReport
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MLR options *)
		dataTransformationMode = MlrOptionDataTransformationMode/.{opts}/.Options[MlrOptionsDataTransformation];
	    (* DataTransformation options *)   
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];

		(* Training set optimization options *)
		numberOfTrainingSetOptimizationSteps = UtilityOptionOptimizationSteps/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];

		ParallelNeeds[{"CIP`MLR`", "CIP`Cluster`", "CIP`DataTransformation`", "CIP`Utility`", "Combinatorica`"}];
		DistributeDefinitions[numberOfTrainingSetOptimizationSteps, dataTransformationMode, clusterMethod, maximumNumberOfEpochs, scalarProductMinimumTreshold, 
			maximumNumberOfTrialSteps, targetInterval, randomValueInitialization, deviationCalculationMethod, blackListLength];
		
		listOfTrainingAndTestSetsInfoAndScanReport = ParallelTable[
			If[numberOfTrainingSetOptimizationSteps > 0,
				
				(* ------------------------------------------------------------------------------------------------------ *)
				(* Training set optimization *)
				mlrTrainOptimization = 
					GetMlrTrainOptimization[
						dataSet, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						MlrOptionDataTransformationMode -> dataTransformationMode,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						DataTransformationOptionTargetInterval -> targetInterval,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestMlrRegressOptimization[mlrTrainOptimization];
				trainingAndTestSetList = mlrTrainOptimization[[3]];
				mlrInfoList = mlrTrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlrInfo = mlrInfoList[[bestIndex]],
				
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
						DataTransformationOptionTargetInterval -> targetInterval
				];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMlrInfo = 
					FitMlr[
						currentTrainingSet,
						MlrOptionDataTransformationMode -> dataTransformationMode,
						DataTransformationOptionTargetInterval -> targetInterval,
						DataTransformationOptionNormalizationType -> normalizationType
					]
			];
			
			pureFunction = Function[inputs, CalculateMlrOutputs[inputs, currentMlrInfo]];
			trainingSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTrainingSet, pureFunction]];
			testSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTestSet, pureFunction]];
			{
				{currentTrainingAndTestSet, currentMlrInfo},
			 
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

ShowMlrOutput3D[

	(* Shows 3D MLR output.

	   Returns: Graphics3D *)


    (* Index of input component that receives argumentValue1 *)
    indexOfInput1_?IntegerQ,

    (* Index of input component that receives argumentValue2 *)
    indexOfInput2_?IntegerQ,

    (* Index of output component that returns function value *)
    indexOfOutput_?IntegerQ,
    
    (* Input in original units: 
       inputsInOriginalUnits = {inputValue1, inputValue2, ...} 
       Values of input components with specified indices (indexOfInput1, indexOfInput2) are replaced by argument values *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mlrInfo_,
    
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
			inputsMinMaxList,
			labels,
			dataSetScaleInfo,
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

		dataSetScaleInfo = mlrInfo[[2]];
		inputsMinMaxList = dataSetScaleInfo[[1, 1]];
		x1Min = inputsMinMaxList[[indexOfInput1, 1]];
		x1Max = inputsMinMaxList[[indexOfInput1, 2]];
		x2Min = inputsMinMaxList[[indexOfInput2, 1]];
		x2Max = inputsMinMaxList[[indexOfInput2, 2]];
		labels = 
			{
				StringJoin["Input ", ToString[indexOfInput1]],
				StringJoin["Input ", ToString[indexOfInput2]],
				StringJoin["Output ", ToString[indexOfOutput]]
			};
		
		Return[
			CIP`Graphics`PlotFunction3D[
				Function[{x1, x2}, CalculateMlrValue3D[x1, x2, indexOfInput1, indexOfInput2, indexOfOutput, input, mlrInfo]], 
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

ShowMlrClassificationResult[

	(* Shows result of MLR classification for training and test set according to named property list.

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
    
    (* {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...}
	   testSet has the same structure and restrictions as trainingSet 
	   NOTE: Training and test set MUST be in original units *)
    trainingAndTestSet_,
    
  	(* See "Frequently used data structures" *)
    mlrInfo_,
    
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
		ShowMlrSingleClassification[
			namedPropertyList,
			trainingSet, 
			mlrInfo,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionMinMaxIndex -> minMaxIndex
		];
		
		(* Analyze test set *)
		If[Length[testSet] > 0,
			Print["Test Set:"];
			ShowMlrSingleClassification[
				namedPropertyList,
				testSet, 
				mlrInfo,
				GraphicsOptionImageSize -> imageSize,
				GraphicsOptionMinMaxIndex -> minMaxIndex
			];
		]
	];

ShowMlrSingleClassification[

	(* Shows result of MLR classification for data set according to named property list.

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

    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...}
	   NOTE: Data set MUST be in original units *)
    dataSet_,
    
  	(* See "Frequently used data structures" *)
    mlrInfo_,
    
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

   		pureFunction = Function[inputs, CalculateMlrClassNumbers[inputs, mlrInfo]];
		CIP`Graphics`ShowClassificationResult[
			namedPropertyList,
			dataSet, 
			pureFunction,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionMinMaxIndex -> minMaxIndex
		]
	];

ShowMlrClassificationScan[

	(* Shows result of MLR based classification scan of clustered training sets.

	   Returns: Nothing *)


	(* mlrClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlrInfo1}, {trainingAndTestSet2, mlrInfo2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, classification result in percent for training set}, {trainingFraction, classification result in percent for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)
	mlrClassificationScan_,
	
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
			mlrClassificationScan,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMlrInputRelevanceClass[

	(* Shows mlrInputComponentRelevanceListForClassification.

	   Returns: Nothing *)


	(* mlrInputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlrInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) correct classification in percent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best correct classification in percent of test set} *)
	mlrInputComponentRelevanceListForClassification_,
	
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
			mlrInputComponentRelevanceListForClassification,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMlrInputRelevanceRegress[

	(* Shows mlrInputComponentRelevanceListForRegression.

	   Returns: Nothing *)


	(* mlrInputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mlrInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)
	mlrInputComponentRelevanceListForRegression_,
	
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
			mlrInputComponentRelevanceListForRegression,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMlrRegressionResult[

	(* Shows result of MLR regression for training and test set according to named property list.
	
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

    (* {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...}
	   testSet has the same structure and restrictions as trainingSet
	   NOTE: Training and test set MUST be in original units *)
    trainingAndTestSet_,
    
  	(* See "Frequently used data structures" *)
    mlrInfo_,
    
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
		ShowMlrSingleRegression[
			namedPropertyList,
			trainingSet, 
			mlrInfo,
			GraphicsOptionPointSize -> pointSize,
			GraphicsOptionPointColor -> pointColor
		];
		
		(* Analyze test set *)
		If[Length[testSet] > 0,
			Print["Test Set:"];
			ShowMlrSingleRegression[
				namedPropertyList,
				testSet, 
				mlrInfo,
				GraphicsOptionPointSize -> pointSize,
				GraphicsOptionPointColor -> pointColor
			]
		]
	];

ShowMlrSingleRegression[
    
	(* Shows result of MLR regression for data set according to named property list.
	
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
    mlrInfo_,
    
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

		pureFunction = Function[inputs, CalculateMlrOutputs[inputs, mlrInfo]];
		CIP`Graphics`ShowRegressionResult[
			namedPropertyList,
			dataSet, 
			pureFunction,
			GraphicsOptionPointSize -> pointSize,
			GraphicsOptionPointColor -> pointColor
		]
	];

ShowMlrRegressionScan[

	(* Shows result of MLR based regression scan of clustered training sets.

	   Returns: Nothing *)


	(* mlrRegressionScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mlrInfo1}, {trainingAndTestSet2, mlrInfo2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, RMSE for training set}, {trainingFraction, RMSE for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)
	mlrRegressionScan_,
	
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
			mlrRegressionScan,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMlrTrainOptimization[

	(* Shows training set optimization result of MLR.

	   Returns: Nothing *)


	(* mlrTrainOptimization = {trainingSetRmseList, testSetRmseList, not interesting, not interesting}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set} *)
	mlrTrainOptimization_,
    
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
			mlrTrainOptimization, 
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

(* ::Section:: *)
(* End of Package *)

End[]

EndPackage[]
