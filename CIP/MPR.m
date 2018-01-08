(*
----------------------------------------------------------------------------------------
Computational Intelligence Packages (CIP): Package Multiple Polynomial Regression (MPR)
Version 3.0 for Mathematica 11 or higher
----------------------------------------------------------------------------------------

Authors: Kolja Berger (parallelization for CIP 2.0), Achim Zielesny 

GNWI - Gesellschaft fuer naturwissenschaftliche Informatik mbH, 
Oer-Erkenschwick, Germany

Citation:
Achim Zielesny, Computational Intelligence Packages (CIP), Version 3.0, 
GNWI mbH (http://www.gnwi.de), Oer-Erkenschwick, Germany, 2018.

Copyright 2018 Achim Zielesny

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
mprInfo: {mprFitResult, dataSetScaleInfo, dataTransformationMode, outputOffsets, normalizationInfo}

	mprFitResult: See code of FitMpr[] and signature of Fit[]
	dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs}, see CIP`DataTransformation`GetDataSetScaleInfo 
	dataTransformationMode <> "None": All values of dataSet are internally transformed by Log/Sqrt operation, "None": Otherwise 
	outputOffsets: Offset value for transformations of outputs
	normalizationInfo: {normalizationType, meanAndStandardDeviationList}, see CIP`DataTransformation`GetDataMatrixNormalizationInfo
-----------------------------------------------------------------------
*)

(* ::Section:: *)
(* Package and dependencies *)

BeginPackage["CIP`MPR`", {"CIP`Utility`", "CIP`Graphics`", "CIP`DataTransformation`", "CIP`Cluster`"}]

(* ::Section:: *)
(* Off settings *)

Off[General::"spell1"]
Off[General::shdw]
Off[General::compat]

(* ::Section:: *)
(* Options *)

Options[MprOptionsDataTransformation] = 
{
	(* Data transformation mode: "None", "Log", "Sqrt" *)
    MprOptionDataTransformationMode -> "None"
}

(* ::Section:: *)
(* Declarations *)

CalculateMprValue2D::usage = 
	"CalculateMprValue2D[argumentValue, indexOfInput, indexOfOutput, input, mprInfo]"

CalculateMprValue3D::usage = 
	"CalculateMprValue3D[argumentValue1, argumentValue2, indexOfInput1, indexOfInput2, indexOfOutput, input, mprInfo]"

CalculateMprClassNumber::usage = 
	"CalculateMprClassNumber[input, mprInfo]"

CalculateMprClassNumbers::usage = 
	"CalculateMprClassNumbers[inputs, mprInfo]"

CalculateMprDataSetRmse::usage = 
	"CalculateMprDataSetRmse[dataSet, mprInfo]"

CalculateMprOutput::usage = 
	"CalculateMprOutput[input, mprInfo]"

CalculateMprOutputs::usage = 
	"CalculateMprOutputs[inputs, mprInfo]"

FitMpr::usage = 
	"FitMpr[dataSet, polynomialDegree, options]"

FitMprSeries::usage = 
	"FitMprSeries[dataSet, polynomialDegreeList, options]"

GetBestMprClassOptimization::usage = 
	"GetBestMprClassOptimization[mprTrainOptimization, options]"

GetBestMprRegressOptimization::usage = 
	"GetBestMprRegressOptimization[mprTrainOptimization, options]"
	
GetMprInputInclusionClass::usage = 
	"GetMprInputInclusionClass[trainingAndTestSet, polynomialDegree, options]"

GetMprInputInclusionRegress::usage = 
	"GetMprInputInclusionRegress[trainingAndTestSet, polynomialDegree, options]"

GetMprInputRelevanceClass::usage = 
	"GetMprInputRelevanceClass[trainingAndTestSet, polynomialDegree, options]"

GetMprClassRelevantComponents::usage = 
    "GetMprClassRelevantComponents[mprInputComponentRelevanceListForClassification, numberOfComponents]"

GetMprInputRelevanceRegress::usage = 
	"GetMprInputRelevanceRegress[trainingAndTestSet, polynomialDegree, options]"

GetMprRegressRelevantComponents::usage = 
    "GetMprRegressRelevantComponents[mprInputComponentRelevanceListForRegression, numberOfComponents]"

GetMprNumberOfParameters::usage = 
	"GetMprNumberOfParameters[dataSet, polynomialDegree]"

GetMprRegressionResult::usage = 
	"GetMprRegressionResult[namedProperty, dataSet, mprInfo, options]"

GetMprSeriesClassificationResult::usage = 
	"GetMprSeriesClassificationResult[trainingAndTestSet, mprInfoList]"

GetMprSeriesRmse::usage = 
	"GetMprSeriesRmse[trainingAndTestSet, mprInfoList]"

GetMprTrainOptimization::usage = 
	"GetMprTrainOptimization[dataSet, polynomialDegree, trainingFraction, numberOfTrainingSetOptimizationSteps, options]"

ScanClassTrainingWithMpr::usage = 
	"ScanClassTrainingWithMpr[dataSet, polynomialDegree, trainingFractionList, options]"

ScanRegressTrainingWithMpr::usage = 
	"ScanRegressTrainingWithMpr[dataSet, polynomialDegree, trainingFractionList, options]"

ShowMprOutput3D::usage = 
	"ShowMprOutput3D[indexOfInput1, indexOfInput2, indexOfOutput, input, mprInfo, graphicsOptions, displayFunction]"

ShowMprClassificationResult::usage = 
	"ShowMprClassificationResult[namedPropertyList, trainingAndTestSet, mprInfo]"

ShowMprSingleClassification::usage = 
	"ShowMprSingleClassification[namedPropertyList, dataSet, mprInfo]"

ShowMprClassificationScan::usage = 
	"ShowMprClassificationScan[mprClassificationScan, options]"

ShowMprInputRelevanceClass::usage = 
	"ShowMprInputRelevanceClass[mprInputComponentRelevanceListForClassification, options]"
	
ShowMprInputRelevanceRegress::usage = 
	"ShowMprInputRelevanceRegress[mprInputComponentRelevanceListForRegression, options]"

ShowMprRegressionResult::usage = 
	"ShowMprRegressionResult[namedPropertyList, trainingAndTestSet, mprInfo, options]"

ShowMprSingleRegression::usage = 
	"ShowMprSingleRegression[namedPropertyList, dataSet, mprInfo, options]"

ShowMprRegressionScan::usage = 
	"ShowMprRegressionScan[mprRegressionScan, options]"

ShowMprSeriesClassificationResult::usage = 
	"ShowMprSeriesClassificationResult[mprSeriesClassificationResult, options]"

ShowMprSeriesRmse::usage = 
	"ShowMprSeriesRmse[mprSeriesRmse, options]"

ShowMprTrainOptimization::usage = 
	"ShowMprTrainOptimization[mprTrainOptimization, options]"
	
(* ::Section:: *)
(* Functions *)

Begin["`Private`"]

CalculateMprValue2D[

	(* Calculates 2D output for specified argument and input for specified MPR.
	   This special method assumes an input and an output with one component only.

	   Returns:
	   Value of specified output component for argument *)


    (* Argument value for input component with index indexOfInput *)
    argumentValue_?NumberQ,
    
  	(* See "Frequently used data structures" *)
    mprInfo_
    
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
			CalculateMprValue2D[argumentValue, indexOfInput, indexOfFunctionValueOutput, input, mprInfo]
		]
	];

CalculateMprValue2D[

	(* Calculates 2D output for specified argument and input for specified MPR.

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
    mprInfo_
    
	] :=
  
	Module[
    
		{
			currentInput,
			output
		},
		
		currentInput = ReplacePart[input, {indexOfInput -> argumentValue}];
		output = CalculateMprOutput[currentInput, mprInfo];
		Return[output[[indexOfOutput]]];
	];

CalculateMprValue3D[

	(* Calculates 3D output for specified arguments for specified MPR. 
	   This specific methods assumes a MPR with input vector of length 2 and an output vector of length 1.

	   Returns:
	   Value of specified output component for input *)


    (* Argument value for input component with index indexOfInput1 *)
    argumentValue1_?NumberQ,
    
    (* Argument value for input component with index indexOfInput2 *)
    argumentValue2_?NumberQ,
    
  	(* See "Frequently used data structures" *)
    mprInfo_
    
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
			CalculateMprValue3D[argumentValue1, argumentValue2, indexOfInput1, indexOfInput2, indexOfOutput, input, mprInfo]
		];
	];

CalculateMprValue3D[

	(* Calculates 3D output for specified arguments and input for specified MPR.

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
    mprInfo_
    
	] :=
  
	Module[
    
		{
			currentInput,
			output
		},
		
		currentInput = ReplacePart[input, {indexOfInput1 -> argumentValue1, indexOfInput2 -> argumentValue2}];
		output = CalculateMprOutput[currentInput, mprInfo];
		Return[output[[indexOfOutput]]];
	];

CalculateMprClassNumber[

	(* Returns class number for specified input for classification MPR.

	   Returns:
	   Class number of input *)

    
    (* Input in original units: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mprInfo_
    
	] :=
  
	Module[
    
		{
			dataSetScaleInfo,
			mprFitResult,
			normalizationInfo,
			scaledInput,
			scaledOutput
		},
    
		mprFitResult = mprInfo[[1]];
    	dataSetScaleInfo = mprInfo[[2]];
    	normalizationInfo = mprInfo[[5]];

		scaledInput = First[CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo]];
		scaledOutput = GetInternalMprOutput[scaledInput, mprFitResult];
		Return[CIP`Utility`GetPositionOfMaximumValue[scaledOutput]]
	];

CalculateMprClassNumbers[

	(* Returns class numbers for specified inputs for classification MPR.

	   Returns:
	   {class number of input1, class number of input2, ...} *)

    
    (* {inputsInOriginalUnit1, inputsInOriginalUnit2, ...}
        inputsInOriginalUnit: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    inputs_/;MatrixQ[inputs, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mprInfo_
    
	] :=
  
	Module[
    
		{
			i,
			dataSetScaleInfo,
			mprFitResult,
			normalizationInfo,
			scaledInputs,
			scaledOutputs
		},

		mprFitResult = mprInfo[[1]];
    	dataSetScaleInfo = mprInfo[[2]];
    	normalizationInfo = mprInfo[[5]];

		scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
		scaledOutputs = GetInternalMprOutputs[scaledInputs, mprFitResult];
		Return[
			Table[
				CIP`Utility`GetPositionOfMaximumValue[scaledOutputs[[i]]],
				
				{i, Length[scaledOutputs]}
			]
		]
	];

CalculateMprCorrectClassificationInPercent[

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
    mprInfo_
    
	] :=
  
	Module[
    
		{
			pureFunction
		},

		pureFunction = Function[inputs, CalculateMprClassNumbers[inputs, mprInfo]];
		Return[CIP`Utility`GetCorrectClassificationInPercent[classificationDataSet, pureFunction]]
	];

CalculateMprDataSetRmse[

	(* Returns RMSE of data set.

	   Returns: 
	   RMSE of data set *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

  	(* See "Frequently used data structures" *)
    mprInfo_
    
	] :=
  
	Module[
    
		{
			pureFunction,
			rmse
		},

		pureFunction = Function[inputs, CalculateMprOutputs[inputs, mprInfo]];
		rmse = Sqrt[CIP`Utility`GetMeanSquaredError[dataSet, pureFunction]];
		Return[rmse]
	];

CalculateMprOutput[

	(* Calculates output for specified input for MPR.

	   Returns:
	   output: {transformedOutputValue1, transformedOutputValue2, ...} *)

    
    (* Input in original units: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    input_/;VectorQ[input, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mprInfo_
    
	] :=
  
	Module[
    
		{
			dataSetScaleInfo,
			i,
			k,
			dataTransformationMode,
			mprFitResult,
			normalizationInfo,
			outputOffsets,
			outputsInOriginalUnits,
			scaledInput,
			scaledOutput,
			unscaledOutputs
		},
    
		mprFitResult = mprInfo[[1]];
    	dataSetScaleInfo = mprInfo[[2]];
    	dataTransformationMode = mprInfo[[3]];
    	outputOffsets = mprInfo[[4]];
    	normalizationInfo = mprInfo[[5]];

		scaledInput = First[CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo]];
		scaledOutput = GetInternalMprOutput[scaledInput, mprFitResult];

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

CalculateMprOutputs[

	(* Calculates outputs for specified inputs for MPR.

	   Returns:
	   outputs = {output1, ..., output<Length[inputs]>}
	   output: {transformedOutputValue1, transformedOutputValue2, ...} *)

    
    (* {inputsInOriginalUnit1, inputsInOriginalUnit2, ...}
        inputsInOriginalUnit: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    inputs_/;MatrixQ[inputs, NumberQ],
    
  	(* See "Frequently used data structures" *)
    mprInfo_
    
	] :=
  
	Module[
    
		{
			dataSetScaleInfo,
			i,
			dataTransformationMode,
			mprFitResult,
			normalizationInfo,
			outputOffsets,
			outputsInOriginalUnits,
			scaledInputs,
			scaledOutputs,
			unscaledOutputs
		},
    
		mprFitResult = mprInfo[[1]];
    	dataSetScaleInfo = mprInfo[[2]];
    	dataTransformationMode = mprInfo[[3]];
    	outputOffsets = mprInfo[[4]];
    	normalizationInfo = mprInfo[[5]];

		scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
		scaledOutputs = GetInternalMprOutputs[scaledInputs, mprFitResult];

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

FitMpr[

	(* Trains with MPR

	   Returns: 
	   mprInfo (see "Frequently used data structures") *)
	   
	   
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* Order of polynomial to be fit (= 1: MPR) *)
	polynomialDegree_?IntegerQ,
		
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
			mprFitResult,
			functionVector,
			i,
			k,
			mprInputDataList,
			normalizationInfo,
			normalizationType,
			numberOfInputVariables,
			numberOfOutputVariables,
			outputOffsets,
			scaledDataSet,
			variableVector,
			targetInterval,
			argumentForTable,
			index
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
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
		
		(* Clear mprVariable variable. NOTE: This is NOT a local variable *)
		Clear[mprVariable];

		(* Create subscripted mprVariable variable vector and function vector. NOTE : These are NOT local variables *)
		variableVector = Table[Subscript[mprVariable, i], {i, 1, numberOfInputVariables}];
		functionVector = {1.0};
		Do[
			argumentForTable ={Product[Subscript[mprVariable, Subscript[index, i]], {i, 1, k}]};
			Do[
				AppendTo[argumentForTable, {Subscript[index, i], 1, numberOfInputVariables}],
				{i, 1, polynomialDegree}
			];		  
			functionVector = 
				Flatten[
					{
						functionVector,
						DeleteDuplicates[Flatten[
							Apply[Table, argumentForTable]
						]]
					}
				],
			{k, 1, polynomialDegree}
		];

		(* Transform data for Fit[] (see signature of Fit) *)
		mprInputDataList = 
			Table[
				Table[
					ioPair = scaledDataSet[[k]];
					AppendTo[ioPair[[1]], ioPair[[2, i]]],
					
					{k, Length[scaledDataSet]}
				],
				
				{i, numberOfOutputVariables}
			];

		(* Fit data *)
		mprFitResult = 
			Table[
				Fit[mprInputDataList[[i]], functionVector, variableVector], 
					
				{i, numberOfOutputVariables}
			];
		
		(* ----------------------------------------------------------------------------------------------------
		   Return mprInfo
		   ---------------------------------------------------------------------------------------------------- *)
    	Return[{mprFitResult, correctedDataSetScaleInfo, dataTransformationMode, outputOffsets, normalizationInfo}]
	];

FitMprSeries[

	(* Fits of a series of MPRs.

	   Returns:
	   mprInfoList: {mprInfo1, mprInfo2, ...}
	   mprInfo[[i]] corresponds to polynomialDegreeList[[i]]
	   (see "Frequently used data structures") *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* List with polynomial degrees *)
	polynomialDegreeList_/;VectorQ[polynomialDegreeList, IntegerQ],

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			dataTransformationMode,
			parallelization,
			targetInterval
		},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    (* Parallelization options *)
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				FitMprSeriesPC[
					dataSet,
					polynomialDegreeList,
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				FitMprSeriesSC[
					dataSet,
					polynomialDegreeList,
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval
				]
			]
		]
	];

FitMprSeriesSC[

	(* Fits of a series of MPRs.

	   Returns:
	   mprInfoList: {mprInfo1, mprInfo2, ...}
	   mprInfo[[i]] corresponds to polynomialDegreeList[[i]]
	   (see "Frequently used data structures") *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* List with polynomial degrees *)
	polynomialDegreeList_/;VectorQ[polynomialDegreeList, IntegerQ],

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			i,
			dataTransformationMode,
			normalizationType,
			targetInterval
		},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		Return[
			Table[
				FitMpr[
					dataSet,
					polynomialDegreeList[[i]],
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType
				],
				
				{i, Length[polynomialDegreeList]}
			]			
		]
	];

FitMprSeriesPC[

	(* Fits of a series of MPRs.

	   Returns:
	   mprInfoList: {mprInfo1, mprInfo2, ...}
	   mprInfo[[i]] corresponds to polynomialDegreeList[[i]]
	   (see "Frequently used data structures") *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* List with polynomial degrees *)
	polynomialDegreeList_/;VectorQ[polynomialDegreeList, IntegerQ],

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			i,
			dataTransformationMode,
			normalizationType,
			targetInterval
		},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		ParallelNeeds[{"CIP`MPR`", "CIP`DataTransformation`", "CIP`Utility`", "Combinatorica`"}];
		DistributeDefinitions[dataTransformationMode, targetInterval];

		Return[
			ParallelTable[
				FitMpr[
					dataSet,
					polynomialDegreeList[[i]],
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType
				],
				
				{i, Length[polynomialDegreeList]}
			]			
		]
	];

GetBestMprClassOptimization[

	(* Returns best training set optimization result of MPR for classification.

	   Returns: 
	   Best index for classification *)


	(* mprTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mprInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mprInfoList: List with mprInfo
	   mprInfoList[[i]] refers to optimization step i *)
	mprTrainOptimization_,
	
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
				GetBestMprClassOptimizationPC[
					mprTrainOptimization,
					UtilityOptionBestOptimization -> bestOptimization
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetBestMprClassOptimizationSC[
					mprTrainOptimization,
					UtilityOptionBestOptimization -> bestOptimization
				]
			]
		]
	];

GetBestMprClassOptimizationSC[

	(* Returns best training set optimization result of MPR for classification.

	   Returns: 
	   Best index for classification *)


	(* mprTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mprInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mprInfoList: List with mprInfo
	   mprInfoList[[i]] refers to optimization step i *)
	mprTrainOptimization_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			k,
			trainingAndTestSetList,
			mprInfoList,
			maximumCorrectClassificationInPercent,
			mprInfo,
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
			trainingAndTestSetList = mprTrainOptimization[[3]];
			mprInfoList = mprTrainOptimization[[4]];
			maximumCorrectClassificationInPercent = -1.0;
			Do[
				testSet = trainingAndTestSetList[[k, 2]];
				mprInfo = mprInfoList[[k]];
				correctClassificationInPercent = CalculateMprCorrectClassificationInPercent[testSet, mprInfo];
				If[correctClassificationInPercent > maximumCorrectClassificationInPercent,
					maximumCorrectClassificationInPercent = correctClassificationInPercent;
					bestIndex = k
				],
				
				{k, Length[mprInfoList]}
			],
			
			(* ------------------------------------------------------------------------------- *)			
			"MinimumDeviation",
			trainingAndTestSetList = mprTrainOptimization[[3]];
			mprInfoList = mprTrainOptimization[[4]];
			minimumDeviation = Infinity;
			Do[
				trainingSet = trainingAndTestSetList[[k, 1]];
				testSet = trainingAndTestSetList[[k, 2]];
				mprInfo = mprInfoList[[k]];
				trainingSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[trainingSet, mprInfo];
				testSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[testSet, mprInfo];
				deviation = Abs[testSetCorrectClassificationInPercent - trainingSetCorrectClassificationInPercent];
				If[deviation < minimumDeviation || (deviation == minimumDeviation && testSetCorrectClassificationInPercent < bestTestSetCorrectClassificationInPercent),
					minimumDeviation = deviation;
					bestTestSetCorrectClassificationInPercent = testSetCorrectClassificationInPercent;
					bestIndex = k
				],
				
				{k, Length[mprInfoList]}
			]
		];

		Return[bestIndex]
	];

GetBestMprClassOptimizationPC[

	(* Returns best training set optimization result of MPR for classification.

	   Returns: 
	   Best index for classification *)


	(* mprTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mprInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mprInfoList: List with mprInfo
	   mprInfoList[[i]] refers to optimization step i *)
	mprTrainOptimization_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			k,
			trainingAndTestSetList,
			mprInfoList,
			maximumCorrectClassificationInPercent,
			mprInfo,
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
			trainingAndTestSetList = mprTrainOptimization[[3]];
			mprInfoList = mprTrainOptimization[[4]];
			maximumCorrectClassificationInPercent = -1.0;
			
			ParallelNeeds[{"CIP`MPR`", "CIP`DataTransformation`", "CIP`Utility`"}];
			DistributeDefinitions[trainingAndTestSetList, mprInfoList];
			
			correctClassificationInPercentList = ParallelTable[
				testSet = trainingAndTestSetList[[k, 2]];
				mprInfo = mprInfoList[[k]];
				
				CalculateMprCorrectClassificationInPercent[testSet, mprInfo],
				
				{k, Length[mprInfoList]}
			];
			
			Do[
				If[correctClassificationInPercentList[[k]] > maximumCorrectClassificationInPercent,
					maximumCorrectClassificationInPercent = correctClassificationInPercentList[[k]];
					bestIndex = k
				],
				
				{k, Length[mprInfoList]}
			],
			
			(* ------------------------------------------------------------------------------- *)			
			"MinimumDeviation",
			trainingAndTestSetList = mprTrainOptimization[[3]];
			mprInfoList = mprTrainOptimization[[4]];
			minimumDeviation = Infinity;
			
			ParallelNeeds[{"CIP`MPR`", "CIP`DataTransformation`", "CIP`Utility`"}];
			DistributeDefinitions[trainingAndTestSetList, mprInfoList];
			
			listOfTestSetCorrectClassificationInPercentAndDeviation = ParallelTable[
				trainingSet = trainingAndTestSetList[[k, 1]];
				testSet = trainingAndTestSetList[[k, 2]];
				mprInfo = mprInfoList[[k]];
				trainingSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[trainingSet, mprInfo];
				testSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[testSet, mprInfo];
				deviation = Abs[testSetCorrectClassificationInPercent - trainingSetCorrectClassificationInPercent];
				
				{
					testSetCorrectClassificationInPercent,
					deviation
				},
				
				{k, Length[mprInfoList]}
			];
			
			Do[
				If[listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]] < minimumDeviation || (listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]] == minimumDeviation && listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 1]] < bestTestSetCorrectClassificationInPercent),
					minimumDeviation = listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]];
					bestTestSetCorrectClassificationInPercent = listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 1]];
					bestIndex = k
				],
				
				{k, Length[mprInfoList]}
			]
		];

		Return[bestIndex]
	];

GetBestMprRegressOptimization[

	(* Returns best optimization result of MPR for regression.

	   Returns: 
	   Best index for regression *)


	(* mprTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mprInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mprInfoList: List with mprInfo
	   mprInfoList[[i]] refers to optimization step i *)
	mprTrainOptimization_,
	
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
				mprTrainOptimization, 
				UtilityOptionBestOptimization -> bestOptimization
			]
		]
	];

GetInternalMprOutput[

	(* Returns output of MPR according to specified input.

	   Returns: output: {outputValue1, outputValue2, ..., outputValue<<numberOfoutputValues>} *)


    (* {inputValue1, inputValue2, ..., inputValue<<numberOfInputValues>} *)
    input_/;VectorQ[input, NumberQ],

	(* mprFitResult: See code of FitMpr[] and signature of Fit[] *)
    mprFitResult_
    
	] :=
  
	Module[
    
		{
			j,
			k,
			output,
			replacementList
		},
    
		(* NOTE: Subscripted mprVariable variables are NOT local *)
		replacementList = 
			Table[
				Subscript[mprVariable, j] -> input[[j]], 
					
				{j, Length[input]}
			];
        output = 
        	Table[
        		mprFitResult[[k]] /. replacementList, 
        			
        		{k, Length[mprFitResult]}
        	];
		Return[output]
	];

GetInternalMprOutputs[

	(* Returns outputs of MPR according to specified inputs.

	   Returns: 
	   outputs: {output1, ..., output<Length[inputs]>} 
	   output[[i]] corresponds to inputs[[i]] *)

    (* inputs: {input1, input2, ...} 
       input: {inputValue1, inputValue2, ..., inputValue<<numberOfInputValues>} *)
    inputs_/;MatrixQ[inputs, NumberQ],

	(* mprFitResult: See code of FitMpr[] and signature of Fit[] *)
    mprFitResult_
	
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
    
		(* NOTE: Subscripted mprVariable variables are NOT local *)
		outputs = 
			Table[
		    	singleInput = inputs[[i]];
				replacementList = 
					Table[
						Subscript[mprVariable, j] -> singleInput[[j]], 
							
						{j, Length[singleInput]}
					];
	    		Table[
	    			mprFitResult[[k]] /. replacementList, 
	    				
	    			{k, Length[mprFitResult]}
	    		],
	    
				{i, Length[inputs]}
		    ];
		Return[outputs]
	];

GetMprInputInclusionClass[

	(* Analyzes relevance of input components by successive get-one-in for classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mprInputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mprInfo}
	   trainingSetResult: {numberOfIncludedInputs, (best) classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, (best) classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Order of polynomial to be fit (= 1: MPR) *)
	polynomialDegree_?IntegerQ,	
	
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
		(* MPR options *)   
		dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
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
				GetMprInputInclusionCalculationPC[
					trainingAndTestSet,
					polynomialDegree,
					isRegression,
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetMprInputInclusionCalculationSC[
					trainingAndTestSet,
					polynomialDegree,
					isRegression,
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			]
		]
	];

GetMprInputInclusionRegress[

	(* Analyzes relevance of input components by successive get-one-in for regression.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mprInputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mprInfo}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Order of polynomial to be fit (= 1: MPR) *)
	polynomialDegree_?IntegerQ,	
	
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
		(* MPR options *)   
		dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
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
				GetMprInputInclusionCalculationPC[
					trainingAndTestSet,
					polynomialDegree,
					isRegression,
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetMprInputInclusionCalculationSC[
					trainingAndTestSet,
					polynomialDegree,
					isRegression,
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			]
		]
	];

GetMprInputInclusionCalculationSC[

	(* Analyzes relevance of input components by successive get-one-in for regression and classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mprInputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mprInfo}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Order of polynomial to be fit (= 1: MPR) *)
	polynomialDegree_?IntegerQ,	
	
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
			mprInputComponentRelevanceList,
	        mprInfo,
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
		(* MPR options *)   
		dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
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
		mprInputComponentRelevanceList = {};
    
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
					mprInfo = 
						FitMpr[
							trainingSet,
							polynomialDegree,
							MprOptionDataTransformationMode -> dataTransformationMode,
							DataTransformationOptionTargetInterval -> targetInterval,
							DataTransformationOptionNormalizationType -> normalizationType
						];
					If[Length[testSet] > 0,
            
						testSetRmse = CalculateMprDataSetRmse[testSet, mprInfo];
						AppendTo[rmseList,{testSetRmse, i}],
          
						trainingSetRmse = CalculateMprDataSetRmse[trainingSet, mprInfo];
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
			mprInfo = 
				FitMpr[
					trainingSet,
					polynomialDegree,
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMprDataSetRmse[trainingSet, mprInfo];
					currentTestSetRmse = CalculateMprDataSetRmse[testSet, mprInfo];
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
							mprInfo
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMprDataSetRmse[trainingSet, mprInfo];
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
							mprInfo
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[trainingSet, mprInfo];
					currentTestSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[testSet, mprInfo];
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
							mprInfo
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[trainingSet, mprInfo];
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
							mprInfo
						}
				]
			];	

			AppendTo[mprInputComponentRelevanceList, relevance];
			includedInputComponentList = currentIncludedInputComponentList,
			
			{k, Length[numberOfInclusionsPerStepList]}
		];
		
		Return[mprInputComponentRelevanceList]
	];

GetMprInputInclusionCalculationPC[

	(* Analyzes relevance of input components by successive get-one-in for regression and classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mprInputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, mprInfo}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Order of polynomial to be fit (= 1: MPR) *)
	polynomialDegree_?IntegerQ,	
	
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
			mprInputComponentRelevanceList,
	        mprInfo,
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
		(* MPR options *)   
		dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
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
		mprInputComponentRelevanceList = {};
    	
    	ParallelNeeds[{"CIP`MPR`", "CIP`DataTransformation`", "CIP`Utility`"}];
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
						mprInfo = 
							FitMpr[
								trainingSet,
								polynomialDegree,
								MprOptionDataTransformationMode -> dataTransformationMode,
								DataTransformationOptionTargetInterval -> targetInterval,
								DataTransformationOptionNormalizationType -> normalizationType
							];
						If[Length[testSet] > 0,
            
							testSetRmse = CalculateMprDataSetRmse[testSet, mprInfo];
							{testSetRmse, i},
         
							trainingSetRmse = CalculateMprDataSetRmse[trainingSet, mprInfo];
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
			mprInfo = 
				FitMpr[
					trainingSet,
					polynomialDegree,
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMprDataSetRmse[trainingSet, mprInfo];
					currentTestSetRmse = CalculateMprDataSetRmse[testSet, mprInfo];
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
							mprInfo
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMprDataSetRmse[trainingSet, mprInfo];
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
							mprInfo
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[trainingSet, mprInfo];
					currentTestSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[testSet, mprInfo];
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
							mprInfo
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[trainingSet, mprInfo];
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
							mprInfo
						}
				]
			];	

			AppendTo[mprInputComponentRelevanceList, relevance];
			includedInputComponentList = currentIncludedInputComponentList,
			
			{k, Length[numberOfInclusionsPerStepList]}
		];
		
		Return[mprInputComponentRelevanceList]
	];

GetMprInputRelevanceClass[

	(* Analyzes relevance of input components by successive leave-one-out for classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mprInputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mprInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, (best) classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* Order of polynomial to be fit (= 1: MPR) *)
	polynomialDegree_?IntegerQ,	
	
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
		(* MPR options *)   
		dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
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
				GetMprInputRelevanceCalculationPC[
					trainingAndTestSet,
					polynomialDegree,
					isRegression,
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetMprInputRelevanceCalculationSC[
					trainingAndTestSet,
					polynomialDegree,
					isRegression,
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			]
		]
	];

GetMprClassRelevantComponents[

	(* Returns most-to-least-relevance sorted components from mprInputComponentRelevanceListForClassification.

	   Returns: Returns most-to-least-relevance sorted components *)


	(* mprInputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mprInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) correct classification in percent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best correct classification in percent of test set} *)
	mprInputComponentRelevanceListForClassification_,
	
	(* Number of components to return *)
	numberOfComponents_?IntegerQ
    
	] :=
  
	Module[
    
		{},

		Return[
			CIP`Graphics`GetClassRelevantComponents[mprInputComponentRelevanceListForClassification, numberOfComponents]
		]
	];

GetMprInputRelevanceRegress[

	(* Analyzes relevance of input components by successive leave-one-out for regression.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mprInputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mprInfo}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Order of polynomial to be fit (= 1: MPR) *)
	polynomialDegree_?IntegerQ,
	
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
		(* MPR options *)   
		dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
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
				GetMprInputRelevanceCalculationPC[
					trainingAndTestSet,
					polynomialDegree,
					isRegression,
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetMprInputRelevanceCalculationSC[
					trainingAndTestSet,
					polynomialDegree,
					isRegression,
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			]
		]
	];

GetMprRegressRelevantComponents[

	(* Returns most-to-least-relevance sorted components from mprInputComponentRelevanceListForRegression.

	   Returns: Returns most-to-least-relevance sorted components *)


	(* mprInputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mprInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)
	mprInputComponentRelevanceListForRegression_,
	
	(* Number of components to return *)
	numberOfComponents_?IntegerQ
    
	] :=
  
	Module[
    
		{},

		Return[
			CIP`Graphics`GetRegressRelevantComponents[mprInputComponentRelevanceListForRegression, numberOfComponents]
		]
	];

GetMprInputRelevanceCalculationSC[

	(* Analyzes relevance of input components by successive leave-one-out for regression and classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mprInputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mprInfo}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Order of polynomial to be fit (= 1: MPR) *)
	polynomialDegree_?IntegerQ,
	
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
			mprInputComponentRelevanceList,
	        mprInfo,
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
		(* MPR options *)   
		dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
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
		mprInputComponentRelevanceList = {};
    
		(* Result for no removal *)
		trainingSet = trainingAndTestSet[[1]];
		testSet = trainingAndTestSet[[2]];
		mprInfo = 
			FitMpr[
				trainingSet,
				polynomialDegree,
				MprOptionDataTransformationMode -> dataTransformationMode,
				DataTransformationOptionTargetInterval -> targetInterval,
				DataTransformationOptionNormalizationType -> normalizationType
			];
		
		initialTrainingSetRmse = CalculateMprDataSetRmse[trainingSet, mprInfo];
		If[isRegression,
			
			(* Regression*)
			If[Length[testSet] > 0,
				
				(* Regression WITH test set*)
				initialTestSetRmse = CalculateMprDataSetRmse[testSet, mprInfo];
				If[isIntermediateOutput,
					Print["initialTrainingSetRmse = ", initialTrainingSetRmse];
					Print["initialTestSetRmse     = ", initialTestSetRmse]
				];
				relevance = 
					{
						{0.0, initialTrainingSetRmse},
						{0.0, initialTestSetRmse},
						{}, 
						mprInfo
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
						mprInfo
					}
			],
			
			(* Classification *)
			initialTrainingSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[trainingSet, mprInfo];
			If[Length[testSet] > 0,
				
				(* Classification WITH test set*)
				initialTestSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[testSet, mprInfo];
				If[isIntermediateOutput,
					Print["initialTrainingSetCorrectClassificationInPercent = ", initialTrainingSetCorrectClassificationInPercent];
					Print["initialTestSetCorrectClassificationInPercent     = ", initialTestSetCorrectClassificationInPercent]
				];
				relevance = 
					{
						{0.0, initialTrainingSetCorrectClassificationInPercent},
						{0.0, initialTestSetCorrectClassificationInPercent},
						{}, 
						mprInfo
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
						mprInfo
					}
			]
		];	
		
		AppendTo[mprInputComponentRelevanceList, relevance];
    
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
					mprInfo = 
						FitMpr[
							trainingSet,
							polynomialDegree,
							MprOptionDataTransformationMode -> dataTransformationMode,
							DataTransformationOptionTargetInterval -> targetInterval,
							DataTransformationOptionNormalizationType -> normalizationType
						];
					If[Length[testSet] > 0,
            
						testSetRmse = CalculateMprDataSetRmse[testSet, mprInfo];
						AppendTo[rmseList,{testSetRmse, i}],
          
						trainingSetRmse = CalculateMprDataSetRmse[trainingSet, mprInfo];
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
			mprInfo = 
				FitMpr[
					trainingSet,
					polynomialDegree,
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMprDataSetRmse[trainingSet, mprInfo];
					currentTestSetRmse = CalculateMprDataSetRmse[testSet, mprInfo];
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
							mprInfo
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMprDataSetRmse[trainingSet, mprInfo];
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
							mprInfo
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[trainingSet, mprInfo];
					currentTestSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[testSet, mprInfo];
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
							mprInfo
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[trainingSet, mprInfo];
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
							mprInfo
						}
				]
			];	

			AppendTo[mprInputComponentRelevanceList, relevance];
			removedInputComponentList = currentRemovedInputComponentList,
			
			{k, Length[numberOfExclusionsPerStepList]}
		];
		
		Return[mprInputComponentRelevanceList]
	];

GetMprInputRelevanceCalculationPC[

	(* Analyzes relevance of input components by successive leave-one-out for regression and classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   mprInputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mprInfo}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* Order of polynomial to be fit (= 1: MPR) *)
	polynomialDegree_?IntegerQ,
	
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
			mprInputComponentRelevanceList,
	        mprInfo,
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
		(* MPR options *)   
		dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
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
		mprInputComponentRelevanceList = {};
    
		(* Result for no removal *)
		trainingSet = trainingAndTestSet[[1]];
		testSet = trainingAndTestSet[[2]];
		mprInfo = 
			FitMpr[
				trainingSet,
				polynomialDegree,
				MprOptionDataTransformationMode -> dataTransformationMode,
				DataTransformationOptionTargetInterval -> targetInterval,
				DataTransformationOptionNormalizationType -> normalizationType
			];
		
		initialTrainingSetRmse = CalculateMprDataSetRmse[trainingSet, mprInfo];
		If[isRegression,
			
			(* Regression*)
			If[Length[testSet] > 0,
				
				(* Regression WITH test set*)
				initialTestSetRmse = CalculateMprDataSetRmse[testSet, mprInfo];
				If[isIntermediateOutput,
					Print["initialTrainingSetRmse = ", initialTrainingSetRmse];
					Print["initialTestSetRmse     = ", initialTestSetRmse]
				];
				relevance = 
					{
						{0.0, initialTrainingSetRmse},
						{0.0, initialTestSetRmse},
						{}, 
						mprInfo
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
						mprInfo
					}
			],
			
			(* Classification *)
			initialTrainingSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[trainingSet, mprInfo];
			If[Length[testSet] > 0,
				
				(* Classification WITH test set*)
				initialTestSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[testSet, mprInfo];
				If[isIntermediateOutput,
					Print["initialTrainingSetCorrectClassificationInPercent = ", initialTrainingSetCorrectClassificationInPercent];
					Print["initialTestSetCorrectClassificationInPercent     = ", initialTestSetCorrectClassificationInPercent]
				];
				relevance = 
					{
						{0.0, initialTrainingSetCorrectClassificationInPercent},
						{0.0, initialTestSetCorrectClassificationInPercent},
						{}, 
						mprInfo
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
						mprInfo
					}
			]
		];	
		
		AppendTo[mprInputComponentRelevanceList, relevance];
		
		ParallelNeeds[{"CIP`MPR`", "CIP`DataTransformation`", "CIP`Utility`"}];
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
						mprInfo = 
							FitMpr[
								trainingSet,
								polynomialDegree,
								MprOptionDataTransformationMode -> dataTransformationMode,
								DataTransformationOptionTargetInterval -> targetInterval,
								DataTransformationOptionNormalizationType -> normalizationType
							];
						If[Length[testSet] > 0,
	            
							testSetRmse = CalculateMprDataSetRmse[testSet, mprInfo];
							{testSetRmse, i},
	          
							trainingSetRmse = CalculateMprDataSetRmse[trainingSet, mprInfo];
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
			mprInfo = 
				FitMpr[
					trainingSet,
					polynomialDegree,
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateMprDataSetRmse[trainingSet, mprInfo];
					currentTestSetRmse = CalculateMprDataSetRmse[testSet, mprInfo];
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
							mprInfo
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateMprDataSetRmse[trainingSet, mprInfo];
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
							mprInfo
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[trainingSet, mprInfo];
					currentTestSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[testSet, mprInfo];
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
							mprInfo
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateMprCorrectClassificationInPercent[trainingSet, mprInfo];
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
							mprInfo
						}
				]
			];	

			AppendTo[mprInputComponentRelevanceList, relevance];
			removedInputComponentList = currentRemovedInputComponentList,
			
			{k, Length[numberOfExclusionsPerStepList]}
		];
		
		Return[mprInputComponentRelevanceList]
	];

GetMprNumberOfParameters[

	(* Returns the number of parameters of the machine learning model.

	   Returns: 
	   Returns the number of parameters of the machine learning model *)
	   
	   
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* Order of polynomial to be fit (= 1: MLR) *)
	polynomialDegree_?IntegerQ

	] :=
  
	Module[
    
		{
			numberOfInputVariables,
			variableVector,
			functionVector,
			i,
			k,
			argumentForTable,
			index
		},

		numberOfInputVariables = Length[dataSet[[1, 1]]];
		
		(* Clear mprVariable variable. NOTE: This is NOT a local variable *)
		Clear[mprVariable];

		(* Create subscripted mprVariable variable vector and function vector. NOTE : These are NOT local variables *)
		variableVector = Table[Subscript[mprVariable, i], {i, 1, numberOfInputVariables}];
		functionVector = {1.0};
		Do[
			argumentForTable ={Product[Subscript[mprVariable, Subscript[index, i]], {i, 1, k}]};
			Do[
				AppendTo[argumentForTable, {Subscript[index, i], 1, numberOfInputVariables}],
				{i, 1, polynomialDegree}
			];		  
			functionVector = 
				Flatten[
					{
						functionVector,
						DeleteDuplicates[Flatten[
							Apply[Table, argumentForTable]
						]]
					}
				],
			{k, 1, polynomialDegree}
		];

    	Return[Length[functionVector]]
	];
	
GetMprRegressionResult[
	
	(* Returns MPR regression result according to named property list.

	   Returns :
	   MPR regression result according to named property *)

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
    mprInfo_,
	
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
	    
		pureFunction = Function[inputs, CalculateMprOutputs[inputs, mprInfo]];
	    Return[
	    	CIP`Graphics`GetSingleRegressionResult[
		    	namedProperty, 
		    	dataSet, 
		    	pureFunction,
		    	GraphicsOptionNumberOfIntervals -> numberOfIntervals
			]
		]
	];

GetMprSeriesClassificationResult[

	(* Shows result of MPR series classifications for training and test set.

	   Returns: 
	   mprSeriesClassificationResult: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in mprInfoList, classification result in percent for training set}
	   testPoint[[i]]: {index i in mprInfoList, classification result in percent for test set} *)


    (* {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...}
	   testSet has the same structure and restrictions as trainingSet 
	   NOTE: Training and test set MUST be in original units *)
    trainingAndTestSet_,

	(* {mprInfo1, mprInfo2, ...}
	   mprInfo (see "Frequently used data structures") *)
    mprInfoList_
    
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
			pureFunction = Function[inputs, CalculateMprClassNumbers[inputs, mprInfoList[[i]]]];
			correctClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[trainingSet, pureFunction];
			AppendTo[trainingPoints2D, {N[i], correctClassificationInPercent}];
			If[Length[testSet] > 0,
				correctClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[testSet, pureFunction];
				AppendTo[testPoints2D, {N[i], correctClassificationInPercent}]
			],
			
			{i, Length[mprInfoList]}
		];
		
		Return[{trainingPoints2D, testPoints2D}]
	];

GetMprSeriesRmse[

	(* Shows RMSE of MPR series for training and test set.

	   Returns: 
	   mprSeriesRmse: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in mprInfoList, RMSE for training set}
	   testPoint[[i]]: {index i in mprInfoList, RMSE for test set} *)


    (* {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...}
	   testSet has the same structure and restrictions as trainingSet 
	   NOTE: Training and test set MUST be in original units *)
    trainingAndTestSet_,

	(* {mprInfo1, mprInfo2, ...}
	   mprInfo (see "Frequently used data structures") *)
    mprInfoList_
    
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
			pureFunction = Function[inputs, CalculateMprOutputs[inputs, mprInfoList[[i]]]];
			rmse = Sqrt[CIP`Utility`GetMeanSquaredError[trainingSet, pureFunction]];
			AppendTo[trainingPoints2D, {N[i], rmse}];
			If[Length[testSet] > 0,
				rmse = Sqrt[CIP`Utility`GetMeanSquaredError[testSet, pureFunction]];
				AppendTo[testPoints2D, {N[i], rmse}]
			],
			
			{i, Length[mprInfoList]}
		];
		
		Return[{trainingPoints2D, testPoints2D}]
	];

GetMprTrainOptimization[

	(* Returns training set optimization result for MPR training.

	   Returns:
	   mprTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, mprInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   mprInfoList: List with mprInfo
	   mprInfoList[[i]] refers to optimization step i *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* Order of polynomial to be fit (= 1: MPR) *)
	polynomialDegree_?IntegerQ,

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
			mprInfo,
			normalizationType,
			trainingSetRMSE,
			testSetRMSE,
			pureOutputFunction,
			trainingSetRmseList,
			testSetRmseList,
			trainingAndTestSetList,
			mprInfoList,
			selectionResult,
			blackList
		},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MPR options *)
	    dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
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
		mprInfoList = {};
		blackList = {};
		Do[
			(* Fit training set and evaluate RMSE *)
			trainingSet = CIP`DataTransformation`GetDataSetPart[dataSet, trainingSetIndexList];
			testSet = CIP`DataTransformation`GetDataSetPart[dataSet, testSetIndexList];
			mprInfo = 
				FitMpr[
					trainingSet,
					polynomialDegree,
					MprOptionDataTransformationMode -> dataTransformationMode,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType
				];
			trainingSetRMSE = CalculateMprDataSetRmse[trainingSet, mprInfo];
			testSetRMSE = CalculateMprDataSetRmse[testSet, mprInfo];

			(* Set iteration results *)
			AppendTo[trainingSetRmseList, {N[i], trainingSetRMSE}];
			AppendTo[testSetRmseList, {N[i], testSetRMSE}];
			AppendTo[trainingAndTestSetList, {trainingSet, testSet}];
			AppendTo[mprInfoList, mprInfo];
			
			(* Break if necessary *)
			If[i == numberOfTrainingSetOptimizationSteps,
				Break[]
			];

			(* Select new training and test set index lists *)
			pureOutputFunction = Function[input, CalculateMprOutput[input, mprInfo]];
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
				mprInfoList
			}
		]
	];

ScanClassTrainingWithMpr[

	(* Scans training and test set for different training fractions based on method FitMpr, see code.
	
	   Returns:
	   mprClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mprInfo1}, {trainingAndTestSet2, mprInfo2}, ...}
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

	(* Order of polynomial to be fit (= 1: MPR) *)
	polynomialDegree_?IntegerQ,

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
		(* MPR options *)
		dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
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
				ScanClassTrainingWithMprPC[
					classificationDataSet,
					polynomialDegree,
					trainingFractionList,
					MprOptionDataTransformationMode -> dataTransformationMode,
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
				ScanClassTrainingWithMprSC[
					classificationDataSet,
					polynomialDegree,
					trainingFractionList,
					MprOptionDataTransformationMode -> dataTransformationMode,
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

ScanClassTrainingWithMprSC[

	(* Scans training and test set for different training fractions based on method FitMpr, see code.
	
	   Returns:
	   mprClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mprInfo1}, {trainingAndTestSet2, mprInfo2}, ...}
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

	(* Order of polynomial to be fit (= 1: MPR) *)
	polynomialDegree_?IntegerQ,

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
			currentMprInfo,
			pureFunction,
			trainingSetCorrectClassificationInPercent,
			testSetCorrectClassificationInPercent,
			mprTrainOptimization,
			trainingAndTestSetList,
			mprInfoList,
			normalizationType,
			bestIndex
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MPR options *)
		dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
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
				mprTrainOptimization = 
					GetMprTrainOptimization[
						classificationDataSet,
						polynomialDegree, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						MprOptionDataTransformationMode -> dataTransformationMode,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						DataTransformationOptionTargetInterval -> targetInterval,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestMprClassOptimization[mprTrainOptimization];
				trainingAndTestSetList = mprTrainOptimization[[3]];
				mprInfoList = mprTrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMprInfo = mprInfoList[[bestIndex]],
				
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
				currentMprInfo = 
					FitMpr[
						currentTrainingSet,
						polynomialDegree,
						MprOptionDataTransformationMode -> dataTransformationMode,
						DataTransformationOptionTargetInterval -> targetInterval,
						DataTransformationOptionNormalizationType -> normalizationType
					]
			];
			
			pureFunction = Function[inputs, CalculateMprClassNumbers[inputs, currentMprInfo]];
			trainingSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTrainingSet, pureFunction];
			testSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTestSet, pureFunction];
			AppendTo[trainingAndTestSetsInfo, {currentTrainingAndTestSet, currentMprInfo}];
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
	
ScanClassTrainingWithMprPC[

	(* Scans training and test set for different training fractions based on method FitMpr, see code.
	
	   Returns:
	   mprClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mprInfo1}, {trainingAndTestSet2, mprInfo2}, ...}
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

	(* Order of polynomial to be fit (= 1: MPR) *)
	polynomialDegree_?IntegerQ,

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
			currentMprInfo,
			pureFunction,
			trainingSetCorrectClassificationInPercent,
			testSetCorrectClassificationInPercent,
			mprTrainOptimization,
			trainingAndTestSetList,
			mprInfoList,
			normalizationType,
			bestIndex,
			listOfTrainingAndTestSetsInfoAndScanReport
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MPR options *)
		dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
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
	    
		ParallelNeeds[{"CIP`MPR`", "CIP`Cluster`", "CIP`DataTransformation`", "CIP`Utility`", "Combinatorica`"}];
		DistributeDefinitions[numberOfTrainingSetOptimizationSteps, dataTransformationMode, clusterMethod, maximumNumberOfEpochs, scalarProductMinimumTreshold, 
			maximumNumberOfTrialSteps, targetInterval, randomValueInitialization, deviationCalculationMethod, blackListLength];
		
		listOfTrainingAndTestSetsInfoAndScanReport = ParallelTable[
			If[numberOfTrainingSetOptimizationSteps > 0,
			
				(* ------------------------------------------------------------------------------------------------------ *)
				(* Training set optimization *)
				mprTrainOptimization = 
					GetMprTrainOptimization[
						classificationDataSet,
						polynomialDegree, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						MprOptionDataTransformationMode -> dataTransformationMode,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						DataTransformationOptionTargetInterval -> targetInterval,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
			 	        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
					
				bestIndex = GetBestMprClassOptimization[mprTrainOptimization];
				
				trainingAndTestSetList = mprTrainOptimization[[3]];
				mprInfoList = mprTrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMprInfo = mprInfoList[[bestIndex]],
			
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
				currentMprInfo = 
					FitMpr[
						currentTrainingSet,
						polynomialDegree,
						MprOptionDataTransformationMode -> dataTransformationMode,
						DataTransformationOptionTargetInterval -> targetInterval,
						DataTransformationOptionNormalizationType -> normalizationType
					]
			];
		
			pureFunction = Function[inputs, CalculateMprClassNumbers[inputs, currentMprInfo]];
			trainingSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTrainingSet, pureFunction];
			testSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTestSet, pureFunction];
			{
				{currentTrainingAndTestSet, currentMprInfo},
			
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

ScanRegressTrainingWithMpr[

	(* Scans training and test set for different training fractions based on method FitMpr, see code.
	
	   Returns:
	   mprRegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mprInfo1}, {trainingAndTestSet2, mprInfo2}, ...}
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

	(* Order of polynomial to be fit (= 1: MPR) *)
	polynomialDegree_?IntegerQ,

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
		(* MPR options *)
		dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
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
			    ScanRegressTrainingWithMprPC[
		    		dataSet,
		    		polynomialDegree,
		    		trainingFractionList,
		    		MprOptionDataTransformationMode -> dataTransformationMode,
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
			    ScanRegressTrainingWithMprSC[
			    	dataSet,
		    		polynomialDegree,
		    		trainingFractionList,
		    		MprOptionDataTransformationMode -> dataTransformationMode,
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

ScanRegressTrainingWithMprSC[

	(* Scans training and test set for different training fractions based on method FitMpr, see code.
	
	   Returns:
	   mprRegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mprInfo1}, {trainingAndTestSet2, mprInfo2}, ...}
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

	(* Order of polynomial to be fit (= 1: MPR) *)
	polynomialDegree_?IntegerQ,

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
			currentMprInfo,
			pureFunction,
			trainingSetRMSE,
			testSetRMSE,
			mprTrainOptimization,
			trainingAndTestSetList,
			mprInfoList,
			normalizationType,
			bestIndex
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MPR options *)
		dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
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
				mprTrainOptimization = 
					GetMprTrainOptimization[
						dataSet,
						polynomialDegree, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						MprOptionDataTransformationMode -> dataTransformationMode,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						DataTransformationOptionTargetInterval -> targetInterval,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestMprRegressOptimization[mprTrainOptimization];
				trainingAndTestSetList = mprTrainOptimization[[3]];
				mprInfoList = mprTrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMprInfo = mprInfoList[[bestIndex]],
				
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
				currentMprInfo = 
					FitMpr[
						currentTrainingSet,
						polynomialDegree,
						MprOptionDataTransformationMode -> dataTransformationMode,
						DataTransformationOptionTargetInterval -> targetInterval,
						DataTransformationOptionNormalizationType -> normalizationType
					]
			];
			
			pureFunction = Function[inputs, CalculateMprOutputs[inputs, currentMprInfo]];
			trainingSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTrainingSet, pureFunction]];
			testSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTestSet, pureFunction]];
			AppendTo[trainingAndTestSetsInfo, {currentTrainingAndTestSet, currentMprInfo}];
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

ScanRegressTrainingWithMprPC[

	(* Scans training and test set for different training fractions based on method FitMpr, see code.
	
	   Returns:
	   mprRegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mprInfo1}, {trainingAndTestSet2, mprInfo2}, ...}
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

	(* Order of polynomial to be fit (= 1: MPR) *)
	polynomialDegree_?IntegerQ,

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
			currentMprInfo,
			pureFunction,
			trainingSetRMSE,
			testSetRMSE,
			mprTrainOptimization,
			trainingAndTestSetList,
			mprInfoList,
			normalizationType,
			bestIndex,
			listOfTrainingAndTestSetsInfoAndScanReport
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* MPR options *)
		dataTransformationMode = MprOptionDataTransformationMode/.{opts}/.Options[MprOptionsDataTransformation];
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

		ParallelNeeds[{"CIP`MPR`", "CIP`Cluster`", "CIP`DataTransformation`", "CIP`Utility`", "Combinatorica`"}];
		DistributeDefinitions[numberOfTrainingSetOptimizationSteps, dataTransformationMode, clusterMethod, maximumNumberOfEpochs, scalarProductMinimumTreshold, 
			maximumNumberOfTrialSteps, targetInterval, randomValueInitialization, deviationCalculationMethod, blackListLength];
		
		listOfTrainingAndTestSetsInfoAndScanReport = ParallelTable[
			If[numberOfTrainingSetOptimizationSteps > 0,
				
				(* ------------------------------------------------------------------------------------------------------ *)
				(* Training set optimization *)
				mprTrainOptimization = 
					GetMprTrainOptimization[
						dataSet,
						polynomialDegree, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						MprOptionDataTransformationMode -> dataTransformationMode,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						DataTransformationOptionTargetInterval -> targetInterval,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestMprRegressOptimization[mprTrainOptimization];
				trainingAndTestSetList = mprTrainOptimization[[3]];
				mprInfoList = mprTrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentMprInfo = mprInfoList[[bestIndex]],
				
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
				currentMprInfo = 
					FitMpr[
						currentTrainingSet,
						polynomialDegree,
						MprOptionDataTransformationMode -> dataTransformationMode,
						DataTransformationOptionTargetInterval -> targetInterval,
						DataTransformationOptionNormalizationType -> normalizationType
					]
			];
			
			pureFunction = Function[inputs, CalculateMprOutputs[inputs, currentMprInfo]];
			trainingSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTrainingSet, pureFunction]];
			testSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTestSet, pureFunction]];
			{
				{currentTrainingAndTestSet, currentMprInfo},
			
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

ShowMprOutput3D[

	(* Shows 3D MPR output.

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
    mprInfo_,
    
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

		dataSetScaleInfo = mprInfo[[2]];
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
			CIP`Graphics`Plot3dFunction[
				Function[{x1, x2}, CalculateMprValue3D[x1, x2, indexOfInput1, indexOfInput2, indexOfOutput, input, mprInfo]], 
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

ShowMprClassificationResult[

	(* Shows result of MPR classification for training and test set according to named property list.

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
    mprInfo_,
    
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
		ShowMprSingleClassification[
			namedPropertyList,
			trainingSet, 
			mprInfo,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionMinMaxIndex -> minMaxIndex
		];
		
		(* Analyze test set *)
		If[Length[testSet] > 0,
			Print["Test Set:"];
			ShowMprSingleClassification[
				namedPropertyList,
				testSet, 
				mprInfo,
				GraphicsOptionImageSize -> imageSize,
				GraphicsOptionMinMaxIndex -> minMaxIndex
			];
		]
	];

ShowMprSingleClassification[

	(* Shows result of MPR classification for data set according to named property list.

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
    mprInfo_,
    
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

   		pureFunction = Function[inputs, CalculateMprClassNumbers[inputs, mprInfo]];
		CIP`Graphics`ShowClassificationResult[
			namedPropertyList,
			dataSet, 
			pureFunction,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionMinMaxIndex -> minMaxIndex
		]
	];

ShowMprClassificationScan[

	(* Shows result of MPR based classification scan of clustered training sets.

	   Returns: Nothing *)


	(* mprClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mprInfo1}, {trainingAndTestSet2, mprInfo2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, classification result in percent for training set}, {trainingFraction, classification result in percent for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)
	mprClassificationScan_,
	
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
			mprClassificationScan,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMprInputRelevanceClass[

	(* Shows mprInputComponentRelevanceListForClassification.

	   Returns: Nothing *)


	(* mprInputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mprInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) correct classification in percent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best correct classification in percent of test set} *)
	mprInputComponentRelevanceListForClassification_,
	
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
			mprInputComponentRelevanceListForClassification,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMprInputRelevanceRegress[

	(* Shows mprInputComponentRelevanceListForRegression.

	   Returns: Nothing *)


	(* mprInputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, mprInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)
	mprInputComponentRelevanceListForRegression_,
	
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
			mprInputComponentRelevanceListForRegression,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMprRegressionResult[

	(* Shows result of MPR regression for training and test set according to named property list.
	
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
    mprInfo_,
    
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
		ShowMprSingleRegression[
			namedPropertyList,
			trainingSet, 
			mprInfo,
			GraphicsOptionPointSize -> pointSize,
			GraphicsOptionPointColor -> pointColor
		];
		
		(* Analyze test set *)
		If[Length[testSet] > 0,
			Print["Test Set:"];
			ShowMprSingleRegression[
				namedPropertyList,
				testSet, 
				mprInfo,
				GraphicsOptionPointSize -> pointSize,
				GraphicsOptionPointColor -> pointColor
			]
		]
	];

ShowMprSingleRegression[
    
	(* Shows result of MPR regression for data set according to named property list.
	
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
    mprInfo_,
    
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

		pureFunction = Function[inputs, CalculateMprOutputs[inputs, mprInfo]];
		CIP`Graphics`ShowRegressionResult[
			namedPropertyList,
			dataSet, 
			pureFunction,
			GraphicsOptionPointSize -> pointSize,
			GraphicsOptionPointColor -> pointColor
		]
	];

ShowMprRegressionScan[

	(* Shows result of MPR based regression scan of clustered training sets.

	   Returns: Nothing *)


	(* mprRegressionScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, mprInfo1}, {trainingAndTestSet2, mprInfo2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, RMSE for training set}, {trainingFraction, RMSE for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)
	mprRegressionScan_,
	
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
			mprRegressionScan,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowMprSeriesClassificationResult[

	(* Shows result of MPR series classifications for training and test set.

	   Returns: Nothing *)


	(* mprSeriesClassificationResult: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in mprInfoList, classification result in percent for training set}
	   testPoint[[i]]: {index i in mprInfoList, classification result in percent for test set} *)
	mprSeriesClassificationResult_,
    
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

		trainingPoints2D = mprSeriesClassificationResult[[1]];
		testPoints2D = mprSeriesClassificationResult[[2]];
		
		If[Length[testPoints2D] > 0,

			(* Training and test set *)
			labels = {"mprInfo index", "Correct classifications [%]", "Training (green), Test (red)"};
			trainingPoints2DWithPlotStyle = {trainingPoints2D, {Thickness[0.005], Green}};
			testPoints2DWithPlotStyle = {testPoints2D, {Thickness[0.005], Red}};
			points2DWithPlotStyleList = {trainingPoints2DWithPlotStyle, testPoints2DWithPlotStyle};
			Print[
				CIP`Graphics`PlotMultiple2dLines[
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
			Print["Best test set classification with mprInfo index = ", bestIndexList],
		
			(* Training set only *)
			labels = {"mprInfo index", "Correct classifications [%]", "Training (green)"};
			trainingPoints2DWithPlotStyle = {trainingPoints2D, {Thickness[0.005], Green}};
			points2DWithPlotStyleList = {trainingPoints2DWithPlotStyle};
			Print[
				CIP`Graphics`PlotMultiple2dLines[
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
			Print["Best training set classification with mprInfo index = ", bestIndexList]			
		]
	];

ShowMprSeriesRmse[

	(* Shows RMSE of MPR series for training and test set.

	   Returns: Nothing *)


	(* mprSeriesRmse: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in mprInfoList, RMSE for training set}
	   testPoint[[i]]: {index i in mprInfoList, RMSE for test set} *)
	mprSeriesRmse_,
    
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

		trainingPoints2D = mprSeriesRmse[[1]];
		testPoints2D = mprSeriesRmse[[2]];

		If[Length[testPoints2D] > 0,
			
			(* Training and test set *)
			labels = {"mprInfo index", "RMSE", "Training (green), Test (red)"};
			trainingPoints2DWithPlotStyle = {trainingPoints2D, {Thickness[0.005], Green}};
			testPoints2DWithPlotStyle = {testPoints2D, {Thickness[0.005], Red}};
			points2DWithPlotStyleList = {trainingPoints2DWithPlotStyle, testPoints2DWithPlotStyle};
			Print[
				CIP`Graphics`PlotMultiple2dLines[
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
			Print["Best test set regression with mprInfo index = ", bestIndexList],

			(* Training set only *)
			labels = {"mprInfo index", "RMSE", "Training (green)"};
			trainingPoints2DWithPlotStyle = {trainingPoints2D, {Thickness[0.005], Green}};
			points2DWithPlotStyleList = {trainingPoints2DWithPlotStyle};
			Print[
				CIP`Graphics`PlotMultiple2dLines[
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
			Print["Best training set regression with mprInfo index = ", bestIndexList]			
		]
	];

ShowMprTrainOptimization[

	(* Shows training set optimization result of MPR.

	   Returns: Nothing *)


	(* mprTrainOptimization = {trainingSetRmseList, testSetRmseList, not interesting, not interesting}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set} *)
	mprTrainOptimization_,
    
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
			mprTrainOptimization, 
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

(* ::Section:: *)
(* End of Package *)

End[]

EndPackage[]
