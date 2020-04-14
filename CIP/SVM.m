(*
-------------------------------------------------------------------------------------
Computational Intelligence Packages (CIP): Package Support Vector Machines (SVM)
Version 3.1 for Mathematica 11 or higher
-------------------------------------------------------------------------------------

Authors: Kolja Berger (parallelization for CIP 2.0), Achim Zielesny 

GNWI - Gesellschaft fuer naturwissenschaftliche Informatik mbH, 
Dortmund, Germany

Citation:
Achim Zielesny, Computational Intelligence Packages (CIP), Version 3.1, 
GNWI mbH (http://www.gnwi.de), Dortmund, Germany, 2020.

Code partially based on:
B. Palancz, L. Voelgyesi, Support Vector Classifier via Mathematica, 
Periodica Polytechnica Civ. Eng 48 (1-2), 15-37, 2004.
B. Palancz, L. Voelgyesi, Gy. Popper, Support Vector Regression via 
Mathematica, Periodica Polytechnica Civ. Eng 49 (1), 59-84, 2005.
R. Nilsson, J. Bjoerkegren, Jesper Tegner, A Flexible Implementation 
for Support Vector Machines, The Mathematica Journal 10:1, 2006 
(Wolfram Media, Inc.).

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
-------------------------------------------------------------------------------------
*)

(* ::Section:: *)
(* Frequently used data structures *)

(*
-----------------------------------------------------------------------
Frequently used data structures
-----------------------------------------------------------------------
svmInfo: {svmResult, dataSetScaleInfo, normalizationInfo, optimizationMethod} 

	svmResult: {singleSvmResult1, singleSvmResult2, ..., singleSvmResult<NumberOfOutputComponents of dataSet>}
	singleSvmResult: {alphas, dataSetInputs, kernelFunction, b}
	alphas: {alpha1, alpha2, ..., alpha<Length[inputs]>}
	dataSetInputs: {input1, input2, ..., input<Length[dataSet]>}
	input: {inputComponent1, ..., inputComponent<NumberOfComponentsInDataSetInput>}
	dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs}, see GetDataSetScaleInfo
	normalizationInfo: {normalizationType, meanAndStandardDeviationList}, see CIP`DataTransformation`GetDataMatrixNormalizationInfo
	optimizationMethod: Optimization method
-----------------------------------------------------------------------
*)

(* ::Section:: *)
(* Package and dependencies *)

BeginPackage["CIP`SVM`", {"CIP`Utility`", "CIP`Graphics`", "CIP`DataTransformation`", "CIP`Cluster`"}]

(* ::Section:: *)
(* Off settings *)

Off[General::"spell1"]
Off[General::shdw]
Off[General::compat]
Off[FindMaximum::eit]
Off[NMaximize::cvmit]

(* ::Section:: *)
(* Options *)

Options[SvmOptionsTraining] = 
{
	(* Optimization method: "NMaximize", "FindMaximum", "QpSolve" *)
    SvmOptionOptimizationMethod -> "NMaximize"
}

Options[SvmOptionsOptimization] = 
{
	(* Initial list of alphas to be improved (may be empty list)
	   initialAlphasList: {initialAlphas1, initialAlphas2, ..., initialAlphas<Number of output components>}
	   initialAlphas: {alpha1, alpha2, ..., alpha<Length[inputs]>} *)
	SvmOptionInitialAlphasList -> {},
	
    SvmOptionObjectiveFunctionEpsilon -> 0.001,
    
    SvmOptionMaximizationPrecision -> 3,
    
    SvmOptionMaximumIterations -> 20,
    
    SvmOptionIsPostProcess -> True,

    SvmOptionScalingFactor -> 0.6,

	SvmOptionAlphaValueLimit -> 50.0
}

Options[SvmOptionsOpSolveOptimization] = 
{
    SvmOptionEpsilonQpSolve -> 0.001,
    
    SvmOptionTauQpSolve -> 0.001,

	SvmOptionPenaltyConstantQpSolve -> 0.5
}

(* ::Section:: *)
(* Declarations *)

(* For testing purposes *)
FitSingleSvmWithQpSolve::usage = 
	"FitSingleSvmWithQpSolve[dataSet, kernelFunction, options]"

(* For testing purposes *)
GetInternalSvmValue::usage = 
	"GetInternalSvmValue[input, singleSvmResult]"

CalculateSvmValue2D::usage = 
	"CalculateSvmValue2D[argumentValue, indexOfInput, indexOfOutput, input, svmInfo]"

CalculateSvmValue3D::usage = 
	"CalculateSvmValue3D[argumentValue1, argumentValue2, indexOfInput1, indexOfInput2, indexOfOutput, input, svmInfo]"

CalculateSvmClassNumber::usage = 
	"CalculateSvmClassNumber[input, svmInfo]"

CalculateSvmClassNumbers::usage = 
	"CalculateSvmClassNumbers[inputs, svmInfo]"

CalculateSvmDataSetRmse::usage = 
	"CalculateSvmDataSetRmse[dataSet, svmInfo]"

CalculateSvmOutput::usage = 
	"CalculateSvmOutput[input, svmInfo]"

CalculateSvmOutputs::usage = 
	"CalculateSvmOutputs[inputs, svmInfo]"

FitSvm::usage = 
	"FitSvm[dataSet, kernelFunction, options]"

FitSvmSeries::usage = 
	"FitSvmSeries[dataSet, kernelFunctionList, options]"
	
GetBestSvmClassOptimization::usage = 
	"GetBestSvmClassOptimization[svmTrainOptimization, options]"

GetBestSvmRegressOptimization::usage = 
	"GetBestSvmRegressOptimization[svmTrainOptimization, options]"

GetKernelFunction::usage = 
	"GetKernelFunction[svmInfo]"

GetSvmInputInclusionClass::usage = 
	"GetSvmInputInclusionClass[trainingAndTestSet, kernelFunction, options]"

GetSvmInputInclusionRegress::usage = 
	"GetSvmInputInclusionRegress[trainingAndTestSet, kernelFunction, options]"

GetSvmInputRelevanceClass::usage = 
	"GetSvmInputRelevanceClass[trainingAndTestSet, kernelFunction, options]"

GetSvmClassRelevantComponents::usage = 
    "GetSvmClassRelevantComponents[svmInputComponentRelevanceListForClassification, numberOfComponents]"

GetSvmInputRelevanceRegress::usage = 
	"GetSvmInputRelevanceRegress[trainingAndTestSet, kernelFunction, options]"

GetSvmRegressRelevantComponents::usage = 
    "GetSvmRegressRelevantComponents[svmInputComponentRelevanceListForRegression, numberOfComponents]"

GetSvmRegressionResult::usage = 
	"GetSvmRegressionResult[namedProperty, dataSet, svmInfo, options]"

GetSvmSeriesClassificationResult::usage = 
	"GetSvmSeriesClassificationResult[trainingAndTestSet, svmInfoList]"

GetSvmSeriesRmse::usage = 
	"GetSvmSeriesRmse[trainingAndTestSet, svmInfoList]"

GetSvmTrainOptimization::usage = 
	"GetSvmTrainOptimization[dataSet, kernelFunction, trainingFraction, numberOfTrainingSetOptimizationSteps, options]"

KernelGaussianRbf::usage = 
	"KernelGaussianRbf[u, v, beta]"

KernelPolynomial::usage = 
	"KernelPolynomial[u, v, c, d]"

KernelUniversalFourier::usage = 
	"KernelUniversalFourier[u, v, q]"

KernelWavelet::usage = 
	"KernelWavelet[u, v, a]"

ScanClassTrainingWithSvm::usage = 
	"ScanClassTrainingWithSvm[dataSet, kernelFunction, trainingFractionList, options]"

ScanRegressTrainingWithSvm::usage = 
	"ScanRegressTrainingWithSvm[dataSet, kernelFunction, trainingFractionList, options]"

ShowSvmOutput3D::usage = 
	"ShowSvmOutput3D[indexOfInput1, indexOfInput2, indexOfOutput, input, svmInfo, options]"

ShowSvmClassificationResult::usage = 
	"ShowSvmClassificationResult[namedPropertyList, trainingAndTestSet, svmInfo]"

ShowSvmSingleClassification::usage = 
	"ShowSvmSingleClassification[namedPropertyList, dataSet, svmInfo]"

ShowSvmClassificationScan::usage = 
	"ShowSvmClassificationScan[svmClassificationScan, options]"

ShowSvmInputRelevanceClass::usage = 
	"ShowSvmInputRelevanceClass[svmInputComponentRelevanceListForClassification, options]"
	
ShowSvmInputRelevanceRegress::usage = 
	"ShowSvmInputRelevanceRegress[svmInputComponentRelevanceListForRegression, options]"

ShowSvmRegressionResult::usage = 
	"ShowSvmRegressionResult[namedPropertyList, trainingAndTestSet, svmInfo]"

ShowSvmSingleRegression::usage = 
	"ShowSvmSingleRegression[namedPropertyList, dataSet, svmInfo]"

ShowSvmRegressionScan::usage = 
	"ShowSvmRegressionScan[svmRegressionScan, options]"

ShowSvmSeriesClassificationResult::usage = 
	"ShowSvmSeriesClassificationResult[svmSeriesClassificationResult, options]"

ShowSvmSeriesRmse::usage = 
	"ShowSvmSeriesRmse[svmSeriesRmse, options]"

ShowSvmTrainOptimization::usage = 
	"ShowSvmTrainOptimization[svmTrainOptimization, options]"

(* ::Section:: *)
(* Functions *)

Begin["`Private`"]

CalculateSvmValue2D[

	(* Calculates 2D output for specified argument and input for specified SVM.
	   This special method assumes an input and an output with one component only.

	   Returns:
	   Value of specified output component for argument *)


    (* Argument value for input component with index indexOfInput *)
    argumentValue_?NumberQ,
    
	(* See "Frequently used data structures" *)
    svmInfo_
    
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
			CalculateSvmValue2D[argumentValue, indexOfInput, indexOfFunctionValueOutput, input, svmInfo]
		]
	];

CalculateSvmValue2D[

	(* Calculates 2D output for specified argument and input for specified SVM.

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
    svmInfo_
    
	] :=
  
	Module[
    
		{
			currentInput,
			output
		},
		
		currentInput = ReplacePart[input, {indexOfInput -> argumentValue}];
		output = CalculateSvmOutput[currentInput, svmInfo];
		Return[output[[indexOfOutput]]];
	];

CalculateSvmValue3D[

	(* Calculates 3D output for specified arguments for specified SVM. 
	   This specific methods assumes a SVM with input vector of length 2 and an output vector of length 1.

	   Returns:
	   Value of specified output component for input *)


    (* Argument value for input component with index indexOfInput1 *)
    argumentValue1_?NumberQ,
    
    (* Argument value for input component with index indexOfInput2 *)
    argumentValue2_?NumberQ,
    
	(* See "Frequently used data structures" *)
    svmInfo_
    
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
			CalculateSvmValue3D[argumentValue1, argumentValue2, indexOfInput1, indexOfInput2, indexOfOutput, input, svmInfo]
		];
	];

CalculateSvmValue3D[

	(* Calculates 3D output for specified arguments and input for specified SVM.

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
    svmInfo_
    
	] :=
  
	Module[
    
		{
			currentInput,
			output
		},
		
		currentInput = ReplacePart[input, {indexOfInput1 -> argumentValue1, indexOfInput2 -> argumentValue2}];
		output = CalculateSvmOutput[currentInput, svmInfo];
		Return[output[[indexOfOutput]]];
	];

CalculateSvmClassNumber[

	(* Returns class number for specified input for classification SVM.

	   Returns:
	   Class number of input *)

    
    (* Input in original units: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    input_/;VectorQ[input, NumberQ],
    
	(* See "Frequently used data structures" *)
    svmInfo_
    
	] :=
  
	Module[
    
		{
			combinedOutputs,
			i,
			normalizationInfo,
			svmResult,
			singleSvmResult,
			dataSetScaleInfo,
			scaledInputs
		},
    
		svmResult = svmInfo[[1]];
    	dataSetScaleInfo = svmInfo[[2]];
    	normalizationInfo = svmInfo[[3]];

		combinedOutputs =
			Table[
				singleSvmResult = svmResult[[i]];
				(* Transform original input *)
				scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo];
				GetInternalSvmValue[scaledInputs[[1]], singleSvmResult],
				
				{i, Length[svmResult]}
			];
		Return[CIP`Utility`GetPositionOfMaximumValue[combinedOutputs]]
	];

CalculateSvmClassNumbers[

	(* Returns class numbers for specified inputs for classification SVM.

	   Returns:
	   {class number of input1, class number of input2, ...} *)

    
    (* {inputsInOriginalUnit1, inputsInOriginalUnit2, ...}
        inputsInOriginalUnit: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    inputs_/;MatrixQ[inputs, NumberQ],
    
	(* See "Frequently used data structures" *)
    svmInfo_
    
	] :=
  
	Module[
    
		{
			combinedOutputs,
			correspondingOutput,
			i,
			normalizationInfo,
			svmResult,
			singleSvmResult,
			dataSetScaleInfo,
			scaledInputs
		},
    
		svmResult = svmInfo[[1]];
    	dataSetScaleInfo = svmInfo[[2]];
    	normalizationInfo = svmInfo[[3]];

		combinedOutputs =
			Table[
				singleSvmResult = svmResult[[i]];
				(* Transform original input *)
				scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
				GetInternalSvmValues[scaledInputs, singleSvmResult],
				
				{i, Length[svmResult]}
			];
		Return[
			Table[
				correspondingOutput = combinedOutputs[[All, i]];
				CIP`Utility`GetPositionOfMaximumValue[correspondingOutput],
			
				{i, Length[First[combinedOutputs]]}
			]
		]
	];

CalculateSvmCorrectClassificationInPercent[

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
    svmInfo_
    
	] :=
  
	Module[
    
		{
			pureFunction
		},

		pureFunction = Function[inputs, CalculateSvmClassNumbers[inputs, svmInfo]];
		Return[CIP`Utility`GetCorrectClassificationInPercent[classificationDataSet, pureFunction]]
	];

CalculateSvmDataSetRmse[

	(* Returns RMSE of data set.

	   Returns: 
	   RMSE of data set *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* See "Frequently used data structures" *)
    svmInfo_
    
	] :=
  
	Module[
    
		{
			pureFunction,
			rmse
		},

		pureFunction = Function[inputs, CalculateSvmOutputs[inputs, svmInfo]];
		rmse = Sqrt[CIP`Utility`GetMeanSquaredError[dataSet, pureFunction]];
		Return[rmse]
	];

CalculateSvmOutput[

	(* Calculates output for specified input for SVM.

	   Returns:
	   output: {transformedOutputValue1, transformedOutputValue2, ...} *)

    
    (* Input in original units: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    input_/;VectorQ[input, NumberQ],
    
	(* See "Frequently used data structures" *)
    svmInfo_
    
	] :=
  
	Module[
    
		{
			dataMatrixScaleInfo,
			dataSetScaleInfo,
			combinedOutputs,
			i,
			normalizationInfo,
			svmResult,
			scaledOutputValue,
			outputsInOriginalUnits,
			singleSvmResult,
			scaledInputs
		},
    
		svmResult = svmInfo[[1]];
    	dataSetScaleInfo = svmInfo[[2]];
    	normalizationInfo = svmInfo[[3]];

		combinedOutputs =
			Table[
				singleSvmResult = svmResult[[i]];
				(* Transform original input *)
				scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[{input}, dataSetScaleInfo[[1]], normalizationInfo];
				scaledOutputValue = GetInternalSvmValue[scaledInputs[[1]], singleSvmResult];
				(* Transform outputs to original units:
				   dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs} 
				   dataMatrixScaleInfo: {minMaxList, targetInterval} *)					
				dataMatrixScaleInfo = {{dataSetScaleInfo[[2, 1, i]]}, dataSetScaleInfo[[2, 2]]};
				outputsInOriginalUnits = CIP`DataTransformation`ScaleDataMatrixReverse[{{scaledOutputValue}}, dataMatrixScaleInfo];
				outputsInOriginalUnits[[1, 1]],
				
				{i, Length[svmResult]}
			];
		Return[combinedOutputs]
	];

CalculateSvmOutputs[

	(* Calculates outputs for specified inputs for SVM.

	   Returns:
	   outputs = {output1, ..., output<Length[inputs]>}
	   output: {transformedOutputValue1, transformedOutputValue2, ...} *)

    
    (* {inputsInOriginalUnit1, inputsInOriginalUnit2, ...}
        inputsInOriginalUnit: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    inputs_/;MatrixQ[inputs, NumberQ],
    
	(* See "Frequently used data structures" *)
    svmInfo_
    
	] :=
  
	Module[
    
		{
			combinedOutputs,
			dataMatrixScaleInfo,
			dataSetScaleInfo,
			i,
			k,
			normalizationInfo,
			svmResult,
			scaledOutputs,
			outputsInOriginalUnits,
			singleSvmResult,
			scaledInputs,
			scaledValues
		},
    
		svmResult = svmInfo[[1]];
    	dataSetScaleInfo = svmInfo[[2]];
    	normalizationInfo = svmInfo[[3]];

		combinedOutputs =
			Table[
				singleSvmResult = svmResult[[i]];
				(* Transform original input *)
				scaledInputs = CIP`DataTransformation`ScaleAndNormalizeDataMatrix[inputs, dataSetScaleInfo[[1]], normalizationInfo];
				scaledValues = GetInternalSvmValues[scaledInputs, singleSvmResult];
				(* Set correct list structure: Transform values to outputValues *)
				scaledOutputs = Table[{scaledValues[[k]]}, {k, Length[scaledValues]}];
				(* Transform outputs to original units:
				   dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs} 
				   dataMatrixScaleInfo: {minMaxList, targetInterval} *)					
				dataMatrixScaleInfo = {{dataSetScaleInfo[[2, 1, i]]}, dataSetScaleInfo[[2, 2]]};
				outputsInOriginalUnits = CIP`DataTransformation`ScaleDataMatrixReverse[scaledOutputs, dataMatrixScaleInfo];
				Flatten[outputsInOriginalUnits],
				
				{i, Length[svmResult]}
			];
		Return[Transpose[combinedOutputs]]
	];

FitSingleSvmWitFindMaximum[

	(* Trains support vector machine with FindMaximum for regression task.

	   Returns: 
	   singleSvmResult: {alphas, dataSetInputs, kernelFunction, b}
	   alphas: {alpha1, alpha2, ..., alpha<Length[inputs]>}
	   dataSetInputs: {input1, input2, ..., input<Length[dataSet]>}
	   input: {inputValue1, ..., inputValue<Length[input]>} *)
	   
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue}
	   NOTE: output is ONLY allowed to contain ONE value *)
    dataSet_,

	(* For details see method Kernel[] *)
	kernelFunction_,
	
	(* Initial alpha values *)
	initialAlphas_,
	
	(* Options *)
	opts___
		
	] :=
  
	Module[
    
		{
			alpha,
			alphasStartValues,
			b,
			constraints, 
			alphaValueLimit,
			i, 
			inputs,
			j, 
			matrixForObjectiveFunction, 
			maximizationPrecision,
			maximumNumberOfIterations,
			numberOfInputVectors, 
			numberOfInputValues, 
			objectiveFunction, 
			objectiveFunctionEpsilon,
			output,
			knownAlphasRules, 
			unknownAlphas
		},
		
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];

		(* ----------------------------------------------------------------------------------------------------
		   Training
		   ---------------------------------------------------------------------------------------------------- *)
		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];
		(* Length[output] = Length[inputs], i.e. output[[i]] corresponds to inputs[[i]] *)
		output = Flatten[CIP`Utility`GetOutputsOfDataSet[dataSet]];
				
		numberOfInputVectors = Length[inputs]; 
		numberOfInputValues = Length[inputs[[1]]];
		
		matrixForObjectiveFunction = 
			Table[
				Kernel[inputs[[i]], inputs[[j]], kernelFunction], 
					
				{i, numberOfInputVectors}, {j, numberOfInputVectors}
			] + (1/alphaValueLimit)*IdentityMatrix[numberOfInputVectors];
			
		objectiveFunction = 
			Sum[Subscript[alpha, i]*output[[i]], {i, numberOfInputVectors}] - 
			objectiveFunctionEpsilon*Sum[Abs[Subscript[alpha, i]], {i, numberOfInputVectors}] - 
			0.5*Sum[Subscript[alpha, i]*Subscript[alpha, j]*matrixForObjectiveFunction[[i, j]], {i, numberOfInputVectors}, {j, numberOfInputVectors}];
		constraints = 
			Apply[And, 
				Join[
					Table[-alphaValueLimit < Subscript[alpha, i] <= alphaValueLimit, {i, numberOfInputVectors}], 
					{Sum[Subscript[alpha, i], {i, numberOfInputVectors}] == 0}
				]
			];
		unknownAlphas = Table[Subscript[alpha, i], {i, numberOfInputVectors}];
		If[Length[initialAlphas] > 0,

			alphasStartValues = 
				Table[
					{unknownAlphas[[i]], initialAlphas[[i]]},
					
					{i, Length[unknownAlphas]}
				],

			alphasStartValues = unknownAlphas
		];
		knownAlphasRules = 
			FindMaximum[
				{objectiveFunction, constraints}, 
				alphasStartValues,
				Method -> "InteriorPoint",
				WorkingPrecision -> MachinePrecision,
				AccuracyGoal -> maximizationPrecision, 
				PrecisionGoal -> maximizationPrecision,
				MaxIterations -> maximumNumberOfIterations
			][[2]];
		b = 
			(1/numberOfInputVectors)*Apply[Plus, 
				Table[
					(
						output[[j]] - 
						Sum[Subscript[alpha, i]*Kernel[inputs[[i]], inputs[[j]], kernelFunction], {i, numberOfInputVectors}] - 
						objectiveFunctionEpsilon - 
						(Subscript[alpha, j]/alphaValueLimit)
					)/.knownAlphasRules, 
						
					{j, numberOfInputVectors}
				]
			];

		(* ----------------------------------------------------------------------------------------------------
		   Return results
		   ---------------------------------------------------------------------------------------------------- *)
		Return[
			{unknownAlphas/.knownAlphasRules, inputs, kernelFunction, b}
		]
	];

FitSingleSvmWithNMaximize[

	(* Trains support vector machine with NMaximize for regression task.

	   Returns: 
	   singleSvmResult: {alphas, dataSetInputs, kernelFunction, b}
	   alphas: {alpha1, alpha2, ..., alpha<Length[inputs]>}
	   dataSetInputs: {input1, input2, ..., input<Length[dataSet]>}
	   input: {inputValue1, ..., inputValue<Length[input]>} *)
	   
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue}
	   NOTE: output is ONLY allowed to contain ONE value *)
    dataSet_,

	(* For details see method Kernel[] *)
	kernelFunction_,
	
	(* Options *)
	opts___
		
	] :=
  
	Module[
    
		{
			alpha,
			b,
			constraints, 
			alphaValueLimit,
			i, 
			inputs,
			j, 
			matrixForObjectiveFunction, 
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			numberOfInputVectors, 
			numberOfInputValues, 
			objectiveFunction, 
			objectiveFunctionEpsilon,
			output,
			knownAlphasRules, 
			unknownAlphas
		},
		
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];

		(* ----------------------------------------------------------------------------------------------------
		   Training
		   ---------------------------------------------------------------------------------------------------- *)
		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];
		(* Length[output] = Length[inputs], i.e. output[[i]] corresponds to inputs[[i]] *)
		output = Flatten[CIP`Utility`GetOutputsOfDataSet[dataSet]];
				
		numberOfInputVectors = Length[inputs]; 
		numberOfInputValues = Length[inputs[[1]]];
		
		matrixForObjectiveFunction = 
			Table[
				Kernel[inputs[[i]], inputs[[j]], kernelFunction], 
					
				{i, numberOfInputVectors}, {j, numberOfInputVectors}
			] + (1/alphaValueLimit)*IdentityMatrix[numberOfInputVectors];
			
		objectiveFunction = 
			Sum[Subscript[alpha, i]*output[[i]], {i, numberOfInputVectors}] - 
			objectiveFunctionEpsilon*Sum[Abs[Subscript[alpha, i]], {i, numberOfInputVectors}] - 
			0.5*Sum[Subscript[alpha, i]*Subscript[alpha, j]*matrixForObjectiveFunction[[i, j]], {i, numberOfInputVectors}, {j, numberOfInputVectors}];
		constraints = 
			Apply[And, 
				Join[
					Table[-alphaValueLimit < Subscript[alpha, i] <= alphaValueLimit, {i, numberOfInputVectors}], 
					{Sum[Subscript[alpha, i], {i, numberOfInputVectors}] == 0}
				]
			];
		unknownAlphas = Table[Subscript[alpha, i], {i, numberOfInputVectors}];
		knownAlphasRules = 
			NMaximize[
				{objectiveFunction, constraints}, 
				unknownAlphas,
				WorkingPrecision -> MachinePrecision,
				AccuracyGoal -> maximizationPrecision, 
				PrecisionGoal -> maximizationPrecision,
				MaxIterations -> maximumNumberOfIterations,
				Method -> {"DifferentialEvolution", "PostProcess" -> isPostProcess, "ScalingFactor" -> scalingFactor}
			][[2]];
		b = 
			(1/numberOfInputVectors)*Apply[Plus, 
				Table[
					(
						output[[j]] - 
						Sum[Subscript[alpha, i]*Kernel[inputs[[i]], inputs[[j]], kernelFunction], {i, numberOfInputVectors}] - 
						objectiveFunctionEpsilon - 
						(Subscript[alpha, j]/alphaValueLimit)
					)/.knownAlphasRules, 
						
					{j, numberOfInputVectors}
				]
			];

		(* ----------------------------------------------------------------------------------------------------
		   Return results
		   ---------------------------------------------------------------------------------------------------- *)
		Return[
			{unknownAlphas/.knownAlphasRules, inputs, kernelFunction, b}
		]
	];

FitSingleSvmWithQpSolve[

	(* Trains support vector machine with QpSolve for regression task.

	   Returns: 
	   singleSvmResult: {alphas, dataSetInputs, kernelFunction, b}
	   alphas: {alpha1, alpha2, ..., alpha<Length[inputs]>}
	   dataSetInputs: {input1, input2, ..., input<Length[dataSet]>}
	   input: {inputValue1, ..., inputValue<Length[input]>} *)

	
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue}
	   NOTE: output is ONLY allowed to contain ONE value *)
    dataSet_,
	
	(* For details see method Kernel[] *)
	kernelFunction_,
	
	(* Options *)
	opts___
	
	] := 
	
	Module[
		
		{
			inputs,
			output,
			reducedAlphas,
			b,
			alphas,
			i,
			k,
			numberOfInputs,
			kernelMatrix,
			extendedInputs,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];

		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];
		(* Length[output] = Length[inputs], i.e. output[[i]] corresponds to inputs[[i]] *)
		output = Flatten[CIP`Utility`GetOutputsOfDataSet[dataSet]];
		
		numberOfInputs = Length[inputs];

		extendedInputs =  Join[inputs, -inputs];
		kernelMatrix = 
			Table[
				Kernel[extendedInputs[[i]], extendedInputs[[k]], kernelFunction],
				
				{i, 2*numberOfInputs}, {k, 2*numberOfInputs}
			];

		alphas =
			QpSolve[
				kernelMatrix, 
				Table[epsilonQpSolve, {2*numberOfInputs}] + Join[output, -output],
				Join[Table[0.0, {2*numberOfInputs}]], 
				Join[Table[penaltyConstantQpSolve, {2*numberOfInputs}]], 
				0.0, 
				Join[Table[1.0, {numberOfInputs}], Table[-1.0, {numberOfInputs}]], 
				tauQpSolve
			];
			
		b = QpRegressionBias[alphas, inputs, output, epsilonQpSolve, kernelFunction];
		
		reducedAlphas = 
			Table[
				alphas[[i + numberOfInputs]] - alphas[[i]], 
				
				{i, numberOfInputs}
			];
			
		Return[
			ReduceSingleSvmResult[
				{reducedAlphas, inputs, kernelFunction, b}
			]
		]
	];

FitSvm[

	(* Trains multiple support vector machines (SVM): 1 SVM per output component of data set.

	   Returns: 
	   svmInfo: See "Frequently used data structures" *)
	   
	   
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* For details see method Kernel[] *)
	kernelFunction_,
	
	(* Options *)
	opts___
		
	] :=
  
	Module[
    
		{
			targetInterval,
			normalizationType,
			alphaValueLimit,
			initialAlphasList,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			objectiveFunctionEpsilon,
			optimizationMethod,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,
			parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
		(* Parallelization options *)
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				FitSvmPC[
					dataSet,
					kernelFunction,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
					SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
					SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve,
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				FitSvmSC[
					dataSet,
					kernelFunction,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
					SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
					SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve,
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve
				]
			]
		]
	];

FitSvmSC[

	(* Trains multiple support vector machines (SVM): 1 SVM per output component of data set.

	   Returns: 
	   svmInfo: See "Frequently used data structures" *)
	   
	   
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* For details see method Kernel[] *)
	kernelFunction_,
	
	(* Options *)
	opts___
		
	] :=
  
	Module[
    
		{
			dataSetScaleInfo,
			targetInterval,
			alphaValueLimit,
			i,
			initialAlphas,
			initialAlphasList,
			maximizationPrecision,
			maximumNumberOfIterations,
			normalizationType,
			normalizationInfo,
			isPostProcess,
			scalingFactor,
			multipleScaledDataSet,
			scaledDataSet,
			svmResult,
			objectiveFunctionEpsilon,
			singleSvmResult,
			optimizationMethod,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];


		(* ----------------------------------------------------------------------------------------------------
		   Training
		   ---------------------------------------------------------------------------------------------------- *)
		dataSetScaleInfo = CIP`DataTransformation`GetDataSetScaleInfo[dataSet, targetInterval, targetInterval];
		normalizationInfo = CIP`DataTransformation`GetDataSetNormalizationInfo[dataSet, normalizationType, dataSetScaleInfo];
		scaledDataSet = CIP`DataTransformation`ScaleAndNormalizeDataSet[dataSet, dataSetScaleInfo, normalizationInfo];
		multipleScaledDataSet = CIP`DataTransformation`TransformDataSetToMultipleDataSet[scaledDataSet];
		svmResult = {};
		Do[
			If[Length[initialAlphasList] > 0,
				
				initialAlphas = initialAlphasList[[i]],
				
				initialAlphas = {}
			];
			Switch[optimizationMethod,
				
				"NMaximize",
				singleSvmResult = 
					FitSingleSvmWithNMaximize[
						multipleScaledDataSet[[i]],
						kernelFunction,
						SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon, 
						SvmOptionMaximizationPrecision -> maximizationPrecision,
						SvmOptionMaximumIterations -> maximumNumberOfIterations,
						SvmOptionIsPostProcess -> isPostProcess,
						SvmOptionScalingFactor -> scalingFactor,
						SvmOptionAlphaValueLimit -> alphaValueLimit
					],
					
				"FindMaximum",
				singleSvmResult = 
					FitSingleSvmWitFindMaximum[
						multipleScaledDataSet[[i]],
						kernelFunction,
						initialAlphas,
						SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon, 
						SvmOptionMaximizationPrecision -> maximizationPrecision,
						SvmOptionMaximumIterations -> maximumNumberOfIterations,
						SvmOptionAlphaValueLimit -> alphaValueLimit
					],
					
				"QpSolve",
				singleSvmResult = 
					FitSingleSvmWithQpSolve[
						multipleScaledDataSet[[i]],
						kernelFunction,
						SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
						SvmOptionTauQpSolve -> tauQpSolve,
						SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve
					]
					
			];
			AppendTo[svmResult, singleSvmResult],
			
			{i, Length[multipleScaledDataSet]}
		];

		(* ----------------------------------------------------------------------------------------------------
		   Return results
		   ---------------------------------------------------------------------------------------------------- *)
    	Return[{svmResult, dataSetScaleInfo, normalizationInfo, optimizationMethod}]
	];
	
FitSvmPC[

	(* Trains multiple support vector machines (SVM): 1 SVM per output component of data set.

	   Returns: 
	   svmInfo: See "Frequently used data structures" *)
	   
	   
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* For details see method Kernel[] *)
	kernelFunction_,
	
	(* Options *)
	opts___
		
	] :=
  
	Module[
    
		{
			dataSetScaleInfo,
			targetInterval,
			alphaValueLimit,
			i,
			initialAlphas,
			initialAlphasList,
			maximizationPrecision,
			maximumNumberOfIterations,
			normalizationInfo,
			normalizationType,
			isPostProcess,
			scalingFactor,
			multipleScaledDataSet,
			scaledDataSet,
			svmResult,
			objectiveFunctionEpsilon,
			singleSvmResult,
			optimizationMethod,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
		
		(* ----------------------------------------------------------------------------------------------------
		   Training
		   ---------------------------------------------------------------------------------------------------- *)
		dataSetScaleInfo = CIP`DataTransformation`GetDataSetScaleInfo[dataSet, targetInterval, targetInterval];
		normalizationInfo = CIP`DataTransformation`GetDataSetNormalizationInfo[dataSet, normalizationType, dataSetScaleInfo];
		scaledDataSet = CIP`DataTransformation`ScaleAndNormalizeDataSet[dataSet, dataSetScaleInfo, normalizationInfo];
		multipleScaledDataSet = CIP`DataTransformation`TransformDataSetToMultipleDataSet[scaledDataSet];
		
		ParallelNeeds[{"CIP`SVM`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[optimizationMethod, multipleScaledDataSet, objectiveFunctionEpsilon, maximizationPrecision, 
			maximumNumberOfIterations, isPostProcess, scalingFactor, alphaValueLimit, initialAlphasList, 
			epsilonQpSolve, tauQpSolve, penaltyConstantQpSolve];
		
		svmResult = ParallelTable[
			
			If[Length[initialAlphasList] > 0,
				
				initialAlphas = initialAlphasList[[i]],
				
				initialAlphas = {}
			];
			Switch[optimizationMethod,
				
				"NMaximize",
				singleSvmResult = 
					FitSingleSvmWithNMaximize[
						multipleScaledDataSet[[i]],
						kernelFunction,
						SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon, 
						SvmOptionMaximizationPrecision -> maximizationPrecision,
						SvmOptionMaximumIterations -> maximumNumberOfIterations,
						SvmOptionIsPostProcess -> isPostProcess,
						SvmOptionScalingFactor -> scalingFactor,
						SvmOptionAlphaValueLimit -> alphaValueLimit
					],
					
				"FindMaximum",
				singleSvmResult = 
					FitSingleSvmWitFindMaximum[
						multipleScaledDataSet[[i]],
						kernelFunction,
						initialAlphas,
						SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon, 
						SvmOptionMaximizationPrecision -> maximizationPrecision,
						SvmOptionMaximumIterations -> maximumNumberOfIterations,
						SvmOptionAlphaValueLimit -> alphaValueLimit
					],
					
				"QpSolve",
				singleSvmResult = 
					FitSingleSvmWithQpSolve[
						multipleScaledDataSet[[i]],
						kernelFunction,
						SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
						SvmOptionTauQpSolve -> tauQpSolve,
						SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve
					]
			],
			
			
			{i, Length[multipleScaledDataSet]}
		];

		(* ----------------------------------------------------------------------------------------------------
		   Return results
		   ---------------------------------------------------------------------------------------------------- *)
    	Return[{svmResult, dataSetScaleInfo, normalizationInfo, optimizationMethod}]
	];

FitSvmSeries[

	(* Trains a series of multiple support vector machines (SVM): 1 SVM per output component of data set.

	   Returns: 
	   svmInfoList: {svmInfo1, svmInfo2, ...}
	   svmInfo[[i]] corresponds to kernelFunctionList[[i]]
	   svmInfo: See "Frequently used data structures" *)
	   
	   
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* {kernelFunction1, kernelFunction2, ...}
	   For details see method Kernel[] *)
	kernelFunctionList_,
	
	(* Options *)
	opts___
		
	] :=
  
	Module[
    
		{
			targetInterval,
			normalizationType,
			alphaValueLimit,
			initialAlphasList,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			objectiveFunctionEpsilon,
			optimizationMethod,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,
			parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    (* Parallelization options *)
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				FitSvmSeriesPC[
					dataSet,
					kernelFunctionList,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
					SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
					SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve,
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				FitSvmSeriesSC[
					dataSet,
					kernelFunctionList,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
					SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
					SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve,
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve
				]
			]
		]
	];

FitSvmSeriesSC[

	(* Trains a series of multiple support vector machines (SVM): 1 SVM per output component of data set.

	   Returns: 
	   svmInfoList: {svmInfo1, svmInfo2, ...}
	   svmInfo[[i]] corresponds to kernelFunctionList[[i]]
	   svmInfo: See "Frequently used data structures" *)
	   
	   
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* {kernelFunction1, kernelFunction2, ...}
	   For details see method Kernel[] *)
	kernelFunctionList_,
	
	(* Options *)
	opts___
		
	] :=
  
	Module[
    
		{
			i,
			targetInterval,
			normalizationType,
			alphaValueLimit,
			initialAlphasList,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			objectiveFunctionEpsilon,
			optimizationMethod,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];

		Return[
			Table[
				FitSvm[
					dataSet,
					kernelFunctionList[[i]],
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
	   				SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve
				],
			
				{i, Length[kernelFunctionList]}
			]			
		]
	];
	
FitSvmSeriesPC[

	(* Trains a series of multiple support vector machines (SVM): 1 SVM per output component of data set.

	   Returns: 
	   svmInfoList: {svmInfo1, svmInfo2, ...}
	   svmInfo[[i]] corresponds to kernelFunctionList[[i]]
	   svmInfo: See "Frequently used data structures" *)
	   
	   
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* {kernelFunction1, kernelFunction2, ...}
	   For details see method Kernel[] *)
	kernelFunctionList_,
	
	(* Options *)
	opts___
		
	] :=
  
	Module[
    
		{
			i,
			targetInterval,
			normalizationType,
			alphaValueLimit,
			initialAlphasList,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			objectiveFunctionEpsilon,
			optimizationMethod,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
		
		ParallelNeeds[{"CIP`SVM`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[optimizationMethod, initialAlphasList, objectiveFunctionEpsilon,
			maximizationPrecision, maximumNumberOfIterations, isPostProcess, scalingFactor, alphaValueLimit, 
			targetInterval, normalizationType, epsilonQpSolve, tauQpSolve, penaltyConstantQpSolve];

		Return[
			ParallelTable[
				FitSvm[
					dataSet,
					kernelFunctionList[[i]],
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
	    			SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve
				],
				
				{i, Length[kernelFunctionList]}
			]
		]
	];

GetAlphasList[

	(* Returns list with alphas.

	   Returns:
	   alphasList: {alphas1, alphas2, ..., alphas<Number of output components>}
	   alphas: {alpha1, alpha2, ..., alpha<Length[inputs]>} *)


	(* See "Frequently used data structures" *)
    svmInfo_
    
	] :=
  
	Module[
    
		{
			alphas,
			alphasList,
			i,
			singleSvmResult,
			svmResult
		},

		svmResult = svmInfo[[1]];
		alphasList = {};
		Do[
			singleSvmResult = svmResult[[i]];
			alphas = singleSvmResult[[1]];
			AppendTo[alphasList, alphas],
			
			{i, Length[svmResult]}
		];
		Return[alphasList];
	];

GetBestSvmClassOptimization[

	(* Returns best training set optimization result of SVM for classification.

	   Returns: 
	   Best index for classification *)


	(* svmTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, svmInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   svmInfoList: List with svmInfo
	   svmInfoList[[i]] refers to optimization step i *)
	svmTrainOptimization_,
	
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
				GetBestSvmClassOptimizationPC[
					svmTrainOptimization,
					UtilityOptionBestOptimization -> bestOptimization
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetBestSvmClassOptimizationSC[
					svmTrainOptimization,
					UtilityOptionBestOptimization -> bestOptimization
				]
			]
		]
	];
	
GetBestSvmClassOptimizationSC[

	(* Returns best training set optimization result of SVM for classification.

	   Returns: 
	   Best index for classification *)


	(* svmTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, svmInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   svmInfoList: List with svmInfo
	   svmInfoList[[i]] refers to optimization step i *)
	svmTrainOptimization_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			k,
			trainingAndTestSetList,
			svmInfoList,
			maximumCorrectClassificationInPercent,
			svmInfo,
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
			trainingAndTestSetList = svmTrainOptimization[[3]];
			svmInfoList = svmTrainOptimization[[4]];
			maximumCorrectClassificationInPercent = -1.0;
			Do[
				testSet = trainingAndTestSetList[[k, 2]];
				svmInfo = svmInfoList[[k]];
				correctClassificationInPercent = CalculateSvmCorrectClassificationInPercent[testSet, svmInfo];
				If[correctClassificationInPercent > maximumCorrectClassificationInPercent,
					maximumCorrectClassificationInPercent = correctClassificationInPercent;
					bestIndex = k
				],
				
				{k, Length[svmInfoList]}
			],
			
			(* ------------------------------------------------------------------------------- *)			
			"MinimumDeviation",
			trainingAndTestSetList = svmTrainOptimization[[3]];
			svmInfoList = svmTrainOptimization[[4]];
			minimumDeviation = Infinity;
			Do[
				trainingSet = trainingAndTestSetList[[k, 1]];
				testSet = trainingAndTestSetList[[k, 2]];
				svmInfo = svmInfoList[[k]];
				trainingSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[trainingSet, svmInfo];
				testSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[testSet, svmInfo];
				deviation = Abs[testSetCorrectClassificationInPercent - trainingSetCorrectClassificationInPercent];
				If[deviation < minimumDeviation || (deviation == minimumDeviation && testSetCorrectClassificationInPercent < bestTestSetCorrectClassificationInPercent),
					minimumDeviation = deviation;
					bestTestSetCorrectClassificationInPercent = testSetCorrectClassificationInPercent;
					bestIndex = k
				],
				
				{k, Length[svmInfoList]}
			]
		];

		Return[bestIndex]
	];

GetBestSvmClassOptimizationPC[

	(* Returns best training set optimization result of SVM for classification.

	   Returns: 
	   Best index for classification *)


	(* svmTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, svmInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   svmInfoList: List with svmInfo
	   svmInfoList[[i]] refers to optimization step i *)
	svmTrainOptimization_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			k,
			trainingAndTestSetList,
			svmInfoList,
			maximumCorrectClassificationInPercent,
			svmInfo,
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
			trainingAndTestSetList = svmTrainOptimization[[3]];
			svmInfoList = svmTrainOptimization[[4]];
			maximumCorrectClassificationInPercent = -1.0;
			
			ParallelNeeds[{"CIP`SVM`", "CIP`DataTransformation`", "CIP`Utility`"}];
			DistributeDefinitions[trainingAndTestSetList, svmInfoList];
			
			correctClassificationInPercentList = ParallelTable[
				testSet = trainingAndTestSetList[[k, 2]];
				svmInfo = svmInfoList[[k]];

				CalculateSvmCorrectClassificationInPercent[testSet, svmInfo],
				
				{k, Length[svmInfoList]}
			];
			
			Do[
				If[correctClassificationInPercentList[[k]] > maximumCorrectClassificationInPercent,
					maximumCorrectClassificationInPercent = correctClassificationInPercentList[[k]];
					bestIndex = k
				],
				
				{k, Length[svmInfoList]}
			],
			
			(* ------------------------------------------------------------------------------- *)			
			"MinimumDeviation",
			trainingAndTestSetList = svmTrainOptimization[[3]];
			svmInfoList = svmTrainOptimization[[4]];
			minimumDeviation = Infinity;
			
			ParallelNeeds[{"CIP`SVM`", "CIP`DataTransformation`", "CIP`Utility`"}];
			DistributeDefinitions[trainingAndTestSetList, svmInfoList];
			
			listOfTestSetCorrectClassificationInPercentAndDeviation = ParallelTable[
				trainingSet = trainingAndTestSetList[[k, 1]];
				testSet = trainingAndTestSetList[[k, 2]];
				svmInfo = svmInfoList[[k]];
				trainingSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[trainingSet, svmInfo];
				testSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[testSet, svmInfo];
				deviation = Abs[testSetCorrectClassificationInPercent - trainingSetCorrectClassificationInPercent];
				
				{
					testSetCorrectClassificationInPercent,
					deviation
				},
				
				{k, Length[svmInfoList]}
			];
			
			Do[
				If[listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]] < minimumDeviation || (listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]] == minimumDeviation && listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]] < bestTestSetCorrectClassificationInPercent),
					minimumDeviation = listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 2]];
					bestTestSetCorrectClassificationInPercent = listOfTestSetCorrectClassificationInPercentAndDeviation[[k, 1]];
					bestIndex = k
				],
				
				{k, Length[svmInfoList]}
			]
		];

		Return[bestIndex]
	];

GetBestSvmRegressOptimization[

	(* Returns best optimization result of SVM for regression.

	   Returns: 
	   Best index for regression *)


	(* svmTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, svmInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   svmInfoList: List with svmInfo
	   svmInfoList[[i]] refers to optimization step i *)
	svmTrainOptimization_,
	
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
				svmTrainOptimization, 
				UtilityOptionBestOptimization -> bestOptimization
			]
		]
	];

GetInternalSvmValue[

	(* Returns output value of single SVM according to specified input.

	   Returns: outputValue *)


    (* {inputValue1, inputValue2, ..., inputValue<<numberOfInputValues>} *)
    input_/;VectorQ[input, NumberQ],

	(* singleSvmResult: {alphas, dataSetInputs, kernelFunction, b}
	   alphas: {alpha1, alpha2, ..., alpha<Length[inputs]>}
	   dataSetInputs: {input1, input2, ..., input<Length[dataSet]>}
	   input: {inputComponent1, ..., inputComponent<NumberOfComponentsInDataSetInput>} *)
	singleSvmResult_
	
	] :=
  
	Module[
    
		{
			b,
			alphas,
			i,
			dataSetInputs,
			kernelFunction
		},
		
		alphas = singleSvmResult[[1]];
		dataSetInputs = singleSvmResult[[2]];
		kernelFunction = singleSvmResult[[3]];
		b = singleSvmResult[[4]];

		Return[
			Sum[alphas[[i]]*Kernel[dataSetInputs[[i]], input, kernelFunction], {i, Length[dataSetInputs]}] + b
		]
	];

GetInternalSvmValues[

	(* Returns output values of single SVM according to specified inputs.

	   Returns: 
	   output: {outputValue1, ..., outputValue<Length[inputs]>} 
	   output[[i]] corresponds to inputs[[i]] *)

    (* inputs: {input1, input2, ...} 
       input: {inputValue1, inputValue2, ..., inputValue<<numberOfInputValues>} *)
    inputs_/;MatrixQ[inputs, NumberQ],

	(* singleSvmResult: {alphas, dataSetInputs, kernelFunction, b}
	   alphas: {alpha1, alpha2, ..., alpha<Length[inputs]>}
	   dataSetInputs: {input1, input2, ..., input<Length[dataSet]>}
	   input: {inputComponent1, ..., inputComponent<NumberOfComponentsInDataSetInput>} *)
	singleSvmResult_
	
	] :=
  
	Module[
    
		{
			b,
			alphas,
			i,
			k,
			dataSetInputs,
			kernelFunction
		},
		
		alphas = singleSvmResult[[1]];
		dataSetInputs = singleSvmResult[[2]];
		kernelFunction = singleSvmResult[[3]];
		b = singleSvmResult[[4]];

		Return[
			Table[
				Sum[alphas[[i]]*Kernel[dataSetInputs[[i]], inputs[[k]], kernelFunction], {i, Length[dataSetInputs]}] + b,
				
				{k, Length[inputs]}
			]
		]
	];

GetKernelFunction[

	(* Returns kernel function.

	   Returns:
	   Kernel function *)


	(* See "Frequently used data structures" *)
    svmInfo_
    
	] :=
  
	Module[
    
		{
			singleSvmResult,
			svmResult
		},

		svmResult = svmInfo[[1]];
		singleSvmResult = svmResult[[1]];
		Return[singleSvmResult[[3]]]
	];

GetSvmInputInclusionClass[

	(* Analyzes relevance of input components by successive get-one-in for classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   svmInputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, svmInfo}
	   trainingSetResult: {numberOfIncludedInputs, (best) classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, (best) classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* For details see method Kernel[] *)
	kernelFunction_,	
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			alphaValueLimit,
			initialAlphasList,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			objectiveFunctionEpsilon,
			optimizationMethod,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,
			targetInterval,
			normalizationType,
			isIntermediateOutput,
			numberOfInclusionsPerStepList,
			isRegression,
			inclusionStartList,
			parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Svm options *)   
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    (* DataTransformation options *)   
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
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
				GetSvmInputInclusionCalculationPC[
					trainingAndTestSet,
					kernelFunction,
					isRegression,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
			    	SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
    				SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetSvmInputInclusionCalculationSC[
					trainingAndTestSet,
					kernelFunction,
					isRegression,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
			    	SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
    				SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			]
		]
	];

GetSvmInputInclusionRegress[

	(* Analyzes relevance of input components by successive get-one-in for regression.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   svmInputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, svmInfo}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* For details see method Kernel[] *)
	kernelFunction_,	
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			alphaValueLimit,
			initialAlphasList,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			objectiveFunctionEpsilon,
			optimizationMethod,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,
			targetInterval,
			normalizationType,
			isIntermediateOutput,
			numberOfInclusionsPerStepList,
			isRegression,
			inclusionStartList,
			parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Svm options *)   
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    (* DataTransformation options *)   
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
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
				GetSvmInputInclusionCalculationPC[
					trainingAndTestSet,
					kernelFunction,
					isRegression,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
    				SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetSvmInputInclusionCalculationSC[
					trainingAndTestSet,
					kernelFunction,
					isRegression,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
    				SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionInclusionsPerStep -> numberOfInclusionsPerStepList,
					UtilityOptionInclusionStartList -> inclusionStartList
				]
			]
		]
	];

GetSvmInputInclusionCalculationSC[

	(* Analyzes relevance of input components by successive get-one-in for regression and classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   svmInputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, svmInfo}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* For details see method Kernel[] *)
	kernelFunction_,	
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			alphaValueLimit,
			initialAlphasList,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			objectiveFunctionEpsilon,
			optimizationMethod,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,
			targetInterval,
			normalizationType,
			currentIncludedInputComponentList,
			i,
			k,
			numberOfInputs,
			numberOfIncludedInputs,
			svmInputComponentRelevanceList,
	        svmInfo,
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
		(* Svm options *)   
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
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
		svmInputComponentRelevanceList = {};
    
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
					svmInfo = 
						FitSvm[
							trainingSet,
							kernelFunction,
							SvmOptionOptimizationMethod -> optimizationMethod,
							SvmOptionInitialAlphasList -> initialAlphasList,
						    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
			    			SvmOptionMaximizationPrecision -> maximizationPrecision,
							SvmOptionMaximumIterations -> maximumNumberOfIterations,
							SvmOptionIsPostProcess -> isPostProcess,
							SvmOptionScalingFactor -> scalingFactor,
							SvmOptionAlphaValueLimit -> alphaValueLimit,
							SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
							SvmOptionTauQpSolve -> tauQpSolve,
							SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
							DataTransformationOptionTargetInterval -> targetInterval,
							DataTransformationOptionNormalizationType -> normalizationType
						];
					If[Length[testSet] > 0,
            
						testSetRmse = CalculateSvmDataSetRmse[testSet, svmInfo];
						AppendTo[rmseList,{testSetRmse, i}],
          
						trainingSetRmse = CalculateSvmDataSetRmse[trainingSet, svmInfo];
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
			svmInfo = 
				FitSvm[
					trainingSet,
					kernelFunction,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
	    			SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateSvmDataSetRmse[trainingSet, svmInfo];
					currentTestSetRmse = CalculateSvmDataSetRmse[testSet, svmInfo];
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
							svmInfo
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateSvmDataSetRmse[trainingSet, svmInfo];
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
							svmInfo
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[trainingSet, svmInfo];
					currentTestSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[testSet, svmInfo];
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
							svmInfo
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[trainingSet, svmInfo];
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
							svmInfo
						}
				]
			];	

			AppendTo[svmInputComponentRelevanceList, relevance];
			includedInputComponentList = currentIncludedInputComponentList,
			
			{k, Length[numberOfInclusionsPerStepList]}
		];
		
		Return[svmInputComponentRelevanceList]
	];

GetSvmInputInclusionCalculationPC[

	(* Analyzes relevance of input components by successive get-one-in for regression and classification.
	   If option UtilityOptionInclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components are included after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   svmInputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, includedInputComponentList, svmInfo}
	   trainingSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfIncludedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* For details see method Kernel[] *)
	kernelFunction_,	
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			alphaValueLimit,
			initialAlphasList,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			objectiveFunctionEpsilon,
			optimizationMethod,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,
			targetInterval,
			normalizationType,
			currentIncludedInputComponentList,
			i,
			k,
			numberOfInputs,
			numberOfIncludedInputs,
			svmInputComponentRelevanceList,
	        svmInfo,
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
		(* Svm options *)   
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
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
		svmInputComponentRelevanceList = {};
    
    	ParallelNeeds[{"CIP`SVM`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[optimizationMethod, initialAlphasList, objectiveFunctionEpsilon, maximizationPrecision,
			maximumNumberOfIterations, isPostProcess, scalingFactor, alphaValueLimit, epsilonQpSolve, tauQpSolve,
			penaltyConstantQpSolve, targetInterval, normalizationType];
			
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
						
						svmInfo = 
							FitSvm[
								trainingSet,
								kernelFunction,
								SvmOptionOptimizationMethod -> optimizationMethod,
								SvmOptionInitialAlphasList -> initialAlphasList,
							    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
			    				SvmOptionMaximizationPrecision -> maximizationPrecision,
								SvmOptionMaximumIterations -> maximumNumberOfIterations,
								SvmOptionIsPostProcess -> isPostProcess,
								SvmOptionScalingFactor -> scalingFactor,
								SvmOptionAlphaValueLimit -> alphaValueLimit,
								SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
								SvmOptionTauQpSolve -> tauQpSolve,
								SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
								DataTransformationOptionTargetInterval -> targetInterval,
								DataTransformationOptionNormalizationType -> normalizationType
							];
						
						If[Length[testSet] > 0,
            
							testSetRmse = CalculateSvmDataSetRmse[testSet, svmInfo];
							{testSetRmse, i},
          
							trainingSetRmse = CalculateSvmDataSetRmse[trainingSet, svmInfo];
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
			svmInfo = 
				FitSvm[
					trainingSet,
					kernelFunction,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
	    			SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionsParallelization -> "ParallelCalculation"
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateSvmDataSetRmse[trainingSet, svmInfo];
					currentTestSetRmse = CalculateSvmDataSetRmse[testSet, svmInfo];
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
							svmInfo
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateSvmDataSetRmse[trainingSet, svmInfo];
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
							svmInfo
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[trainingSet, svmInfo];
					currentTestSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[testSet, svmInfo];
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
							svmInfo
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[trainingSet, svmInfo];
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
							svmInfo
						}
				]
			];	

			AppendTo[svmInputComponentRelevanceList, relevance];
			includedInputComponentList = currentIncludedInputComponentList,
			
			{k, Length[numberOfInclusionsPerStepList]}
		];
		
		Return[svmInputComponentRelevanceList]
	];

GetSvmInputRelevanceClass[

	(* Analyzes relevance of input components by successive leave-one-out for classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   svmInputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, svmInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, (best) classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,
	
	(* For details see method Kernel[] *)
	kernelFunction_,	
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			alphaValueLimit,
			initialAlphasList,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			objectiveFunctionEpsilon,
			optimizationMethod,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,
			targetInterval,
			normalizationType,
			isIntermediateOutput,
			numberOfExclusionsPerStepList,
			isRegression,
			parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Svm options *)   
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    (* DataTransformation options *)   
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
	    (* Utility options *)   
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfExclusionsPerStepList = UtilityOptionExclusionsPerStep/.{opts}/.Options[UtilityOptionsExclusion];
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		isRegression = False;
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				GetSvmInputRelevanceCalculationPC[
					trainingAndTestSet,
					kernelFunction,
					isRegression,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
	    			SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetSvmInputRelevanceCalculationSC[
					trainingAndTestSet,
					kernelFunction,
					isRegression,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
	    			SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			]
		]
	];

GetSvmClassRelevantComponents[

	(* Returns most-to-least-relevance sorted components from svmInputComponentRelevanceListForClassification.

	   Returns: Returns most-to-least-relevance sorted components *)


	(* svmInputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, svmInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) correct classification in percent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best correct classification in percent of test set} *)
	svmInputComponentRelevanceListForClassification_,
	
	(* Number of components to return *)
	numberOfComponents_?IntegerQ
    
	] :=
  
	Module[
    
		{},

		Return[
			CIP`Graphics`GetClassRelevantComponents[svmInputComponentRelevanceListForClassification, numberOfComponents]
		]
	];

GetSvmInputRelevanceRegress[

	(* Analyzes relevance of input components by successive leave-one-out for regression.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   svmInputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, svmInfo}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* For details see method Kernel[] *)
	kernelFunction_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			alphaValueLimit,
			initialAlphasList,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			objectiveFunctionEpsilon,
			optimizationMethod,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,
			targetInterval,
			normalizationType,
			isIntermediateOutput,
			numberOfExclusionsPerStepList,
			isRegression,
			parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Svm options *)   
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    (* DataTransformation options *)   
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
	    (* Utility options *)   
	    isIntermediateOutput = UtilityOptionIsIntermediateOutput/.{opts}/.Options[UtilityOptionsIntermediateOutput];
	    numberOfExclusionsPerStepList = UtilityOptionExclusionsPerStep/.{opts}/.Options[UtilityOptionsExclusion];
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		isRegression = True;
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				GetSvmInputRelevanceCalculationPC[
					traintrainingAndTestSet,
					kernelFunction,
					isRegression,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
	    			SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetSvmInputRelevanceCalculationSC[
					trainingAndTestSet,
					kernelFunction,
					isRegression,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
	    			SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionIsIntermediateOutput -> isIntermediateOutput,
					UtilityOptionExclusionsPerStep -> numberOfExclusionsPerStepList
				]
			]
		]
	];

GetSvmRegressRelevantComponents[

	(* Returns most-to-least-relevance sorted components from svmInputComponentRelevanceListForRegression.

	   Returns: Returns most-to-least-relevance sorted components *)


	(* svmInputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, svmInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)
	svmInputComponentRelevanceListForRegression_,
	
	(* Number of components to return *)
	numberOfComponents_?IntegerQ
    
	] :=
  
	Module[
    
		{},

		Return[
			CIP`Graphics`GetRegressRelevantComponents[svmInputComponentRelevanceListForRegression, numberOfComponents]
		]
	];

GetSvmInputRelevanceCalculationSC[

	(* Analyzes relevance of input components by successive leave-one-out for regression and classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   svmInputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, svmInfo}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* For details see method Kernel[] *)
	kernelFunction_,
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			alphaValueLimit,
			initialAlphasList,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			objectiveFunctionEpsilon,
			optimizationMethod,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,
			targetInterval,
			normalizationType,
			currentRemovedInputComponentList,
			i,
			k,
			numberOfInputs,
			numberOfRemovedInputs,
			svmInputComponentRelevanceList,
	        svmInfo,
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
		(* Svm options *)   
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
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
		svmInputComponentRelevanceList = {};
    
		(* Result for no removal *)
		trainingSet = trainingAndTestSet[[1]];
		testSet = trainingAndTestSet[[2]];
		svmInfo = 
			FitSvm[
				trainingSet,
				kernelFunction,
				SvmOptionOptimizationMethod -> optimizationMethod,
				SvmOptionInitialAlphasList -> initialAlphasList,
			    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
    			SvmOptionMaximizationPrecision -> maximizationPrecision,
				SvmOptionMaximumIterations -> maximumNumberOfIterations,
				SvmOptionIsPostProcess -> isPostProcess,
				SvmOptionScalingFactor -> scalingFactor,
				SvmOptionAlphaValueLimit -> alphaValueLimit,
				SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
				SvmOptionTauQpSolve -> tauQpSolve,
				SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
				DataTransformationOptionTargetInterval -> targetInterval,
				DataTransformationOptionNormalizationType -> normalizationType
			];
		
		initialTrainingSetRmse = CalculateSvmDataSetRmse[trainingSet, svmInfo];
		If[isRegression,
			
			(* Regression*)
			If[Length[testSet] > 0,
				
				(* Regression WITH test set*)
				initialTestSetRmse = CalculateSvmDataSetRmse[testSet, svmInfo];
				If[isIntermediateOutput,
					Print["initialTrainingSetRmse = ", initialTrainingSetRmse];
					Print["initialTestSetRmse     = ", initialTestSetRmse]
				];
				relevance = 
					{
						{0.0, initialTrainingSetRmse},
						{0.0, initialTestSetRmse},
						{}, 
						svmInfo
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
						svmInfo
					}
			],
			
			(* Classification *)
			initialTrainingSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[trainingSet, svmInfo];
			If[Length[testSet] > 0,
				
				(* Classification WITH test set*)
				initialTestSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[testSet, svmInfo];
				If[isIntermediateOutput,
					Print["initialTrainingSetCorrectClassificationInPercent = ", initialTrainingSetCorrectClassificationInPercent];
					Print["initialTestSetCorrectClassificationInPercent     = ", initialTestSetCorrectClassificationInPercent]
				];
				relevance = 
					{
						{0.0, initialTrainingSetCorrectClassificationInPercent},
						{0.0, initialTestSetCorrectClassificationInPercent},
						{}, 
						svmInfo
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
						svmInfo
					}
			]
		];	
		
		AppendTo[svmInputComponentRelevanceList, relevance];
    
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
					svmInfo = 
						FitSvm[
							trainingSet,
							kernelFunction,
							SvmOptionOptimizationMethod -> optimizationMethod,
							SvmOptionInitialAlphasList -> initialAlphasList,
						    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
			    			SvmOptionMaximizationPrecision -> maximizationPrecision,
							SvmOptionMaximumIterations -> maximumNumberOfIterations,
							SvmOptionIsPostProcess -> isPostProcess,
							SvmOptionScalingFactor -> scalingFactor,
							SvmOptionAlphaValueLimit -> alphaValueLimit,
							SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
							SvmOptionTauQpSolve -> tauQpSolve,
							SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
							DataTransformationOptionTargetInterval -> targetInterval,
							DataTransformationOptionNormalizationType -> normalizationType
						];
					If[Length[testSet] > 0,
            
						testSetRmse = CalculateSvmDataSetRmse[testSet, svmInfo];
						AppendTo[rmseList,{testSetRmse, i}],
          
						trainingSetRmse = CalculateSvmDataSetRmse[trainingSet, svmInfo];
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
			svmInfo = 
				FitSvm[
					trainingSet,
					kernelFunction,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
	    			SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateSvmDataSetRmse[trainingSet, svmInfo];
					currentTestSetRmse = CalculateSvmDataSetRmse[testSet, svmInfo];
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
							svmInfo
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateSvmDataSetRmse[trainingSet, svmInfo];
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
							svmInfo
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[trainingSet, svmInfo];
					currentTestSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[testSet, svmInfo];
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
							svmInfo
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[trainingSet, svmInfo];
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
							svmInfo
						}
				]
			];	

			AppendTo[svmInputComponentRelevanceList, relevance];
			removedInputComponentList = currentRemovedInputComponentList,
			
			{k, Length[numberOfExclusionsPerStepList]}
		];
		
		Return[svmInputComponentRelevanceList]
	];

GetSvmInputRelevanceCalculationPC[

	(* Analyzes relevance of input components by successive leave-one-out for regression and classification.
	   If option UtilityOptionExclusionsPerStep is specified then after each loop over all input components the number of input components 
	   specified are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components are excluded after the first loop, 
	   after the second loop 10 input components etc. 

	   Returns: 
	   svmInputComponentRelevanceList: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, svmInfo}
	   trainingSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE/classificationInPercent of test set} *)


	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   testSet has the same structure as trainingSet *)
	trainingAndTestSet_,

	(* For details see method Kernel[] *)
	kernelFunction_,
	
	(* True: Regression, False: Classification*)
	isRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			alphaValueLimit,
			initialAlphasList,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			objectiveFunctionEpsilon,
			optimizationMethod,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,
			targetInterval,
			normalizationType,
			currentRemovedInputComponentList,
			i,
			k,
			numberOfInputs,
			numberOfRemovedInputs,
			svmInputComponentRelevanceList,
	        svmInfo,
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
		(* Svm options *)   
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
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
		svmInputComponentRelevanceList = {};
    
		(* Result for no removal *)
		trainingSet = trainingAndTestSet[[1]];
		testSet = trainingAndTestSet[[2]];
		svmInfo = 
			FitSvm[
				trainingSet,
				kernelFunction,
				SvmOptionOptimizationMethod -> optimizationMethod,
				SvmOptionInitialAlphasList -> initialAlphasList,
			    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
    			SvmOptionMaximizationPrecision -> maximizationPrecision,
				SvmOptionMaximumIterations -> maximumNumberOfIterations,
				SvmOptionIsPostProcess -> isPostProcess,
				SvmOptionScalingFactor -> scalingFactor,
				SvmOptionAlphaValueLimit -> alphaValueLimit,
				SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
				SvmOptionTauQpSolve -> tauQpSolve,
				SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
				DataTransformationOptionTargetInterval -> targetInterval,
				DataTransformationOptionNormalizationType -> normalizationType
			];
		
		initialTrainingSetRmse = CalculateSvmDataSetRmse[trainingSet, svmInfo];
		If[isRegression,
			
			(* Regression*)
			If[Length[testSet] > 0,
				
				(* Regression WITH test set*)
				initialTestSetRmse = CalculateSvmDataSetRmse[testSet, svmInfo];
				If[isIntermediateOutput,
					Print["initialTrainingSetRmse = ", initialTrainingSetRmse];
					Print["initialTestSetRmse     = ", initialTestSetRmse]
				];
				relevance = 
					{
						{0.0, initialTrainingSetRmse},
						{0.0, initialTestSetRmse},
						{}, 
						svmInfo
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
						svmInfo
					}
			],
			
			(* Classification *)
			initialTrainingSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[trainingSet, svmInfo];
			If[Length[testSet] > 0,
				
				(* Classification WITH test set*)
				initialTestSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[testSet, svmInfo];
				If[isIntermediateOutput,
					Print["initialTrainingSetCorrectClassificationInPercent = ", initialTrainingSetCorrectClassificationInPercent];
					Print["initialTestSetCorrectClassificationInPercent     = ", initialTestSetCorrectClassificationInPercent]
				];
				relevance = 
					{
						{0.0, initialTrainingSetCorrectClassificationInPercent},
						{0.0, initialTestSetCorrectClassificationInPercent},
						{}, 
						svmInfo
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
						svmInfo
					}
			]
		];	
		
		AppendTo[svmInputComponentRelevanceList, relevance];
    
    	ParallelNeeds[{"CIP`SVM`", "CIP`DataTransformation`", "CIP`Utility`"}];
		DistributeDefinitions[optimizationMethod, initialAlphasList, objectiveFunctionEpsilon, maximizationPrecision,
			maximumNumberOfIterations, isPostProcess, scalingFactor, alphaValueLimit, epsilonQpSolve, tauQpSolve,
			penaltyConstantQpSolve, targetInterval, normalizationType];
			
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
						
						svmInfo = 
							FitSvm[
								trainingSet,
								kernelFunction,
								SvmOptionOptimizationMethod -> optimizationMethod,
								SvmOptionInitialAlphasList -> initialAlphasList,
							    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
				    			SvmOptionMaximizationPrecision -> maximizationPrecision,
								SvmOptionMaximumIterations -> maximumNumberOfIterations,
								SvmOptionIsPostProcess -> isPostProcess,
								SvmOptionScalingFactor -> scalingFactor,
								SvmOptionAlphaValueLimit -> alphaValueLimit,
								SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
								SvmOptionTauQpSolve -> tauQpSolve,
								SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
								DataTransformationOptionTargetInterval -> targetInterval,
								DataTransformationOptionNormalizationType -> normalizationType
							];
								
						If[Length[testSet] > 0,
	            
							testSetRmse = CalculateSvmDataSetRmse[testSet, svmInfo];
							{testSetRmse, i},
	          
							trainingSetRmse = CalculateSvmDataSetRmse[trainingSet, svmInfo];
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
			svmInfo = 
				FitSvm[
					trainingSet,
					kernelFunction,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
	    			SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionsParallelization -> "ParallelCalculation"
				];

			If[isRegression,
				
				(* Regression*)
				If[Length[testSet] > 0,
					
					(* Regression WITH test set *)
	    			currentTrainingSetRmse = CalculateSvmDataSetRmse[trainingSet, svmInfo];
					currentTestSetRmse = CalculateSvmDataSetRmse[testSet, svmInfo];
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
							svmInfo
						},
		          
					(* Regression WITHOUT test set *)
					currentTrainingSetRmse = CalculateSvmDataSetRmse[trainingSet, svmInfo];
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
							svmInfo
						}
				],
				
				(* Classification *)
				If[Length[testSet] > 0,
					
					(* Classification WITH test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[trainingSet, svmInfo];
					currentTestSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[testSet, svmInfo];
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
							svmInfo
						},
		          
					(* Classification WITHOUT test set *)
					currentTrainingSetCorrectClassificationInPercent = CalculateSvmCorrectClassificationInPercent[trainingSet, svmInfo];
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
							svmInfo
						}
				]
			];	

			AppendTo[svmInputComponentRelevanceList, relevance];
			removedInputComponentList = currentRemovedInputComponentList,
			
			{k, Length[numberOfExclusionsPerStepList]}
		];
		
		Return[svmInputComponentRelevanceList]
	];
	
GetSvmRegressionResult[
	
	(* Returns SVM regression result according to named property list.

	   Returns :
	   SVM regression result according to named property *)

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
    svmInfo_,
	
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
	    
		pureFunction = Function[inputs, CalculateSvmOutputs[inputs, svmInfo]];
	    Return[
	    	CIP`Graphics`GetSingleRegressionResult[
		    	namedProperty, 
		    	dataSet, 
		    	pureFunction,
		    	GraphicsOptionNumberOfIntervals -> numberOfIntervals
			]
		]
	];

GetSvmSeriesClassificationResult[

	(* Returns result of SVM series classifications for training and test set.

	   Returns: 
	   svmSeriesClassificationResult: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in svmInfoList, classification result in percent for training set}
	   testPoint[[i]]: {index i in svmInfoList, classification result in percent for test set} *)


    (* {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...}
	   testSet has the same structure and restrictions as trainingSet 
	   NOTE: Training and test set MUST be in original units *)
    trainingAndTestSet_,
    
	(* {svmInfo1, svmInfo2, ...}
	   svmInfo: See "Frequently used data structures" *)
    svmInfoList_
    
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
			pureFunction = Function[inputs, CalculateSvmClassNumbers[inputs, svmInfoList[[i]]]];
			correctClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[trainingSet, pureFunction];
			AppendTo[trainingPoints2D, {N[i], correctClassificationInPercent}];
			If[Length[testSet] > 0,
				correctClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[testSet, pureFunction];
				AppendTo[testPoints2D, {N[i], correctClassificationInPercent}]
			],
			
			{i, Length[svmInfoList]}
		];
		
		Return[{trainingPoints2D, testPoints2D}]
	];

GetSvmSeriesRmse[

	(* Returns RMSE of SVM series for training and test set.

	   Returns: 
	   svmSeriesRmse: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in svmInfoList, RMSE for training set}
	   testPoint[[i]]: {index i in svmInfoList, RMSE for test set} *)


    (* {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...}
	   testSet has the same structure and restrictions as trainingSet 
	   NOTE: Training and test set MUST be in original units *)
    trainingAndTestSet_,
    
	(* {svmInfo1, svmInfo2, ...}
	   svmInfo: See "Frequently used data structures" *)
    svmInfoList_
    
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
			pureFunction = Function[inputs, CalculateSvmOutputs[inputs, svmInfoList[[i]]]];
			rmse = Sqrt[CIP`Utility`GetMeanSquaredError[trainingSet, pureFunction]];
			AppendTo[trainingPoints2D, {N[i], rmse}];
			If[Length[testSet] > 0,
				rmse = Sqrt[CIP`Utility`GetMeanSquaredError[testSet, pureFunction]];
				AppendTo[testPoints2D, {N[i], rmse}]
			],
			
			{i, Length[svmInfoList]}
		];
		
		Return[{trainingPoints2D, testPoints2D}]
	];

GetSvmTrainOptimization[

	(* Returns training set optimization result for SVM training.

	   Returns:
	   svmTrainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, svmInfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   svmInfoList: List with svmInfo
	   svmInfoList[[i]] refers to optimization step i *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* For details see method Kernel[] *)
	kernelFunction_,

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
			targetInterval,
			normalizationType,
			alphaValueLimit,
			initialAlphasList,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			objectiveFunctionEpsilon,
			optimizationMethod,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,
			randomValueInitialization,

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
			svmInfo,
			trainingSetRMSE,
			testSetRMSE,
			pureOutputFunction,
			trainingSetRmseList,
			testSetRmseList,
			trainingAndTestSetList,
			svmInfoList,
			selectionResult,
			blackList,
			parallelization
		},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* SVM options *)
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];

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
				DataTransformationOptionTargetInterval -> targetInterval
			];
		trainingSetIndexList = clusterRepresentativesRelatedIndexLists[[1]];
		testSetIndexList = clusterRepresentativesRelatedIndexLists[[2]];
		indexLists = clusterRepresentativesRelatedIndexLists[[3]];

		trainingSetRmseList = {};
		testSetRmseList = {};
		trainingAndTestSetList = {};
		svmInfoList = {};
		blackList = {};
		Do[
			(* Fit training set and evaluate RMSE *)
			trainingSet = CIP`DataTransformation`GetDataSetPart[dataSet, trainingSetIndexList];
			testSet = CIP`DataTransformation`GetDataSetPart[dataSet, testSetIndexList];
			
			svmInfo = 
				FitSvm[
					trainingSet,
					kernelFunction,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
	    			SvmOptionMaximizationPrecision -> maximizationPrecision,
					SvmOptionMaximumIterations -> maximumNumberOfIterations,
					SvmOptionIsPostProcess -> isPostProcess,
					SvmOptionScalingFactor -> scalingFactor,
					SvmOptionAlphaValueLimit -> alphaValueLimit,
					DataTransformationOptionTargetInterval -> targetInterval,
					DataTransformationOptionNormalizationType -> normalizationType,
					SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
					SvmOptionTauQpSolve -> tauQpSolve,
					SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					UtilityOptionsParallelization -> parallelization
				];
			
			trainingSetRMSE = CalculateSvmDataSetRmse[trainingSet, svmInfo];
			testSetRMSE = CalculateSvmDataSetRmse[testSet, svmInfo];

			(* Set iteration results *)
			AppendTo[trainingSetRmseList, {N[i], trainingSetRMSE}];
			AppendTo[testSetRmseList, {N[i], testSetRMSE}];
			AppendTo[trainingAndTestSetList, {trainingSet, testSet}];
			AppendTo[svmInfoList, svmInfo];
			
			(* Break if necessary *)
			If[i == numberOfTrainingSetOptimizationSteps,
				Break[]
			];

			(* Select new training and test set index lists *)
			pureOutputFunction = Function[input, CalculateSvmOutput[input, svmInfo]];
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
				svmInfoList
			}
		]
	];

Kernel[

	(* Kernel.

	   Returns: 
	   Kernel value/expression *)

    
    u_?VectorQ,
    
    v_?VectorQ,

	(* Kernel specific parameters:
	   kernelFunction[[1]]: Name of kernel function
	   
	   Wavelet kernel:
	   kernelFunction[[2]]: a 

	   GaussianRBF kernel:
	   kernelFunction[[2]]: beta 

	   Polynomial kernel:
	   kernelFunction[[2]]: c
	   kernelFunction[[3]]: d

	   InfinityPolynomialUniversal kernel:
	   kernelFunction[[2]]: d
	   
	   UniversalFourier kernel:
	   kernelFunction[[2]]: q *)
	kernelFunction_
    
	] :=
  
	Module[
    
		{},

		Switch[kernelFunction[[1]],
			
			"Wavelet",
			Return[KernelWavelet[u, v, kernelFunction[[2]]]],

			"GaussianRBF",
			Return[KernelGaussianRbf[u, v, kernelFunction[[2]]]],

			"Polynomial",
			Return[KernelPolynomial[u, v, kernelFunction[[2]], kernelFunction[[3]]]],

			"InfinityPolynomialUniversal",
			Return[KernelInfinityPolynomialUniversal[u, v, kernelFunction[[2]]]],
			
			"UniversalFourier",
			Return[KernelUniversalFourier[u, v, kernelFunction[[2]]]]
		]
	];

KernelGaussianRbf[

	(* Gaussian RBF (Radial basis function) universal kernel.

	   Returns: 
	   Gaussian RBF (Radial basis function) universal kernel value/expression *)

    
    u_?VectorQ,
    
    v_?VectorQ,

	beta_?NumberQ
    
	] :=
  
	Module[
    
		{},

		Return[
			Exp[-beta*(Norm[u - v])^2]
		]
	];

KernelInfinityPolynomialUniversal[

	(* Infinity polynomial universal kernel.

	   Returns: 
	   Infinity polynomial universal kernel value/expression *)

    
    u_?VectorQ,
    
    v_?VectorQ,

	d_?IntegerQ
    
	] :=
  
	Module[
    
		{},

		Return[
			(1 - u . v)^(-d)
		]
	];

KernelPolynomial[

	(* Polynomial kernel.

	   Returns: 
	   Polynomial kernel value/expression *)

    
    u_?VectorQ,
    
    v_?VectorQ,

	c_?NumberQ,

	d_?IntegerQ
    
	] :=
  
	Module[
    
		{},

		Return[
			(c + u . v)^d
		]
	];

KernelUniversalFourier[

	(* Universal fourier kernel.

	   Returns: 
	   Universal fourier kernel value/expression *)

    
    u_?VectorQ,
    
    v_?VectorQ,

	q_?NumberQ
    
	] :=
  
	Module[
    
		{
			qSquare,
			i
		},

		qSquare = q*q;
		Return[
			Product[
				(1 - qSquare)/(2*(1 - 2*q*Cos[u[[i]] - v[[i]]] + qSquare)), 
				
				{i, Length[u]}
			]
		]
	];

KernelWavelet[

	(* Wavelet kernel.

	   Returns: 
	   Wavelet kernel value/expression *)

    
    u_?VectorQ,
    
    v_?VectorQ,

	a_?NumberQ
    
	] :=
  
	Module[
    
		{
			b,
			i
		},

		Return[
			Product[
				b = (u[[i]] - v[[i]])/a;
				Cos[1.75*b]*Exp[-0.5*b*b], 
				
				{i, Length[u]}
			]
		]
	];

QpSolve[
	
	kernelMatrix_,
	
	p_,
	
	a_,
	
	b_,
	
	c_,
	
	y_,
	
	tauQpSolve_
	
	] := 

	Module[
		
		{
			alphas,
			l,
			indexSets,
			upperBoundarySet,
			lowerBoundarySet,
			F,
			violatingPair,
			oldViolatingPair,
			M,
			k
		},
		
		l = Length[kernelMatrix];
		alphas = QpFeasiblePoint[a, b, y, c];
		If[alphas == Null,
			Return[Null]
		];
		k = 0;
		While[True,
			indexSets = QpGetIndexSets[alphas, a, b, y];
			upperBoundarySet = QpUpperBoundarySet[indexSets];
			lowerBoundarySet = QpLowerBoundarySet[indexSets];
			F = (kernelMatrix . alphas + p)/y;
		
		If[k == -1,
			Print["indexSets = ", indexSets];
			Print["upperBoundarySet = ", upperBoundarySet];
			Print["lowerBoundarySet = ", lowerBoundarySet];
			Print["F = ", F];
		];
			
			If[QpOptimalQ[upperBoundarySet, lowerBoundarySet, F, tauQpSolve],
				Break[]
			];
			violatingPair = QpGetViolatingPair[indexSets[[1]], upperBoundarySet, lowerBoundarySet, F, tauQpSolve];

		If[k == -1,
			Print["violatingPair = ", violatingPair];
		];

			If[violatingPair == Null,
				Return["Error: no violating pair"]
			];
			If[violatingPair == oldViolatingPair,
				Print["Error: stuck on violating pair ", violatingPair];
				Return[alphas]
			];
			M = Complement[Range[1, l], violatingPair];
			alphas[[violatingPair]] = 
				QpSolve2D[
					kernelMatrix[[violatingPair, violatingPair]], 
					p[[violatingPair]] + kernelMatrix[[violatingPair, M]] . alphas[[M]], 
					a[[violatingPair]], 
					b[[violatingPair]], 
					y[[violatingPair]], 
					alphas[[violatingPair]]
				];
			oldViolatingPair = violatingPair;
			k++;
		];

		Return[alphas]
	];

QpSolve2D[
	
	kernelMatrix_ /; Length[kernelMatrix]  ==  2, 
	
	p_, 
	
	a_, 
	
	b_, 
	
	y_, 
	
	alphas_
	
	]  :=  

	Module[
		
		{
			t, 
			newAlphas, 
			c, 
			tn, 
			td
		},

		c = alphas.y;
		tn = -((kernelMatrix[[1]].alphas + p[[1]])/y[[1]] - (kernelMatrix[[2]].alphas + p[[2]])/y[[2]]);
		td = kernelMatrix[[1, 1]]/y[[1]]^2 + kernelMatrix[[2, 2]]/y[[2]]^2 - (kernelMatrix[[1, 2]] + kernelMatrix[[2, 1]])/(y[[1]] y[[2]]);
		t = 
			If[td != 0.0, 
				tn/td, 
				Sign[tn]*Infinity
			];
		newAlphas = alphas + {t/y[[1]], -t/y[[2]]};
		If[newAlphas[[1]] < a[[1]], 
			newAlphas = {a[[1]], (c - a[[1]] y[[1]])/y[[2]]}
		];
		If[newAlphas[[1]] > b[[1]], 
			newAlphas = {b[[1]], (c - b[[1]] y[[1]])/y[[2]]}
		];
		If[newAlphas[[2]] < a[[2]], 
			newAlphas = {(c - a[[2]] y[[2]])/y[[1]], a[[2]]}
		];
		If[newAlphas[[2]] > b[[2]], 
			newAlphas = {(c - b[[2]] y[[2]])/y[[1]], b[[2]]}
		];
		
		Return[newAlphas]
	];

QpFeasiblePoint[
	
	a_,
	
	b_,
	
	y_,
	
	c_
	
	] := 

	Module[
		
		{
			i,
			alphas,
			l
		},
		
		l = Length[y];
		alphas = Table[0.0, {l}];
		Do[
			alphas[[i]]=(c - Drop[y, {i}] . Drop[alphas, {i}])/y[[i]];
			alphas[[i]]=Min[Max[alphas[[i]], a[[i]]], b[[i]]];
			If[y.alphas == c, Break],
			
			{i, l}
		];
		If[y.alphas == c,
			
			Return[alphas],
			
			Return[Null]
		]
	];

QpGetIndexSets[
	
	alphas_,
	
	a_,
	
	b_,
	
	y_

	] := 

	Map[Flatten,
		{
			Position[(1 - UnitStep[a - alphas])*(1 - UnitStep[-(b - alphas)]),1],
			Position[QpKroneckerDelta[alphas, a]*UnitStep[y], 1],
			Position[QpKroneckerDelta[alphas, b]*UnitStep[-y], 1],
			Position[QpKroneckerDelta[alphas, b]*UnitStep[y], 1],
			Position[QpKroneckerDelta[alphas, a]*UnitStep[-y], 1]
		}
	];

QpGetViolatingPair[
	
	I0_,
	
	upperBoundarySet_,
	
	lowerBoundarySet_,
	
	F_,
	
	tauQpSolve_
	
	] :=
	
	Module[
		
		{
			o,
			i,
			j,
			l
		},

		If[Max[F[[I0]]] - Min[F[[I0]]] > tauQpSolve,

      		o = Ordering[F[[I0]]];
      		{I0[[First[o]]], I0[[Last[o]]]},

			(* else no violating pair on I0, scan entire {1...l} *)
			i = 1;
			l = Length[F];
			While[i <= l,
				j = 1;
				While[j <= l,
					If[QpViolatingPairQ[upperBoundarySet, lowerBoundarySet, i, j, F, tauQpSolve],
						Return[{i,j}]
					];
					j++
				];
				i++
			];
			
			Return[Null]
		]
	];

QpKroneckerDelta[
	
	a_?VectorQ,
	
	b_?VectorQ
	
	] := Map[KroneckerDelta[#[[1]],#[[2]]]&, Thread[{a,b}]];

QpLowerBoundarySet[
	
	indexSets_
	
	] := Union[indexSets[[1]], indexSets[[4]], indexSets[[5]]];

QpOptimalQ[
	
	upperBoundarySet_,
	
	lowerBoundarySet_,
	
	F_,
	
	tauQpSolve_
	
	] := Max[F[[lowerBoundarySet]]] - Min[F[[upperBoundarySet]]] <= tauQpSolve;

QpRegressionBias[
	
	alphas_,
	
	(* {input1, input2, ...}
	   input: {inputValue1, inputValue2, ...}
	   NOTE: inputs is a matrix!
	   NOTE: inputs[[i]] corresponds to outputs[[i]] *)
	inputs_?MatrixQ,
	
	(* {outputValue1, outputValue2, ...}
	   NOTE: outputs is a vector!
	   NOTE: outputs[[i]] corresponds to inputs[[i]] *)
	outputs_?VectorQ,
	
	epsilonQpSolve_,
	
	(* For details see method Kernel[] *)
	kernelFunction_
	
	] := 
	
	Module[
		
		{
			i,
			numberOfInputs,
			nonZeroSupportVectorIndexList
		},
		
		numberOfInputs = Length[inputs];
		nonZeroSupportVectorIndexList = QpSupportVectors[alphas][[1]];

		Return[
			epsilonQpSolve + 
			outputs[[nonZeroSupportVectorIndexList]] - 
			Sum[
				(alphas[[i + numberOfInputs]] - alphas[[i]])*Kernel[inputs[[i]], inputs[[nonZeroSupportVectorIndexList]], kernelFunction],
				
				{i, numberOfInputs}
			]
		]
	];

QpSupportVectors[
	
	alphas_
	
	] := Flatten[Position[alphas, a_/;a != 0.0]];

QpUpperBoundarySet[
	
	indexSets_
	
	] := Union[indexSets[[1]], indexSets[[2]], indexSets[[3]]];

QpViolatingPairQ[
	
	upperBoundarySet_,
	
	lowerBoundarySet_,
	
	i_,
	
	j_,
	
	F_,
	
	tauQpSolve_
	
	] := (MemberQ[upperBoundarySet, i] && MemberQ[lowerBoundarySet, j] && (F[[j]] - F[[i]] > tauQpSolve)) || (MemberQ[lowerBoundarySet, i] && MemberQ[upperBoundarySet, j] && (F[[i]] - F[[j]] > tauQpSolve));

ReduceSingleSvmResult[

	(* Removes zero alphas and corresponding dataSetInputs from singleSvmResult
	
	   Returns: 
	   singleSvmResult: {alphas, dataSetInputs, kernelFunction, b}
	   alphas: {alpha1, alpha2, ..., alpha<Length[inputs]>}
	   dataSetInputs: {input1, input2, ..., input<Length[dataSet]>}
	   input: {inputValue1, ..., inputValue<Length[input]>} *)


	(* singleSvmResult: {alphas, dataSetInputs, kernelFunction, b}
	   alphas: {alpha1, alpha2, ..., alpha<Length[inputs]>}
	   dataSetInputs: {input1, input2, ..., input<Length[dataSet]>}
	   input: {inputValue1, ..., inputValue<Length[input]>} *)
	singleSvmResult_
	
	] := 
	
	Module[
		
		{
			alphas,
			dataSetInputs,
			kernelFunction,
			i,
			b,
			reducedAlphas,
			reducedDataSetInputs
		},

		alphas = singleSvmResult[[1]];
		dataSetInputs = singleSvmResult[[2]];
		kernelFunction = singleSvmResult[[3]];
		b = singleSvmResult[[4]];
		
		reducedAlphas = {};
		reducedDataSetInputs = {};
		Do[
			If[alphas[[i]] != 0.0,
				AppendTo[reducedAlphas, alphas[[i]]];
				AppendTo[reducedDataSetInputs, dataSetInputs[[i]]]
			],
			
			{i, Length[alphas]}
		];
		
		Return[
			{reducedAlphas, reducedDataSetInputs, kernelFunction, b}
		]
	];

ScanClassTrainingWithSvm[

	(* Scans training and test set for different training fractions based on method FitSvm, see code.
	
	   Returns:
	   svmClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, svmInfo1}, {trainingAndTestSet2, svmInfo2}, ...}
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

	(* For details see method Kernel[] *)
	kernelFunction_,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			optimizationMethod,
			initialAlphasList,
			objectiveFunctionEpsilon,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			alphaValueLimit,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
	    	targetInterval,
	    	normalizationType,

			numberOfTrainingSetOptimizationSteps,
			deviationCalculationMethod,
			blackListLength,
			
			parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* SVM options *)
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
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
				ScanClassTrainingWithSvmPC[
					classificationDataSet,
					kernelFunction,
					trainingFractionList,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
				    SvmOptionMaximizationPrecision -> maximizationPrecision,
				    SvmOptionMaximumIterations -> maximumNumberOfIterations,
				    SvmOptionIsPostProcess -> isPostProcess,
				    SvmOptionScalingFactor -> scalingFactor,
				    SvmOptionAlphaValueLimit -> alphaValueLimit,
				    SvmOptionEpsilonQpSolve -> epsilonQpSolve,
				    SvmOptionTauQpSolve -> tauQpSolve,
				    SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					ClusterOptionMethod -> clusterMethod,
				    ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				    ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				    ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
			   	    DataTransformationOptionTargetInterval -> targetInterval,
			   	    DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
				    UtilityOptionBlackListLength -> blackListLength
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				ScanClassTrainingWithSvmSC[
					classificationDataSet,
					kernelFunction,
					trainingFractionList,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
				    SvmOptionMaximizationPrecision -> maximizationPrecision,
				    SvmOptionMaximumIterations -> maximumNumberOfIterations,
				    SvmOptionIsPostProcess -> isPostProcess,
				    SvmOptionScalingFactor -> scalingFactor,
				    SvmOptionAlphaValueLimit -> alphaValueLimit,
				    SvmOptionEpsilonQpSolve -> epsilonQpSolve,
				    SvmOptionTauQpSolve -> tauQpSolve,
				    SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					ClusterOptionMethod -> clusterMethod,
				    ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				    ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				    ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
			   	    DataTransformationOptionTargetInterval -> targetInterval,
			   	    DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
				    UtilityOptionBlackListLength -> blackListLength
				]
			]
		]
	];

ScanClassTrainingWithSvmSC[

	(* Scans training and test set for different training fractions based on method FitSvm, see code.
	
	   Returns:
	   svmClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, svmInfo1}, {trainingAndTestSet2, svmInfo2}, ...}
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

	(* For details see method Kernel[] *)
	kernelFunction_,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			optimizationMethod,
			initialAlphasList,
			objectiveFunctionEpsilon,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			alphaValueLimit,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
	    	targetInterval,
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
			currentSvmInfo,
			pureFunction,
			trainingSetCorrectClassificationInPercent,
			testSetCorrectClassificationInPercent,
			svmTrainOptimization,
			trainingAndTestSetList,
			svmInfoList,
			bestIndex
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* SVM options *)
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
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
				svmTrainOptimization = 
					GetSvmTrainOptimization[
						classificationDataSet, 
						kernelFunction, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						SvmOptionOptimizationMethod -> optimizationMethod,
						SvmOptionInitialAlphasList -> initialAlphasList,
					    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
		    			SvmOptionMaximizationPrecision -> maximizationPrecision,
						SvmOptionMaximumIterations -> maximumNumberOfIterations,
						SvmOptionIsPostProcess -> isPostProcess,
						SvmOptionScalingFactor -> scalingFactor,
						SvmOptionAlphaValueLimit -> alphaValueLimit,
						SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
						SvmOptionTauQpSolve -> tauQpSolve,
						SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						DataTransformationOptionTargetInterval -> targetInterval,
						DataTransformationOptionNormalizationType -> normalizationType,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestSvmClassOptimization[svmTrainOptimization];
				trainingAndTestSetList = svmTrainOptimization[[3]];
				svmInfoList = svmTrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentSvmInfo = svmInfoList[[bestIndex]],
				
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
				currentSvmInfo = 
					FitSvm[
						currentTrainingSet,
						kernelFunction,
						SvmOptionOptimizationMethod -> optimizationMethod,
						SvmOptionInitialAlphasList -> initialAlphasList,
					    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
		    			SvmOptionMaximizationPrecision -> maximizationPrecision,
						SvmOptionMaximumIterations -> maximumNumberOfIterations,
						SvmOptionIsPostProcess -> isPostProcess,
						SvmOptionScalingFactor -> scalingFactor,
						SvmOptionAlphaValueLimit -> alphaValueLimit,
						SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
						SvmOptionTauQpSolve -> tauQpSolve,
						SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
						DataTransformationOptionTargetInterval -> targetInterval,
						DataTransformationOptionNormalizationType -> normalizationType
					]
			];
			
			pureFunction = Function[inputs, CalculateSvmClassNumbers[inputs, currentSvmInfo]];
			trainingSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTrainingSet, pureFunction];
			testSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTestSet, pureFunction];
			AppendTo[trainingAndTestSetsInfo, {currentTrainingAndTestSet, currentSvmInfo}];
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

ScanClassTrainingWithSvmPC[

	(* Scans training and test set for different training fractions based on method FitSvm, see code.
	
	   Returns:
	   svmClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, svmInfo1}, {trainingAndTestSet2, svmInfo2}, ...}
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

	(* For details see method Kernel[] *)
	kernelFunction_,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			optimizationMethod,
			initialAlphasList,
			objectiveFunctionEpsilon,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			alphaValueLimit,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
	    	targetInterval,
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
			currentSvmInfo,
			pureFunction,
			trainingSetCorrectClassificationInPercent,
			testSetCorrectClassificationInPercent,
			svmTrainOptimization,
			trainingAndTestSetList,
			svmInfoList,
			bestIndex,
			listOfTrainingAndTestSetsInfoAndScanReport
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* SVM options *)
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* Training set optimization options *)
		numberOfTrainingSetOptimizationSteps = UtilityOptionOptimizationSteps/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];

		ParallelNeeds[{"CIP`SVM`", "CIP`Cluster`", "CIP`DataTransformation`", "CIP`Utility`", "Combinatorica`"}];
		DistributeDefinitions[numberOfTrainingSetOptimizationSteps, optimizationMethod, initialAlphasList, objectiveFunctionEpsilon, 
			maximizationPrecision, maximumNumberOfIterations, isPostProcess, scalingFactor, alphaValueLimit, epsilonQpSolve, 
			tauQpSolve, penaltyConstantQpSolve, clusterMethod, maximumNumberOfEpochs, scalarProductMinimumTreshold, maximumNumberOfTrialSteps, 
			targetInterval, normalizationType, randomValueInitialization, deviationCalculationMethod, blackListLength];
		
		listOfTrainingAndTestSetsInfoAndScanReport = ParallelTable[
			If[numberOfTrainingSetOptimizationSteps > 0,
				
				(* ------------------------------------------------------------------------------------------------------ *)
				(* Training set optimization *)			
				svmTrainOptimization = 
					GetSvmTrainOptimization[
						classificationDataSet, 
						kernelFunction, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						SvmOptionOptimizationMethod -> optimizationMethod,
						SvmOptionInitialAlphasList -> initialAlphasList,
					    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
		    			SvmOptionMaximizationPrecision -> maximizationPrecision,
						SvmOptionMaximumIterations -> maximumNumberOfIterations,
						SvmOptionIsPostProcess -> isPostProcess,
						SvmOptionScalingFactor -> scalingFactor,
						SvmOptionAlphaValueLimit -> alphaValueLimit,
						SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
						SvmOptionTauQpSolve -> tauQpSolve,
						SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						DataTransformationOptionTargetInterval -> targetInterval,
						DataTransformationOptionNormalizationType -> normalizationType,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength
					];
				bestIndex = GetBestSvmClassOptimization[svmTrainOptimization];
				trainingAndTestSetList = svmTrainOptimization[[3]];
				svmInfoList = svmTrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentSvmInfo = svmInfoList[[bestIndex]],
				
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
				
				currentSvmInfo = 
					FitSvm[
						currentTrainingSet,
						kernelFunction,
						SvmOptionOptimizationMethod -> optimizationMethod,
						SvmOptionInitialAlphasList -> initialAlphasList,
					    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
		    			SvmOptionMaximizationPrecision -> maximizationPrecision,
						SvmOptionMaximumIterations -> maximumNumberOfIterations,
						SvmOptionIsPostProcess -> isPostProcess,
						SvmOptionScalingFactor -> scalingFactor,
						SvmOptionAlphaValueLimit -> alphaValueLimit,
						SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
						SvmOptionTauQpSolve -> tauQpSolve,
						SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
						DataTransformationOptionTargetInterval -> targetInterval,
						DataTransformationOptionNormalizationType -> normalizationType
					];
				
				];
			
			pureFunction = Function[inputs, CalculateSvmClassNumbers[inputs, currentSvmInfo]];
			trainingSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTrainingSet, pureFunction];
			testSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTestSet, pureFunction];
			{
				{currentTrainingAndTestSet, currentSvmInfo},
			
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

ScanRegressTrainingWithSvm[

	(* Scans training and test set for different training fractions based on method FitSvm, see code.
	
	   Returns:
	   svmRegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, svmInfo1}, {trainingAndTestSet2, svmInfo2}, ...}
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

	(* For details see method Kernel[] *)
	kernelFunction_,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			optimizationMethod,
			initialAlphasList,
			objectiveFunctionEpsilon,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			alphaValueLimit,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
	    	targetInterval,
	    	normalizationType,

			numberOfTrainingSetOptimizationSteps,
			deviationCalculationMethod,
			blackListLength,

			parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* SVM options *)
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
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
				ScanRegressTrainingWithSvmPC[
					dataSet,
					kernelFunction,
					trainingFractionList,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
				    SvmOptionMaximizationPrecision -> maximizationPrecision,
				    SvmOptionMaximumIterations -> maximumNumberOfIterations,
				    SvmOptionIsPostProcess -> isPostProcess,
				    SvmOptionScalingFactor -> scalingFactor,
				    SvmOptionAlphaValueLimit -> alphaValueLimit,
				    SvmOptionEpsilonQpSolve -> epsilonQpSolve,
				    SvmOptionTauQpSolve -> tauQpSolve,
				    SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					ClusterOptionMethod -> clusterMethod,
				    ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				    ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				    ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
			   	    DataTransformationOptionTargetInterval -> targetInterval,
			   	    DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
				    UtilityOptionBlackListLength -> blackListLength
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				ScanRegressTrainingWithSvmSC[
					dataSet,
					kernelFunction,
					trainingFractionList,
					SvmOptionOptimizationMethod -> optimizationMethod,
					SvmOptionInitialAlphasList -> initialAlphasList,
				    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
				    SvmOptionMaximizationPrecision -> maximizationPrecision,
				    SvmOptionMaximumIterations -> maximumNumberOfIterations,
				    SvmOptionIsPostProcess -> isPostProcess,
				    SvmOptionScalingFactor -> scalingFactor,
				    SvmOptionAlphaValueLimit -> alphaValueLimit,
				    SvmOptionEpsilonQpSolve -> epsilonQpSolve,
				    SvmOptionTauQpSolve -> tauQpSolve,
				    SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
					ClusterOptionMethod -> clusterMethod,
				    ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				    ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				    ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				    UtilityOptionRandomInitializationMode -> randomValueInitialization,
			   	    DataTransformationOptionTargetInterval -> targetInterval,
			   	    DataTransformationOptionNormalizationType -> normalizationType,
					UtilityOptionOptimizationSteps -> numberOfTrainingSetOptimizationSteps,
					UtilityOptionDeviationCalculation -> deviationCalculationMethod,
				    UtilityOptionBlackListLength -> blackListLength
				]
			]
		]
	];

ScanRegressTrainingWithSvmSC[

	(* Scans training and test set for different training fractions based on method FitSvm, see code.
	
	   Returns:
	   svmRegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, svmInfo1}, {trainingAndTestSet2, svmInfo2}, ...}
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

	(* For details see method Kernel[] *)
	kernelFunction_,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			optimizationMethod,
			initialAlphasList,
			objectiveFunctionEpsilon,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			alphaValueLimit,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
	    	targetInterval,
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
			currentSvmInfo,
			pureFunction,
			trainingSetRMSE,
			testSetRMSE,
			svmTrainOptimization,
			trainingAndTestSetList,
			svmInfoList,
			bestIndex
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* SVM options *)
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
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
				svmTrainOptimization = 
					GetSvmTrainOptimization[
						dataSet, 
						kernelFunction, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						SvmOptionOptimizationMethod -> optimizationMethod,
						SvmOptionInitialAlphasList -> initialAlphasList,
					    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
		    			SvmOptionMaximizationPrecision -> maximizationPrecision,
						SvmOptionMaximumIterations -> maximumNumberOfIterations,
						SvmOptionIsPostProcess -> isPostProcess,
						SvmOptionScalingFactor -> scalingFactor,
						SvmOptionAlphaValueLimit -> alphaValueLimit,
						SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
						SvmOptionTauQpSolve -> tauQpSolve,
						SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						DataTransformationOptionTargetInterval -> targetInterval,
						DataTransformationOptionNormalizationType -> normalizationType,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength				        
					];
				bestIndex = GetBestSvmRegressOptimization[svmTrainOptimization];
				trainingAndTestSetList = svmTrainOptimization[[3]];
				svmInfoList = svmTrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentSvmInfo = svmInfoList[[bestIndex]],
				
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
				currentSvmInfo = 
					FitSvm[
						currentTrainingSet,
						kernelFunction,
						SvmOptionOptimizationMethod -> optimizationMethod,
						SvmOptionInitialAlphasList -> initialAlphasList,
					    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
		    			SvmOptionMaximizationPrecision -> maximizationPrecision,
						SvmOptionMaximumIterations -> maximumNumberOfIterations,
						SvmOptionIsPostProcess -> isPostProcess,
						SvmOptionScalingFactor -> scalingFactor,
						SvmOptionAlphaValueLimit -> alphaValueLimit,
						DataTransformationOptionTargetInterval -> targetInterval,
						DataTransformationOptionNormalizationType -> normalizationType,
						SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
						SvmOptionTauQpSolve -> tauQpSolve,
						SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve
					]
			];

			pureFunction = Function[inputs, CalculateSvmOutputs[inputs, currentSvmInfo]];
			trainingSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTrainingSet, pureFunction]];
			testSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTestSet, pureFunction]];
			AppendTo[trainingAndTestSetsInfo, {currentTrainingAndTestSet, currentSvmInfo}];
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

ScanRegressTrainingWithSvmPC[

	(* Scans training and test set for different training fractions based on method FitSvm, see code.
	
	   Returns:
	   svmRegressopmTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, svmInfo1}, {trainingAndTestSet2, svmInfo2}, ...}
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

	(* For details see method Kernel[] *)
	kernelFunction_,

	(* trainingFractionList = {trainingFraction1, trainingFraction2, ...}
	   0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFractionList_/;VectorQ[trainingFractionList, NumberQ],
	
	(* Options *)
	opts___

	] :=
    
	Module[
      
		{
			optimizationMethod,
			initialAlphasList,
			objectiveFunctionEpsilon,
			maximizationPrecision,
			maximumNumberOfIterations,
			isPostProcess,
			scalingFactor,
			alphaValueLimit,
			epsilonQpSolve,
			tauQpSolve,
			penaltyConstantQpSolve,

	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
	    	targetInterval,
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
			currentSvmInfo,
			pureFunction,
			trainingSetRMSE,
			testSetRMSE,
			svmTrainOptimization,
			trainingAndTestSetList,
			svmInfoList,
			bestIndex,
			listOfTrainingAndTestSetsInfoAndScanReport
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* SVM options *)
		optimizationMethod = SvmOptionOptimizationMethod/.{opts}/.Options[SvmOptionsTraining];
		initialAlphasList = SvmOptionInitialAlphasList/.{opts}/.Options[SvmOptionsOptimization];
	    objectiveFunctionEpsilon = SvmOptionObjectiveFunctionEpsilon/.{opts}/.Options[SvmOptionsOptimization];
	    maximizationPrecision = SvmOptionMaximizationPrecision/.{opts}/.Options[SvmOptionsOptimization];
	    maximumNumberOfIterations = SvmOptionMaximumIterations/.{opts}/.Options[SvmOptionsOptimization];
	    isPostProcess = SvmOptionIsPostProcess/.{opts}/.Options[SvmOptionsOptimization];
	    scalingFactor = SvmOptionScalingFactor/.{opts}/.Options[SvmOptionsOptimization];
	    alphaValueLimit = SvmOptionAlphaValueLimit/.{opts}/.Options[SvmOptionsOptimization];
	    epsilonQpSolve = SvmOptionEpsilonQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    tauQpSolve = SvmOptionTauQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];
	    penaltyConstantQpSolve = SvmOptionPenaltyConstantQpSolve/.{opts}/.Options[SvmOptionsOpSolveOptimization];

		(* Cluster options *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    normalizationType = DataTransformationOptionNormalizationType/.{opts}/.Options[DataTransformationOptions];

		(* Training set optimization options *)
		numberOfTrainingSetOptimizationSteps = UtilityOptionOptimizationSteps/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];

		ParallelNeeds[{"CIP`SVM`", "CIP`Cluster`", "CIP`DataTransformation`", "CIP`Utility`", "Combinatorica`"}];
		DistributeDefinitions[numberOfTrainingSetOptimizationSteps, optimizationMethod, initialAlphasList, objectiveFunctionEpsilon, 
			maximizationPrecision, maximumNumberOfIterations, isPostProcess, scalingFactor, alphaValueLimit, epsilonQpSolve, 
			tauQpSolve, penaltyConstantQpSolve, clusterMethod, maximumNumberOfEpochs, scalarProductMinimumTreshold, maximumNumberOfTrialSteps, 
			targetInterval, normalizationType, randomValueInitialization, deviationCalculationMethod, blackListLength];
		
				
		listOfTrainingAndTestSetsInfoAndScanReport = ParallelTable[
			If[numberOfTrainingSetOptimizationSteps > 0,
			
				(* ------------------------------------------------------------------------------------------------------ *)
				(* Training set optimization *)	
				svmTrainOptimization = 
					GetSvmTrainOptimization[
						dataSet, 
						kernelFunction, 
						trainingFractionList[[i]],
						numberOfTrainingSetOptimizationSteps,
						SvmOptionOptimizationMethod -> optimizationMethod,
						SvmOptionInitialAlphasList -> initialAlphasList,
					    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
		    			SvmOptionMaximizationPrecision -> maximizationPrecision,
						SvmOptionMaximumIterations -> maximumNumberOfIterations,
						SvmOptionIsPostProcess -> isPostProcess,
						SvmOptionScalingFactor -> scalingFactor,
						SvmOptionAlphaValueLimit -> alphaValueLimit,
						SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
						SvmOptionTauQpSolve -> tauQpSolve,
						SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						DataTransformationOptionTargetInterval -> targetInterval,
						DataTransformationOptionNormalizationType -> normalizationType,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
				        UtilityOptionDeviationCalculation -> deviationCalculationMethod,
						UtilityOptionBlackListLength -> blackListLength		        
					];
				bestIndex = GetBestSvmRegressOptimization[svmTrainOptimization];
				trainingAndTestSetList = svmTrainOptimization[[3]];
				svmInfoList = svmTrainOptimization[[4]];
				currentTrainingAndTestSet = trainingAndTestSetList[[bestIndex]];
				currentTrainingSet = currentTrainingAndTestSet[[1]];
				currentTestSet = currentTrainingAndTestSet[[2]];
				currentSvmInfo = svmInfoList[[bestIndex]],
				
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
				
				currentSvmInfo =
					FitSvm[
						currentTrainingSet,
						kernelFunction,
						SvmOptionOptimizationMethod -> optimizationMethod,
						SvmOptionInitialAlphasList -> initialAlphasList,
					    SvmOptionObjectiveFunctionEpsilon -> objectiveFunctionEpsilon,
		    			SvmOptionMaximizationPrecision -> maximizationPrecision,
						SvmOptionMaximumIterations -> maximumNumberOfIterations,
						SvmOptionIsPostProcess -> isPostProcess,
						SvmOptionScalingFactor -> scalingFactor,
						SvmOptionAlphaValueLimit -> alphaValueLimit,
						DataTransformationOptionTargetInterval -> targetInterval,
						DataTransformationOptionNormalizationType -> normalizationType,
						SvmOptionEpsilonQpSolve -> epsilonQpSolve, 
						SvmOptionTauQpSolve -> tauQpSolve,
						SvmOptionPenaltyConstantQpSolve -> penaltyConstantQpSolve
					]
			];		

			pureFunction = Function[inputs, CalculateSvmOutputs[inputs, currentSvmInfo]];
			trainingSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTrainingSet, pureFunction]];
			testSetRMSE = Sqrt[CIP`Utility`GetMeanSquaredError[currentTestSet, pureFunction]];
			{
				{currentTrainingAndTestSet, currentSvmInfo},
			
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

ShowSvmOutput3D[

	(* Shows 3D SVM output.

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
    svmInfo_,
    
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
	
		dataSetScaleInfo = svmInfo[[2]];
		inputsMinMaxList = dataSetScaleInfo[[1, 1]];
		x1Min = inputsMinMaxList[[indexOfInput1, 1]];
		x1Max = inputsMinMaxList[[indexOfInput1, 2]];
		x2Min = inputsMinMaxList[[indexOfInput2, 1]];
		x2Max = inputsMinMaxList[[indexOfInput2, 2]];
		labels = 
			{
				StringJoin["In ", ToString[indexOfInput1]],
				StringJoin["In ", ToString[indexOfInput2]],
				StringJoin["Out ", ToString[indexOfOutput]]
			};
		
		Return[
			CIP`Graphics`PlotFunction3D[
				Function[{x1, x2}, CalculateSvmValue3D[x1, x2, indexOfInput1, indexOfInput2, indexOfOutput, input, svmInfo]], 
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

ShowSvmClassificationResult[

	(* Shows result of SVM classification for training and test set according to named property list.

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
    svmInfo_,
    
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
		ShowSvmSingleClassification[
			namedPropertyList,
			trainingSet, 
			svmInfo,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionMinMaxIndex -> minMaxIndex
		];
		
		(* Analyze test set *)
		If[Length[testSet] > 0,
			Print["Test Set:"];
			ShowSvmSingleClassification[
				namedPropertyList,
				testSet, 
				svmInfo,
				GraphicsOptionImageSize -> imageSize,
				GraphicsOptionMinMaxIndex -> minMaxIndex
			];
		]
	];

ShowSvmSingleClassification[

	(* Shows result of SVM classification for data set according to named property list.

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
    svmInfo_,
    
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

   		pureFunction = Function[inputs, CalculateSvmClassNumbers[inputs, svmInfo]];
		CIP`Graphics`ShowClassificationResult[
			namedPropertyList,
			dataSet, 
			pureFunction,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionMinMaxIndex -> minMaxIndex			
		]
	];

ShowSvmClassificationScan[

	(* Shows result of SVM based classification scan of clustered training sets.

	   Returns: Nothing *)


	(* svmClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, svmInfo1}, {trainingAndTestSet2, svmInfo2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, classification result in percent for training set}, {trainingFraction, classification result in percent for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)
	svmClassificationScan_,
	
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
			svmClassificationScan,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowSvmInputRelevanceClass[

	(* Shows svmInputComponentRelevanceListForClassification.

	   Returns: Nothing *)


	(* svmInputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, svmInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) correct classification in percent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best correct classification in percent of test set} *)
	svmInputComponentRelevanceListForClassification_,
	
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
			svmInputComponentRelevanceListForClassification,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowSvmInputRelevanceRegress[

	(* Shows svmInputComponentRelevanceListForRegression.

	   Returns: Nothing *)


	(* svmInputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, svmInfo}
	   trainingSetResult: {numberOfRemovedInputs, (best) RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)
	svmInputComponentRelevanceListForRegression_,
	
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
			svmInputComponentRelevanceListForRegression,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowSvmRegressionResult[

	(* Shows result of SVM regression for training and test set according to named property list.
	
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
    svmInfo_,
    
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
		ShowSvmSingleRegression[
			namedPropertyList,
			trainingSet, 
			svmInfo,
			GraphicsOptionPointSize -> pointSize,
			GraphicsOptionPointColor -> pointColor
		];
		
		(* Analyze test set *)
		If[Length[testSet] > 0,
			Print["Test Set:"];
			ShowSvmSingleRegression[
				namedPropertyList,
				testSet, 
				svmInfo,
				GraphicsOptionPointSize -> pointSize,
				GraphicsOptionPointColor -> pointColor
			]
		];
	];

ShowSvmSingleRegression[
    
	(* Shows result of SVM regression for data set according to named property list.
	
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
    svmInfo_,
    
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

		pureFunction = Function[inputs, CalculateSvmOutputs[inputs, svmInfo]];
		CIP`Graphics`ShowRegressionResult[
			namedPropertyList,
			dataSet, 
			pureFunction,
			GraphicsOptionPointSize -> pointSize,
			GraphicsOptionPointColor -> pointColor
		]
	];

ShowSvmRegressionScan[

	(* Shows result of SVM based regression scan of clustered training sets.

	   Returns: Nothing *)


	(* svmRegressionScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, svmInfo1}, {trainingAndTestSet2, svmInfo2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, RMSE for training set}, {trainingFraction, RMSE for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)
	svmRegressionScan_,
	
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
			svmRegressionScan,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowSvmSeriesClassificationResult[

	(* Shows result of SVM series classifications for training and test set.

	   Returns: Nothing *)

	(* svmSeriesClassificationResult: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in svmInfoList, classification result in percent for training set}
	   testPoint[[i]]: {index i in svmInfoList, classification result in percent for test set} *)
	svmSeriesClassificationResult_,
    
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

		trainingPoints2D = svmSeriesClassificationResult[[1]];
		testPoints2D = svmSeriesClassificationResult[[2]];

		If[Length[testPoints2D] > 0,
			
			(* Training and test set *)
			labels = {"svmInfo index", "Correct classifications [%]", "Training (green), Test (red)"};
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
			Print["Best test set classification with svmInfo index = ", bestIndexList],

			(* Training set only *)
			labels = {"svmInfo index", "Correct classifications [%]", "Training (green)"};
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
			Print["Best training set classification with svmInfo index = ", bestIndexList]
		]		
	];

ShowSvmSeriesRmse[

	(* Shows RMSE of SVM series for training and test set.

	   Returns: Nothing *)

	(* svmSeriesRmse: {trainingPoints2D, testPoints2D}
	   trainingPoints2D: {trainingPoint1, trainingPoint2, ...}
	   testPoints2D: {testPoint1, testPoint2, ...}
	   trainingPoint[[i]]: {index i in svmInfoList, RMSE for training set}
	   testPoint[[i]]: {index i in svmInfoList, RMSE for test set} *)
	svmSeriesRmse_,
    
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

		trainingPoints2D = svmSeriesRmse[[1]];
		testPoints2D = svmSeriesRmse[[2]];

		If[Length[testPoints2D] > 0,
			
			(* Training and test set *)
			labels = {"svmInfo index", "RMSE", "Training (green), Test (red)"};
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
			Print["Best test set regression with svmInfo index = ", bestIndexList],

			(* Training set only *)
			labels = {"svmInfo index", "RMSE", "Training (green)"};
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
			Print["Best training set regression with svmInfo index = ", bestIndexList]
		]		
	];

ShowSvmTrainOptimization[

	(* Shows training set optimization result of SVM.

	   Returns: Nothing *)


	(* svmTrainOptimization = {trainingSetRmseList, testSetRmseList, not interesting, not interesting}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set} *)
	svmTrainOptimization_,
    
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
			svmTrainOptimization, 
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

(* ::Section:: *)
(* End of Package *)

End[]

EndPackage[]
