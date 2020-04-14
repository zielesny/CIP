(*
--------------------------------------------------------------------------
Computational Intelligence Packages (CIP): Package DataTransformation
Version 3.1 for Mathematica 11 or higher
--------------------------------------------------------------------------

Author: Achim Zielesny

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
--------------------------------------------------------------------------
*)

(* ::Section:: *)
(* Package and dependencies *)

BeginPackage["CIP`DataTransformation`", {"CIP`Utility`"}]

(* ::Section:: *)
(* Off settings *)

Off[General::"spell1"]
Off[General::shdw]
Off[General::compat]

(* ::Section:: *)
(* Options *)

Options[DataTransformationOptions] = 
{
	(* targetInterval contains the desired minimum and maximum value for each column
	   targetInterval: {targetMin, targetMax} 
	   targetMin: Minimum value for each column 
	   targetMax: Maximum value for each column *)
	DataTransformationOptionTargetInterval -> {0.05, 0.95},
	
	(* Normalization type
	   {"None", {}}    : No normalization 
	   {"HampelTanh"}  : Hampel-tanh normalization to intervall {-1, 1}
	   {"HampelTanh01"}: Hampel-tanh normalization to intervall {0, 1} 
	   Hampel-tanh normalization: 
	   Ross etal., Score normalization in multimodal biometric systems, Pattern Recognition 38 (2005) 2270-2285 *)
	DataTransformationOptionNormalizationType -> {"None", {}}	
}

(* ::Section:: *)
(* Declarations *)

AddErrorToXYData::usage = 
	"AddErrorToXYData[xyData, errorValue]"

ApplyLogToDataSetOutputs::usage = 
	"ApplyLogToDataSetOutputs[dataSet]"
	
ApplyLogToDataSetOutputsReverse::usage = 
	"ApplyLogToDataSetOutputsReverse[transformedDataSet, outputOffsets]"

ApplySqrtToDataSetOutputs::usage = 
	"ApplySqrtToDataSetOutputs[dataSet]"
	
ApplySqrtToDataSetOutputsReverse::usage = 
	"ApplySqrtToDataSetOutputsReverse[transformedDataSet, outputOffsets]"

BlurImageDataSet::usage = 
	"BlurImageDataSet[imageDataSet]"

CleanDataSet::usage = 
	"CleanDataSet[dataSet]"
	
ConvertImageDataSet::usage = 
	"ConvertImageDataSet[imageDataSet]"

ConvertToImageDataSet::usage = 
	"ConvertToImageDataSet[dataSet, widthOfImageInPixel, heightOfImageInPixel]"

CorrectDataSetScaleInfoForLogApplication::usage = 
	"CorrectDataSetScaleInfoForLogApplication[dataSetScaleInfo]"

CorrectDataSetScaleInfoForLogApplicationReverse::usage = 
	"CorrectDataSetScaleInfoForLogApplicationReverse[correctedDataSetScaleInfo, outputOffsets]"

CorrectDataSetScaleInfoForSqrtApplication::usage = 
	"CorrectDataSetScaleInfoForSqrtApplication[dataSetScaleInfo]"

CorrectDataSetScaleInfoForSqrtApplicationReverse::usage = 
	"CorrectDataSetScaleInfoForSqrtApplicationReverse[correctedDataSetScaleInfo, outputOffsets]"

GetDataMatrixScaleInfo::usage = 
	"GetDataMatrixScaleInfo[dataMatrix, targetInterval]"

GetSpecificClassDataSubSet::usage = 
	"GetSpecificClassDataSubSet[classificationDataSet, classIndex]"

GetDataSetNormalizationInfo::usage = 
	"GetDataSetNormalizationInfo[dataSet, normalizationType]"

GetDataSetNormalizationInfoForTrainingAndTestSet::usage = 
	"GetDataSetNormalizationInfoForTrainingAndTestSet[trainingAndTestSet, normalizationType]"

GetDataSetPart::usage = 
	"GetDataSetPart[dataSet, indexList]"

GetDataSetScaleInfo::usage = 
	"GetDataSetScaleInfo[dataSet, targetIntervalInputs, targetIntervalOutputs]"

GetDataSetScaleInfoForTrainingAndTestSet::usage = 
	"GetDataSetScaleInfoForTrainingAndTestSet[trainingAndTestSet, targetIntervalInputs, targetIntervalOutputs]"

GetInputsForSpecifiedClass::usage = 
	"GetInputsForSpecifiedClass[classificationDataSet, classIndex]"

GetPartialClassificationDataSet::usage = 
	"GetPartialClassificationDataSet[classificationDataSet, classNumbers]"

GetPartialClassificationTrainingAndTestSet::usage = 
	"GetPartialClassificationTrainingAndTestSet[trainingAndTestSet, classNumbers]"

IncludeInputComponentsOfDataSet::usage = 
	"IncludeInputComponentsOfDataSet[dataSet, inputComponentInclusionList]"

NormalizeDataMatrix::usage = 
	"NormalizeDataMatrix[dataMatrix, normalizationInfo]"

NormalizeInputsOfDataSet::usage = 
	"NormalizeInputsOfDataSet[dataSet, normalizationInfo]"

NormalizeInputsOfTrainingAndTestSet::usage = 
	"NormalizeInputsOfTrainingAndTestSet[trainingAndTestSet, normalizationInfo]"
	
RemoveInputComponentsOfDataSet::usage = 
	"RemoveInputComponentsOfDataSet[dataSet, inputComponentRemovalList]"

RemoveNonNumberIoPairs::usage = 
	"RemoveNonNumberIoPairs[dataSet]"

RemoveOutputComponentsOfDataSet::usage = 
	"RemoveOutputComponentsOfDataSet[dataSet, outputComponentRemovalList]"

ScaleAndNormalizeDataMatrix::usage = 
	"ScaleAndNormalizeDataMatrix[dataMatrix, dataMatrixScaleInfo, normalizationInfo]"
	
ScaleDataMatrix::usage = 
	"ScaleDataMatrix[dataMatrix, dataMatrixScaleInfo]"
	
ScaleDataMatrixReverse::usage = 
	"ScaleDataMatrixReverse[dataMatrix, dataMatrixScaleInfo]"

ScaleAndNormalizeDataSet::usage = 
	"ScaleAndNormalizeDataSet[dataSet, dataSetScaleInfo, normalizationInfo]"
	ScaleDataSet::usage = 
	"ScaleDataSet[dataSet, dataSetScaleInfo]"

ScaleSizeOfImageDataSet::usage = 
	"ScaleSizeOfImageDataSet[imageDataSet, scaleFactor]"

ScaleAndNormalizeTrainingAndTestSet::usage = 
	"ScaleAndNormalizeTrainingAndTestSet[trainingAndTestSet, dataSetScaleInfo, normalizationInfo]"

ScaleTrainingAndTestSet::usage = 
	"ScaleTrainingAndTestSet[trainingAndTestSet, dataSetScaleInfo]"

SmoothWithFFT::usage = 
	"SmoothWithFFT[yData, threshold]"

SortClassificationDataSet::usage = 
	"SortClassificationDataSet[classificationDataSet]"

SplitClassificationDataSet::usage = 
	"SplitClassificationDataSet[dataSet]"

SplitDataSet::usage = 
	"SplitDataSet[dataSet, inputComponentForSplit]"

SplitDataSetByMinMeanMedianDifference::usage = 
	"SplitDataSetByMinMeanMedianDifference[dataSet]"

TransformDataMatrixByExp::usage = 
	"TransformDataMatrixByExp[dataMatrix]"

TransformDataMatrixByLog::usage = 
	"TransformDataMatrixByLog[dataMatrix]"
	
TransformDataSetToMultipleDataSet::usage = 
	"TransformDataSetToMultipleDataSet[dataSet]"
	
TransformLinear::usage = 
	"TransformLinear[x, min1, max1, min2, max2]"
	
TransformXYDataToDataSet::usage = 
	"TransformXYDataToDataSet[xyData]"
	
TransformXyErrorDataToDataSet::usage = 
	"TransformXyErrorDataToDataSet[xyErrorData]"

(* ::Section:: *)
(* Functions *)

Begin["`Private`"]

AddErrorToXYData[

	(* Adds error to (x,y) data.

	   Returns:
	   (x,y,error) data: {{x1, y1, error}, {x2, y2, error}, ...} *)


	(* {{x1, y1}, {x2, y2}, ...} *)      
	xyData_/;MatrixQ[xyData, NumberQ],
	
	errorValue_?NumberQ
      
	] :=
    
    Module[
      
		{i},
      
		Return[
			Table[
				{xyData[[i, 1]], xyData[[i, 2]], errorValue},
								
				{i, Length[xyData]}
			]
		];
	]

ApplyLogToDataSetOutputs[

	(* Applies Log function to (possibly incremented) data set outputs.

	   Returns:
	   {transformedDataSet, outputOffsets}
	   transformedDataSet with same structure as dataSet
	   outputOffsets: {outputOffset1, outputOffset2, ..., outputOffset<NumberOFOutputComponents>} *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_
        
	] :=
  
	Module[
    
		{
			i,
			inputs,
			transformedDataSet,
			newOutputs,
			outputOffsets,
			outputs,
			outputsMinList
		},
    
		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];
	    outputs = CIP`Utility`GetOutputsOfDataSet[dataSet];
	    
	    outputsMinList = CIP`Utility`GetMinList[outputs];
	    outputOffsets = 
			Table[
				If[outputsMinList[[i]] <= 0.0,
					
				    (* NOTE: Value of 1.0 must correspond to value in method CorrectDataSetScaleInfoForLogApplication[] *)
					1.0 - outputsMinList[[i]],
					
					0.0
				],
				
				{i, Length[outputsMinList]}
			];

	    newOutputs = 
			Table[
				N[Log[outputs[[i]] + outputOffsets]],
				
				{i, Length[outputs]}
			];

    	transformedDataSet = 
			Table[
				{inputs[[i]], newOutputs[[i]]}, 
					
				{i, Length[inputs]}
			];
		Return[{transformedDataSet, outputOffsets}]
	];

ApplyLogToDataSetOutputsReverse[

	(* Creates original data set from the one transformed by method ApplyLogToDataSetOutputs[].

	   Returns:
	   Original data set *)

    
	(* Data set transformed by method ApplyLogToDataSetOutputs[]
	   dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    transformedDataSet_,
    
	(* Offsets for output components created by method ApplyLogToDataSetOutputs[] *)
    outputOffsets_
        
	] :=
  
	Module[
    
		{
			i,
			inputs,
			originalDataSet,
			originalOutputs,
			outputs
		},
    
		inputs = CIP`Utility`GetInputsOfDataSet[transformedDataSet];
	    outputs = CIP`Utility`GetOutputsOfDataSet[transformedDataSet];

	    originalOutputs = 
			Table[
				Exp[outputs[[i]]] - outputOffsets,
				
				{i, Length[outputs]}
			];
	    
    	originalDataSet = 
			Table[
				{inputs[[i]], originalOutputs[[i]]}, 
					
				{i, Length[inputs]}
			];
		Return[originalDataSet]
	];

ApplySqrtToDataSetOutputs[

	(* Applies Sqrt function to (possibly incremented) data set outputs.

	   Returns:
	   {transformedDataSet, outputOffsets}
	   transformedDataSet with same structure as dataSet
	   outputOffsets: {outputOffset1, outputOffset2, ..., outputOffset<NumberOFOutputComponents>} *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_
        
	] :=
  
	Module[
    
		{
			i,
			inputs,
			transformedDataSet,
			newOutputs,
			outputOffsets,
			outputs,
			outputsMinList
		},
    
		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];
	    outputs = CIP`Utility`GetOutputsOfDataSet[dataSet];
	    
	    outputsMinList = CIP`Utility`GetMinList[outputs];
	    outputOffsets = 
			Table[
				If[outputsMinList[[i]] <= 0.0,
					
				    (* NOTE: Value of 0.0 must correspond to value in method CorrectDataSetScaleInfoForSqrtApplication[] *)
					0.0 - outputsMinList[[i]],
					
					0.0
				],
				
				{i, Length[outputsMinList]}
			];

	    newOutputs = 
			Table[
				N[Sqrt[outputs[[i]] + outputOffsets]],
				
				{i, Length[outputs]}
			];

    	transformedDataSet = 
			Table[
				{inputs[[i]], newOutputs[[i]]}, 
					
				{i, Length[inputs]}
			];
		Return[{transformedDataSet, outputOffsets}]
	];

ApplySqrtToDataSetOutputsReverse[

	(* Creates original data set from the one transformed by method ApplySqrtToDataSetOutputs[].

	   Returns:
	   Original data set *)

    
	(* Data set transformed by method ApplyLogToDataSetOutputs[]
	   dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    transformedDataSet_,
    
	(* Offsets for output components created by method ApplyLogToDataSetOutputs[] *)
    outputOffsets_
        
	] :=
  
	Module[
    
		{
			i,
			k,
			inputs,
			originalDataSet,
			originalOutputs,
			outputs
		},
    
		inputs = CIP`Utility`GetInputsOfDataSet[transformedDataSet];
	    outputs = CIP`Utility`GetOutputsOfDataSet[transformedDataSet];

	    originalOutputs = 
			Table[
				Table[
					outputs[[i, k]]*outputs[[i, k]],
					
					{k, Length[outputs[[i]]]}
				] - outputOffsets,
				
				{i, Length[outputs]}
			];
	    
    	originalDataSet = 
			Table[
				{inputs[[i]], originalOutputs[[i]]}, 
					
				{i, Length[inputs]}
			];
		Return[originalDataSet]
	];

BlurImageDataSet[

	(* Blurs grayscale image data set.

	   Returns:
	   Blurred image data set *)

    
	(* Grayscale image data set *)
	imageDataSet_
    
    ] :=
  
	Module[
    
		{
			i,
			blurredImagedataSet,
			imageIoPair,
			imageInput,
			imageOutput,
			blurredImage,
			blurredImageInput
		},

		blurredImagedataSet = {};
		Do[
			imageIoPair = imageDataSet[[i]];
			imageInput = imageIoPair[[1]];
			imageOutput = imageIoPair[[2]];
			blurredImage = Blur[Image[imageInput, "Byte"]];
			blurredImageInput = ImageData[blurredImage, "Byte"];
			AppendTo[blurredImagedataSet, {blurredImageInput, imageOutput}],
			
			{i, Length[imageDataSet]}
		];

		Return[blurredImagedataSet]    
	];

CleanDataSet[

	(* Cleans data set: Removes constant components and components with non-defined numbers. Transforms all numbers to real values.

	   Returns:
	   Cleaned data set with possibly different structure as dataSet *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_

	] :=
  
	Module[
    
		{
			cleanedInputs,
			cleanedOutputs,
			inputs,
			inputComponentRemovalList,
			outputs,
			outputComponentRemovalList,
			i
		},

		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];
	    outputs = CIP`Utility`GetOutputsOfDataSet[dataSet];

		cleanedInputs = CIP`Utility`CleanMatrix[inputs];
	    cleanedOutputs = CIP`Utility`CleanMatrix[outputs];

	    inputComponentRemovalList = CIP`Utility`GetConstantComponentList[cleanedInputs];
	    outputComponentRemovalList = CIP`Utility`GetConstantComponentList[cleanedOutputs];
	    
		cleanedInputs = CIP`Utility`CleanMatrix[CIP`Utility`RemoveComponents[cleanedInputs, inputComponentRemovalList]];
	    cleanedOutputs = CIP`Utility`CleanMatrix[CIP`Utility`RemoveComponents[cleanedOutputs, outputComponentRemovalList]];
    	
		Return[
			Table[
				{cleanedInputs[[i]], cleanedOutputs[[i]]}, 
					
				{i, Length[cleanedInputs]}
			]
		]
	];

ConvertImageDataSet[

	(* Converts grayscale image data set to data set for machine learning.

	   Returns:
	   Data set for machine learning *)

    
	(* Grayscale image data set *)
	imageDataSet_
    
    ] :=
  
	Module[
    
		{
			i,
			dataSet,
			imageIoPair,
			imageInput,
			imageOutput
		},

		dataSet = {};
		Do[
			imageIoPair = imageDataSet[[i]];
			imageInput = imageIoPair[[1]];
			imageOutput = imageIoPair[[2]];
			AppendTo[dataSet, {Flatten[imageInput], imageOutput}],
			
			{i, Length[imageDataSet]}
		];

		Return[dataSet]    
	];

ConvertToImageDataSet[

	(* Converts inputs of data set to grayscale image data set.

	   Returns:
	   Grayscale image data set *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   NOTE: inputs must be convertable into grayscale images *)
    dataSet_,
    
    (* Width of image in pixel 
       NOTE: Width x Height = Length[singleInput] *)
    widthOfImageInPixel_?IntegerQ,
    
    (* Height of image in pixel 
       NOTE: Width x Height = Length[singleInput] *)
    heightOfImageInPixel_?IntegerQ
    
    ] :=
  
	Module[
    
		{
			i,
			k,
			imageDataSet,
			ioPair,
			input,
			imageInput,
			output,
			offset
		},

		imageDataSet = {};
		Do[
			ioPair = dataSet[[i]];
			input = ioPair[[1]];
			output = ioPair[[2]];
			imageInput =
				Table[
					offset = (k - 1) * widthOfImageInPixel;
					Take[input, {offset + 1, offset + widthOfImageInPixel}],
					
					{k, heightOfImageInPixel}
				];
			AppendTo[imageDataSet, {imageInput, output}],
			
			{i, Length[dataSet]}
		];

		Return[imageDataSet]    
	];

CorrectDataSetScaleInfoForLogApplication[

	(* Corrects data set scale info for Log function application to data set outputs.

	   Returns:
	   Corrected data set scale info. *)

    
	(* dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs}, see GetDataSetScaleInfo *)
	dataSetScaleInfo_
    
    ] :=
  
	Module[
    
		{
			correctedOutputsDataMatrixScaleInfo,
			correctedOutputsMinMaxList,
			outputOffset,
			outputsDataMatrixScaleInfo,
			outputsMinMaxList
		},

		outputsDataMatrixScaleInfo = dataSetScaleInfo[[2]];
		outputsMinMaxList = outputsDataMatrixScaleInfo[[1]];
		correctedOutputsMinMaxList = 
			Table[
				If[outputsMinMaxList[[i, 1]] <= 0.0,
					
				    (* NOTE: Value of 1.0 must correspond to value in method ApplyLogToDataSetOutputs[] *)
					outputOffset = 1.0 - outputsMinMaxList[[i, 1]];
					N[Log[outputsMinMaxList[[i]] + outputOffset]],
					
					N[Log[outputsMinMaxList[[i]]]]
				],
				
				{i, Length[outputsMinMaxList]}
			];
		correctedOutputsDataMatrixScaleInfo = {correctedOutputsMinMaxList, outputsDataMatrixScaleInfo[[2]]};

		Return[{dataSetScaleInfo[[1]], correctedOutputsDataMatrixScaleInfo}]    
	];

CorrectDataSetScaleInfoForLogApplicationReverse[

	(* Reverses method CorrectDataSetScaleInfoForLogApplication[].

	   Returns:
	   Original data set scale info. *)

    
	(* Corrected dataSetScaleInfo from method CorrectDataSetScaleInfoForLogApplication[] *)
	correctedDataSetScaleInfo_,
    
	(* Offsets for output components created by method ApplyLogToDataSetOutputs[] *)
    outputOffsets_
    
    ] :=
  
	Module[
    
		{
			correctedOutputsDataMatrixScaleInfo,
			correctedOutputsMinMaxList,
			i,
			originalOutputsDataMatrixScaleInfo,
			originalOutputsMinMaxList
		},

		correctedOutputsDataMatrixScaleInfo = correctedDataSetScaleInfo[[2]];
		correctedOutputsMinMaxList = correctedOutputsDataMatrixScaleInfo[[1]];

		originalOutputsMinMaxList =
			Table[
				Exp[correctedOutputsMinMaxList[[i]]] - outputOffsets[[i]],
				
				{i, Length[outputOffsets]}
			];

		originalOutputsDataMatrixScaleInfo = {originalOutputsMinMaxList, correctedOutputsDataMatrixScaleInfo[[2]]};

		Return[{correctedDataSetScaleInfo[[1]], originalOutputsDataMatrixScaleInfo}]    
	];

CorrectDataSetScaleInfoForSqrtApplication[

	(* Corrects data set scale info for Sqrt function application to data set outputs.

	   Returns:
	   Corrected data set scale info. *)

    
	(* dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs}, see GetDataSetScaleInfo *)
	dataSetScaleInfo_
    
    ] :=
  
	Module[
    
		{
			correctedOutputsDataMatrixScaleInfo,
			correctedOutputsMinMaxList,
			outputOffset,
			outputsDataMatrixScaleInfo,
			outputsMinMaxList
		},

		outputsDataMatrixScaleInfo = dataSetScaleInfo[[2]];
		outputsMinMaxList = outputsDataMatrixScaleInfo[[1]];
		correctedOutputsMinMaxList = 
			Table[
				If[outputsMinMaxList[[i, 1]] <= 0.0,
					
				    (* NOTE: Value of 0.0 must correspond to value in method ApplySqrtToDataSetOutputs[] *)
					outputOffset = 0.0 - outputsMinMaxList[[i, 1]];
					N[Sqrt[outputsMinMaxList[[i]] + outputOffset]],
					
					N[Sqrt[outputsMinMaxList[[i]]]]
				],
				
				{i, Length[outputsMinMaxList]}
			];
		correctedOutputsDataMatrixScaleInfo = {correctedOutputsMinMaxList, outputsDataMatrixScaleInfo[[2]]};

		Return[{dataSetScaleInfo[[1]], correctedOutputsDataMatrixScaleInfo}]    
	];

CorrectDataSetScaleInfoForSqrtApplicationReverse[

	(* Reverses method CorrectDataSetScaleInfoForSqrtApplication[].

	   Returns:
	   Original data set scale info. *)

    
	(* Corrected dataSetScaleInfo from method CorrectDataSetScaleInfoForLogApplication[] *)
	correctedDataSetScaleInfo_,
    
	(* Offsets for output components created by method ApplyLogToDataSetOutputs[] *)
    outputOffsets_
    
    ] :=
  
	Module[
    
		{
			correctedOutputsDataMatrixScaleInfo,
			correctedOutputsMinMaxList,
			i,
			k,
			originalOutputsDataMatrixScaleInfo,
			originalOutputsMinMaxList
		},

		correctedOutputsDataMatrixScaleInfo = correctedDataSetScaleInfo[[2]];
		correctedOutputsMinMaxList = correctedOutputsDataMatrixScaleInfo[[1]];

		originalOutputsMinMaxList =
			Table[
				Table[
					correctedOutputsMinMaxList[[i, k]]*correctedOutputsMinMaxList[[i, k]],
					
					{k, Length[correctedOutputsMinMaxList[[i]]]}
				] - outputOffsets[[i]],
				
				{i, Length[outputOffsets]}
			];

		originalOutputsDataMatrixScaleInfo = {originalOutputsMinMaxList, correctedOutputsDataMatrixScaleInfo[[2]]};

		Return[{correctedDataSetScaleInfo[[1]], originalOutputsDataMatrixScaleInfo}]    
	];

GetSpecificClassDataSubSet[

	(* Returns classification data sub set for specified class.

	   Returns:
	   classificationDataSubSet with the same structure as classificationDataSet *)

    
	(* classificationDataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} 
	   outputValue: 0/1
	   Data set must be a classification data set, i.e. the output components must 0/1 code a class,
	   i.e. class 4 of 5 must be coded {0.0, 0.0, 0.0, 1.0, 0.0} *)
    classificationDataSet_,
    
    (* Index of class *)
    classIndex_?IntegerQ
	    
	] :=
  
	Module[
    
		{
			i,
			classificationDataSubSet,
			ioPair,
			output
		},

		classificationDataSubSet = {};
		Do[
			ioPair = classificationDataSet[[i]];
			output = ioPair[[2]];
			If[N[output[[classIndex]]] == 1.0,
				AppendTo[classificationDataSubSet, ioPair]
			],
			
			{i, Length[classificationDataSet]}
		];

		Return[classificationDataSubSet]
	];

GetDataMatrixNormalizationInfo[

	(* Returns info for normalization of each data matrix component.

	   Returns:
	   normalizationInfo: {normalizationType, meanAndStandardDeviationList}
	   normalizationType: Normalization type
	   {"None", {}}    : No normalization 
	   {"HampelTanh"}  : Hampel-tanh normalization to intervall {-1, 1}
	   {"HampelTanh01"}: Hampel-tanh normalization to intervall {0, 1} 
	   Hampel-tanh normalization: 
	   Ross etal., Score normalization in multimodal biometric systems, Pattern Recognition 38 (2005) 2270-2285
	   
	   meanAndStandardDeviationList: {MeanAndStandardDeviation1, ..., MeanAndStandardDeviation<NumberOfComponentsInVectorOfMatrix>}
	   MeanAndStandardDeviation: {mean, standard deviation}
	   MeanAndStandardDeviation[[i]] corresponds to component [[i]] of vectors of data matrix *)

    
	(* dataMatrix: {dataVector1, dataVector2, ...}
	   dataVector: {dataValue1, dataValue2, ...} *)
    dataMatrix_/;MatrixQ[dataMatrix, NumberQ],
    
	(* Normalization type
	   {"None", {}}    : No normalization 
	   {"HampelTanh"}  : Hampel-tanh normalization to intervall {-1, 1}
	   {"HampelTanh01"}: Hampel-tanh normalization to intervall {0, 1} 
	   Hampel-tanh normalization: 
	   Ross etal., Score normalization in multimodal biometric systems, Pattern Recognition 38 (2005) 2270-2285 *)
    normalizationType_,
    
	(* dataMatrixScaleInfo: {minMaxList, targetInterval}, see GetDataMatrixScaleInfo *)
	dataMatrixScaleInfo_
    
	] :=
  
	Module[
    
		{
			scaledDataMatrix
		},

		If[normalizationType[[1]] == "None",
			
			Return[
				{normalizationType, {}}
			],
			
			scaledDataMatrix = ScaleDataMatrix[dataMatrix, dataMatrixScaleInfo];
			Return[
				{normalizationType, CIP`Utility`GetMeanAndStandardDeviationList[scaledDataMatrix]}
			]
		]
	];

GetDataMatrixScaleInfo[

	(* Returns info for linear transformation of each data matrix component from its min-max interval to target interval [targetMin, targetMax].

	   Returns:
	   dataMatrixScaleInfo: {minMaxList, targetInterval}

	   minMaxList contains the minimum and maximum values of each column of the matrix
	   minMaxList: {minMaxColumn1, ..., minMaxColumn1<numberOfColumns>}
	   minMaxColumn contains the minimum and maximum value of the column:
	   minMaxColumn: {minValueOfColumn, maxValueOfColumn}

	   targetInterval contains the desired minimum and maximum value for each column
	   targetInterval: {targetMin, targetMax} 
	   targetMin: Minimum value for each column 
	   targetMax: Maximum value for each column *)

    
	(* dataMatrix: {dataVector1, dataVector2, ...}
	   dataVector: {dataValue1, dataValue2, ...} *)
    dataMatrix_/;MatrixQ[dataMatrix, NumberQ],
    
	(* targetInterval contains the desired minimum and maximum value for each column
	   targetInterval: {targetMin, targetMax} 
	   targetMin: Minimum value for each column 
	   targetMax: Maximum value for each column *)
    targetInterval_/;VectorQ[targetInterval, NumberQ]
    
	] :=
  
	Module[
    
		{
			i,
			minMaxList
		},

		minMaxList = CIP`Utility`GetMinMaxList[dataMatrix];
		
		(* minMaxList must be checked: If max == min the max value must be arbitrarily incremented (otherwise linear scaling inevitably fails later) *)
		Do[
			If[minMaxList[[i, 1]] == minMaxList[[i, 2]],
				minMaxList[[i, 2]] = minMaxList[[i, 1]] + 1.0
			],
			
			{i, Length[minMaxList]}
		];
		
		Return[
			{minMaxList, targetInterval}
		]
	];

GetDataSetNormalizationInfo[

	(* Returns info for normalization of each component of inputs of data set.

	   Returns:
	   normalizationInfo: {normalizationType, meanAndStandardDeviationList}
	   normalizationType: Normalization type
	   {"None", {}}    : No normalization 
	   {"HampelTanh"}  : Hampel-tanh normalization to intervall {-1, 1}
	   {"HampelTanh01"}: Hampel-tanh normalization to intervall {0, 1} 
	   Hampel-tanh normalization: 
	   Ross etal., Score normalization in multimodal biometric systems, Pattern Recognition 38 (2005) 2270-2285
	   
	   meanAndStandardDeviationList: {MeanAndStandardDeviation1, ..., MeanAndStandardDeviation<NumberOfComponentsInVectorOfMatrix>}
	   MeanAndStandardDeviation: {mean, standard deviation}
	   MeanAndStandardDeviation[[i]] corresponds to component [[i]] of vectors of data matrix *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,
    
	(* Normalization type
	   {"None", {}}    : No normalization 
	   {"HampelTanh"}  : Hampel-tanh normalization to intervall {-1, 1}
	   {"HampelTanh01"}: Hampel-tanh normalization to intervall {0, 1} 
	   Hampel-tanh normalization: 
	   Ross etal., Score normalization in multimodal biometric systems, Pattern Recognition 38 (2005) 2270-2285 *)
    normalizationType_,
    
	(* dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs}, see GetDataSetScaleInfo *)    
    dataSetScaleInfo_
    
	] :=
  
	Module[
    
		{},
		
		Return[
			GetDataMatrixNormalizationInfo[CIP`Utility`GetInputsOfDataSet[dataSet], normalizationType, dataSetScaleInfo[[1]]]
		]
	];

GetDataSetNormalizationInfoForTrainingAndTestSet[

	(* Returns inputs normalization info for training and test set.

	   Returns:
	   normalizationInfo: {normalizationType, meanAndStandardDeviationList}
	   normalizationType: Normalization type
	   {"None", {}}    : No normalization 
	   {"HampelTanh"}  : Hampel-tanh normalization to intervall {-1, 1}
	   {"HampelTanh01"}: Hampel-tanh normalization to intervall {0, 1} 
	   Hampel-tanh normalization: 
	   Ross etal., Score normalization in multimodal biometric systems, Pattern Recognition 38 (2005) 2270-2285
	   
	   meanAndStandardDeviationList: {MeanAndStandardDeviation1, ..., MeanAndStandardDeviation<NumberOfComponentsInVectorOfMatrix>}
	   MeanAndStandardDeviation: {mean, standard deviation}
	   MeanAndStandardDeviation[[i]] corresponds to component [[i]] of vectors of data matrix *)

    
	(* {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output: {outputComponent1, outputComponent2, ...}
	   testSet has the same structure and restrictions as trainingSet
	   NOTE: Data sets MUST be in original units *)
	trainingAndTestSet_,
    
	(* Normalization type
	   {"None", {}}    : No normalization 
	   {"HampelTanh"}  : Hampel-tanh normalization to intervall {-1, 1}
	   {"HampelTanh01"}: Hampel-tanh normalization to intervall {0, 1} 
	   Hampel-tanh normalization: 
	   Ross etal., Score normalization in multimodal biometric systems, Pattern Recognition 38 (2005) 2270-2285 *)
    normalizationType_,
    
	(* dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs}, see GetDataSetScaleInfo *)    
    dataSetScaleInfo_
    
	] :=
  
	Module[
    
		{
			dataSet
		},

    	dataSet = Join[trainingAndTestSet[[1]], trainingAndTestSet[[2]]];
    
		Return[
			GetDataSetNormalizationInfo[dataSet, normalizationType, dataSetScaleInfo]
		]
	];

GetDataSetPart[

	(* Returns part of data set according to index list.

	   Returns:
	   dataSetPart with same structure as dataSet *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* {index1, index2, ...} *)
	indexList_/;VectorQ[indexList, IntegerQ]
	    
	] :=
  
	Module[
    
		{
			i
		},

		If[Length[indexList] == 0,
			Return[{}]
		];
		Return[
			Table[
				dataSet[[indexList[[i]]]],
				
				{i, Length[indexList]}
			]
		]
	];

GetDataSetScaleInfo[

	(* Returns scale info for inputs and outpus of data set.

	   Returns:
	   dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs} *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,
    
	(* targetIntervalInputs contains the desired minimum and maximum value for each inputs column
	   targetInterval: {targetMin, targetMax} 
	   targetMin: Minimum value for each column 
	   targetMax: Maximum value for each column *)
    targetIntervalInputs_/;VectorQ[targetIntervalInputs, NumberQ],

	(* targetIntervalOutputs contains the desired minimum and maximum value for each outputs column
	   targetInterval: {targetMin, targetMax} 
	   targetMin: Minimum value for each column 
	   targetMax: Maximum value for each column *)
    targetIntervalOutputs_/;VectorQ[targetIntervalOutputs, NumberQ]
    
	] :=
  
	Module[
    
		{},

		Return[
			{
				GetDataMatrixScaleInfo[CIP`Utility`GetInputsOfDataSet[dataSet], targetIntervalInputs], 
				GetDataMatrixScaleInfo[CIP`Utility`GetOutputsOfDataSet[dataSet], targetIntervalOutputs]
			}
		]
	];

GetDataSetScaleInfoForTrainingAndTestSet[

	(* Returns scale info for training and test set.

	   Returns:
	   dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs} *)

    
	(* {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output: {outputComponent1, outputComponent2, ...}
	   testSet has the same structure and restrictions as trainingSet
	   NOTE: Data sets MUST be in original units *)
	trainingAndTestSet_,
    
	(* targetIntervalInputs contains the desired minimum and maximum value for each inputs column
	   targetInterval: {targetMin, targetMax} 
	   targetMin: Minimum value for each column 
	   targetMax: Maximum value for each column *)
    targetIntervalInputs_/;VectorQ[targetIntervalInputs, NumberQ],

	(* targetIntervalOutputs contains the desired minimum and maximum value for each outputs column
	   targetInterval: {targetMin, targetMax} 
	   targetMin: Minimum value for each column 
	   targetMax: Maximum value for each column *)
    targetIntervalOutputs_/;VectorQ[targetIntervalOutputs, NumberQ]
    
	] :=
  
	Module[
    
		{
			dataSet
		},

    	dataSet = Join[trainingAndTestSet[[1]], trainingAndTestSet[[2]]];
    
		Return[
			GetDataSetScaleInfo[dataSet, targetIntervalInputs, targetIntervalOutputs]
		]
	];

GetInputsForSpecifiedClass[

	(* Returns inputs of classification data sub set for specified class.

	   Returns:
	   Inputs of classificationDataSubSet *)

    
	(* classificationDataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} 
	   outputValue: 0/1
	   Data set must be a classification data set, i.e. the output components must 0/1 code a class,
	   i.e. class 4 of 5 must be coded {0.0, 0.0, 0.0, 1.0, 0.0} *)
    classificationDataSet_,
    
    (* Index of class *)
    classIndex_?IntegerQ
	    
	] :=
  
	Module[
    
		{
			classificationDataSubSet
		},

		classificationDataSubSet = GetSpecificClassDataSubSet[classificationDataSet, classIndex];
		
		Return[CIP`Utility`GetInputsOfDataSet[classificationDataSubSet]]
	];

GetPartialClassificationDataSet[
	
	(* Returns partial data set with IOPairs that belong to specified classes.
	
	   Returns:
	   Partial data set with IOPairs that belong to specified classes with the same structure as data set.
	   If classNumbers is empty specified data set is returned. *)

	(* classificationDataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} 
	   outputValue: 0/1
	   Data set must be a classification data set, i.e. the output components must 0/1 code a class,
	   i.e. class 4 of 5 must be coded {0.0, 0.0, 0.0, 1.0, 0.0} *)
    classificationDataSet_,
	
	(* List of class numbers *)
	classNumbers_/;VectorQ[classNumbers, IntegerQ]
	
    ] :=
  
	Module[
    
		{
			i,
			partialClassificationDataSet
		},
    
		If[Length[classNumbers] == 0, 
			Return[classificationDataSet]
		];
				
		partialClassificationDataSet = {};
		Do[
			If[MemberQ[classNumbers, CIP`Utility`GetPositionOfMaximumValue[classificationDataSet[[i, 2]]]],
				AppendTo[partialClassificationDataSet, classificationDataSet[[i]]]
			],
			
			{i, Length[classificationDataSet]}
		];
		
		Return[partialClassificationDataSet];
	];

GetPartialClassificationTrainingAndTestSet[
	
	(* Returns partial training and test set with IOPairs that belong to specified classes.
	
	   Returns:
	   Partial training and test set with IOPairs that belong to specified classes with the same structure as specified training and test set.
	   If classNumbers is empty specified traning and test set is returned. *)

	(* {trainingSet, testSet}
	   trainingSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output: {outputComponent1, outputComponent2, ...}
	   NOTE: Each component must be in [0, 1]
	   testSet has the same structure and restrictions as trainingSet *)
	trainingAndTestSet_,
	
	(* List of class numbers *)
	classNumbers_/;VectorQ[classNumbers, IntegerQ]
	
    ] :=
  
	Module[
    
		{
			testSet,
			trainingSet
		},
    
		If[Length[classNumbers] == 0, Return[trainingAndTestSet]];
				
		trainingSet = trainingAndTestSet[[1]];
		testSet = trainingAndTestSet[[2]];
		Return[
			{
				GetPartialClassificationDataSet[trainingSet, classNumbers], 
				GetPartialClassificationDataSet[testSet, classNumbers]
			}
		];
	];

IncludeInputComponentsOfDataSet[

	(* Includes specified input components of data set, i.e. removes all others

	   Returns:
	   Data set with included input components *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_ ,
    
    (* {first input component to be included, second input component to be included, ...} *)
    inputComponentInclusionList_/;VectorQ[inputComponentInclusionList, IntegerQ]
    
	] :=
  
	Module[
    
		{
			i,
			k,
			ioPair,
			input,
			newInput,
			output
		},
    
		Return[
			Table[
				ioPair = dataSet[[i]];
				input = ioPair[[1]];
				output = ioPair[[2]];
				newInput = {};
				Do[
					If[Length[Position[inputComponentInclusionList, k]] > 0, 
						newInput = {newInput, input[[k]]}
					],

					{k, Length[input]}
				];
				{Flatten[newInput], output},

				{i, Length[dataSet]}
			]
		];
	];

NormalizeDataMatrix[

	(* Normalizes data matrix with normalization info.

	   Returns:
	   Normalized data matrix with same structure as data matrix *)

    
	(* dataMatrix: {dataVector1, dataVector2, ...}
	   dataVector: {dataValue1, dataValue2, ...} *)
    dataMatrix_/;MatrixQ[dataMatrix, NumberQ],

	(* Normalization info: {normalizationType, meanAndStandardDeviationList}, see method GetDataMatrixNormalizationInfo *)
    normalizationInfo_
    
	] :=
  
	Module[
    
		{
			normalizedDataMatrix
		},
		
		If[normalizationInfo[[1, 1]] == "None",

			Return[dataMatrix],
			
			Switch[normalizationInfo[[1, 1]],
				
				"HampelTanh",
				normalizedDataMatrix = NormalizeDataMatrixWithHampelTanh[dataMatrix, normalizationInfo[[2]]],
				
				"HampelTanh01",
				normalizedDataMatrix = NormalizeDataMatrixWithHampelTanh01[dataMatrix, normalizationInfo[[2]]]			
			];
			Return[normalizedDataMatrix]
		]
	];

NormalizeDataMatrixWithHampelTanh[

	(* Performs Hampel-tanh {-1, 1} normalization on every column of data matrix, see code.

	   Returns:
	   Transformed matrix with same structure as dataMatrix *)

	(* dataMatrix: {dataVector1, dataVector2, ...}
	   dataVector: {dataValue1, dataValue2, ...} *)
    dataMatrix_/;MatrixQ[dataMatrix, NumberQ],
    
	(* meanAndStandardDeviationList: {MeanAndStandardDeviation1, ..., MeanAndStandardDeviation<NumberOfComponentsInVectorOfMatrix>}
	   MeanAndStandardDeviation: {mean, standard deviation}
	   MeanAndStandardDeviation[[i]] corresponds to component [[i]] of vectors of data matrix *)
    meanAndStandardDeviationList_/;MatrixQ[meanAndStandardDeviationList, NumberQ]

	] :=
  
	Module[
    
		{
			i,
			k
		},

		Return[
			Table[
				Table[
					If[meanAndStandardDeviationList[[i, 2]] == 0.0,
						0.0,
						Tanh[0.01 * (dataMatrix[[k, i]] - meanAndStandardDeviationList[[i, 1]])/meanAndStandardDeviationList[[i, 2]]]
					],
            
					{i, Length[dataMatrix[[1]]]}
				],
          
				{k, Length[dataMatrix]}
			]
		]
	];

NormalizeDataMatrixWithHampelTanh01[

	(* Performs Hampel-tanh {0, 1} normalization on every column of data matrix, see code.

	   Returns:
	   Transformed matrix with same structure as dataMatrix *)

	(* dataMatrix: {dataVector1, dataVector2, ...}
	   dataVector: {dataValue1, dataValue2, ...} *)
    dataMatrix_/;MatrixQ[dataMatrix, NumberQ],
    
	(* meanAndStandardDeviationList: {MeanAndStandardDeviation1, ..., MeanAndStandardDeviation<NumberOfComponentsInVectorOfMatrix>}
	   MeanAndStandardDeviation: {mean, standard deviation}
	   MeanAndStandardDeviation[[i]] corresponds to component [[i]] of vectors of data matrix *)
    meanAndStandardDeviationList_/;MatrixQ[meanAndStandardDeviationList, NumberQ]

	] :=
  
	Module[
    
		{
			i,
			k
		},

		Return[
			Table[
				Table[
					If[meanAndStandardDeviationList[[i, 2]] == 0.0,
						0.5,
						0.5 * (Tanh[0.01 * (dataMatrix[[k, i]] - meanAndStandardDeviationList[[i, 1]])/meanAndStandardDeviationList[[i, 2]]] + 1.0)
					],
            
					{i, Length[dataMatrix[[1]]]}
				],
          
				{k, Length[dataMatrix]}
			]
		]
	];

NormalizeInputsOfDataSet[

	(* Normalizes inputs of data set with normalizationInfo.

	   Returns:
	   Normalized inputs data set with same structure as dataSet *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* Normalization info: {normalizationType, meanAndStandardDeviationList}, see method GetDataMatrixNormalizationInfo *)
    normalizationInfo_
    
	] :=
  
	Module[
    
		{
			i,
			normalizedInputs,
			outputs
		},
		
		If[normalizationInfo[[1, 1]] == "None",

			Return[dataSet],

			normalizedInputs = NormalizeDataMatrix[CIP`Utility`GetInputsOfDataSet[dataSet], normalizationInfo];			
    		outputs = CIP`Utility`GetOutputsOfDataSet[dataSet];
			Return[
				Table[
					{normalizedInputs[[i]], outputs[[i]]}, 
				
					{i, Length[normalizedInputs]}
				]
			]
		]
	];

NormalizeInputsOfTrainingAndTestSet[

	(* Normalizes inputs of training and test set with normalizationInfo.

	   Returns:
	   Normalized inputs training and test set with same structure as trainingAndTestSet *)

    
	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet and testSet have dataSet structure:
	   dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    trainingAndTestSet_ ,

	(* Normalization info: {normalizationType, meanAndStandardDeviationList}, see method GetDataMatrixNormalizationInfo *)
    normalizationInfo_
    
	] :=
  
	Module[
    
		{},
		
    	If[Length[trainingAndTestSet[[2]]] > 0,

			Return[
				{
					NormalizeInputsOfDataSet[trainingAndTestSet[[1]], normalizationInfo], 
					NormalizeInputsOfDataSet[trainingAndTestSet[[2]], normalizationInfo]
				}
			],
			
   			Return[
				{
					NormalizeInputsOfDataSet[trainingAndTestSet[[1]], normalizationInfo], 
					{}
				}
			]
    	]
	];

RemoveInputComponentsOfDataSet[

	(* Removes specified input components of data set

	   Returns:
	   Data set with removed input components *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_ ,
    
    (* {first input component to be removed, second input component to be removed, ...} *)
    inputComponentRemovalList_/;VectorQ[inputComponentRemovalList, IntegerQ]
    
	] :=
  
	Module[
    
		{
			i,
			k,
			ioPair,
			input,
			newInput,
			output
		},
    
		Return[
			Table[
				ioPair = dataSet[[i]];
				input = ioPair[[1]];
				output = ioPair[[2]];
				newInput = {};
				Do[
					If[Length[Position[inputComponentRemovalList, k]] == 0, 
						newInput = {newInput, input[[k]]}
					],

					{k, Length[input]}
				];
				{Flatten[newInput], output},

				{i, Length[dataSet]}
			]
		];
	];

RemoveNonNumberIoPairs[

	(* Removes IO pairs of data set that contain non-number components.

	   Returns:
	   Data set with components that are numbers *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   NOTE: inputs must be convertable into grayscale images *)
    dataSet_
    
    ] :=
  
	Module[
    
		{
			i,
			newDataSet,
			ioPair,
			input,
			output
		},

		newDataSet = {};
		Do[
			ioPair = dataSet[[i]];
			input = ioPair[[1]];
			output = ioPair[[2]];
			If[!CIP`Utility`HasNonNumberComponent[input] && !CIP`Utility`HasNonNumberComponent[output],
				AppendTo[newDataSet, ioPair]
			],
			
			{i, Length[dataSet]}
		];

		Return[newDataSet]    
	];

RemoveOutputComponentsOfDataSet[

	(* Removes specified output components of data set

	   Returns:
	   Data set with removed output components *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_ ,
    
    (* {first output component to be removed, second output component to be removed, ...} *)
    outputComponentRemovalList_/;VectorQ[outputComponentRemovalList, IntegerQ]
    
	] :=
  
	Module[
    
		{
			i,
			k,
			ioPair,
			input,
			newOutput,
			output
		},
    
		Return[
			Table[
				ioPair = dataSet[[i]];
				input = ioPair[[1]];
				output = ioPair[[2]];
				newOutput = {};
				Do[
					If[Length[Position[outputComponentRemovalList, k]] == 0, 
						newOutput = {newOutput, input[[k]]}
					],

					{k, Length[output]}
				];
				{input, Flatten[newOutput]},

				{i, Length[dataSet]}
			]
		];
	];

ScaleAndNormalizeDataMatrix[

	(* Scales and normalizes data matrix with dataMatrixScaleInfo and normalizationInfo, see code.

	   Returns:
	   Returns scaled and normalized matrix with same structure as dataMatrix *)

	(* dataMatrix: {dataVector1, dataVector2, ...}
	   dataVector: {dataValue1, dataValue2, ...} *)
    dataMatrix_/;MatrixQ[dataMatrix, NumberQ],

	(* dataMatrixScaleInfo: {minMaxList, targetInterval}, see GetDataMatrixScaleInfo *)
	dataMatrixScaleInfo_,
	
	(* Normalization info: {normalizationType, meanAndStandardDeviationList}, see method GetDataMatrixNormalizationInfo *)
    normalizationInfo_
	    
	] :=
  
	Module[
    
		{},
    
		Return[NormalizeDataMatrix[ScaleDataMatrix[dataMatrix, dataMatrixScaleInfo], normalizationInfo]]
	];

ScaleDataMatrix[

	(* Scales data matrix with dataMatrixScaleInfo, see code.

	   Returns:
	   Returns scaled matrix with same structure as dataMatrix *)

	(* dataMatrix: {dataVector1, dataVector2, ...}
	   dataVector: {dataValue1, dataValue2, ...} *)
    dataMatrix_/;MatrixQ[dataMatrix, NumberQ],

	(* dataMatrixScaleInfo: {minMaxList, targetInterval}, see GetDataMatrixScaleInfo *)
	dataMatrixScaleInfo_
	    
	] :=
  
	Module[
    
		{
			i,
			k,
			minMaxList,
			targetMax,
			targetMin
		},
    
    	minMaxList = dataMatrixScaleInfo[[1]];
    	targetMin = dataMatrixScaleInfo[[2, 1]];
    	targetMax = dataMatrixScaleInfo[[2, 2]];

		Return[
			Table[
				Table[
					TransformLinear[dataMatrix[[k, i]], minMaxList[[i, 1]], minMaxList[[i, 2]], targetMin, targetMax],
            
					{i, Length[dataMatrix[[1]]]}
				],
          
				{k, Length[dataMatrix]}
			]
		]
	];
	
ScaleDataMatrixReverse[

	(* Inverse operation to ScaleDataMatrix.

	   Returns:
	   Returns original unscaled data matrix *)

	(* scaledDataMatrix: {scaledDataVector1, scaledDataVector2, ...}
	   scaledDataVector: {scaledDataValue1, scaledDataValue2, ...} *)
    scaledDataMatrix_/;MatrixQ[scaledDataMatrix, NumberQ],

	(* dataMatrixScaleInfo: {minMaxList, targetInterval}, see GetDataMatrixScaleInfo *)
	dataMatrixScaleInfo_

	] :=
  
	Module[
    
		{
			i,
			k,
			minMaxList,
			targetMax,
			targetMin
		},

    	minMaxList = dataMatrixScaleInfo[[1]];
    	targetMin = dataMatrixScaleInfo[[2, 1]];
    	targetMax = dataMatrixScaleInfo[[2, 2]];
    
		Return[
			Table[
				Table[
					TransformLinear[scaledDataMatrix[[k, i]], targetMin, targetMax, minMaxList[[i, 1]], minMaxList[[i, 2]]],
            
					{i, Length[scaledDataMatrix[[1]]]}
				],
          
				{k, Length[scaledDataMatrix]}
			]
		];
	];

ScaleAndNormalizeDataSet[

	(* Scales and normalizes data set with dataSetScaleInfo and normalizationInfo (for inputs), see code.

	   Returns:
	   Returns scaled and (inputs) normalized data set with same structure as data set *)

	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs}, see GetDataSetScaleInfo *)    
    dataSetScaleInfo_,
	
	(* Normalization info: {normalizationType, meanAndStandardDeviationList}, see method GetDataMatrixNormalizationInfo *)
    normalizationInfo_
	    
	] :=
  
	Module[
    
		{},
    
		Return[NormalizeInputsOfDataSet[ScaleDataSet[dataSet, dataSetScaleInfo], normalizationInfo]]
	];

ScaleDataSet[

	(* Scales data set with dataSetScaleInfo.

	   Returns:
	   Scaled data set with same structure as dataSet *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs}, see GetDataSetScaleInfo *)    
    dataSetScaleInfo_
    
	] :=
  
	Module[
    
		{
			i,
			scaledInputs,
			scaledOutputs
		},
    
		scaledInputs = ScaleDataMatrix[CIP`Utility`GetInputsOfDataSet[dataSet], dataSetScaleInfo[[1]]];
	    scaledOutputs = ScaleDataMatrix[CIP`Utility`GetOutputsOfDataSet[dataSet], dataSetScaleInfo[[2]]];
    
		Return[
			Table[
				{scaledInputs[[i]], scaledOutputs[[i]]}, 
					
				{i, Length[scaledInputs]}
			]
		];
	];

ScaleSizeOfImageDataSet[

	(* Scales size of grayscale images of image data set.

	   Returns:
	   Image data set with scaled image sizes*)

    
	(* Grayscale image data set *)
	imageDataSet_,

	(* Scale factor (< 1.0: Reduced image size, > 1.0: Enlarged image size *)
	scaleFactor_?NumberQ
    
    ] :=
  
	Module[
    
		{
			i,
			scaledImagedataSet,
			imageIoPair,
			imageInput,
			imageOutput,
			scaledImage,
			scaledImageInput
		},

		scaledImagedataSet = {};
		Do[
			imageIoPair = imageDataSet[[i]];
			imageInput = imageIoPair[[1]];
			imageOutput = imageIoPair[[2]];
			scaledImage = ImageResize[Image[imageInput, "Byte"], Scaled[scaleFactor]];
			scaledImageInput = ImageData[scaledImage, "Byte"];
			AppendTo[scaledImagedataSet, {scaledImageInput, imageOutput}],
			
			{i, Length[imageDataSet]}
		];

		Return[scaledImagedataSet]    
	];

ScaleAndNormalizeTrainingAndTestSet[

	(* Scales and normalizes training and test set with dataSetScaleInfo and normalizationInfo (for inputs), see code.

	   Returns:
	   Returns scaled and (inputs) normalized training and test set with same structure as training and test set *)

	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet and testSet have dataSet structure:
	   dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    trainingAndTestSet_ ,

	(* dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs}, see GetDataSetScaleInfo *)    
    dataSetScaleInfo_,
	
	(* Normalization info: {normalizationType, meanAndStandardDeviationList}, see method GetDataMatrixNormalizationInfo *)
    normalizationInfo_
	    
	] :=
  
	Module[
    
		{},
    
		Return[NormalizeInputsOfTrainingAndTestSet[ScaleTrainingAndTestSet[trainingAndTestSet, dataSetScaleInfo], normalizationInfo]]
	];

ScaleTrainingAndTestSet[

	(* Scales training and test set with dataSetScaleInfo.

	   Returns:
	   scaledTrainingAndTestSet with same structure as trainingAndTestSet *)

    
	(* trainingAndTestSet: {trainingSet, testSet}
	   trainingSet and testSet have dataSet structure:
	   dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    trainingAndTestSet_ ,
    
	(* dataSetScaleInfo: {dataMatrixScaleInfo for inputs, dataMatrixScaleInfo for outputs}, see GetDataSetScaleInfo  *)    
    dataSetScaleInfo_
    
	] :=
  
	Module[
    
		{},
    
    	If[Length[trainingAndTestSet[[2]]] > 0,

			Return[
				{
					ScaleDataSet[trainingAndTestSet[[1]], dataSetScaleInfo], 
					ScaleDataSet[trainingAndTestSet[[2]], dataSetScaleInfo]
				}
			],
			
   			Return[
				{
					ScaleDataSet[trainingAndTestSet[[1]], dataSetScaleInfo], 
					{}
				}
			]
    	]
	];

SmoothWithFFT[

	(* Smoothes yData with FFT by removal of frequencies with power smaller than threshold.

	   Returns:
	   Smoothed yData *)

    
	(* yData: {y1, y2, ... } *)
    yData_ /;VectorQ[yData, NumberQ],
    
	(* Threshold: Fraction of (powerMax - powerMin) *)    
    threshold_?NumberQ
    
	] :=
  
	Module[
    
		{i, yDataFFT, yPower, yPowerThreshold},
    
		yDataFFT = Fourier[yData];
        yPower = Abs[yDataFFT];
        yPowerThreshold = Max[yPower] * threshold; 
		Do[
  			If[yPower[[i]] < yPowerThreshold,
   				yDataFFT[[i]] = 0.0
			],
 			{i, Length[yDataFFT]}
		];

		Return[Re[InverseFourier[yDataFFT]]]	
	];

SortClassificationDataSet[

	(* Sorts classification data set according to its classes.

	   Returns:
	   {sortedClassificationDataSet, classIndexMinMaxList}
	   sortedClassificationDataSet: classificationDataSet sorted according to its classes 
	   classIndexMinMaxList: {minMaxIndex for class 1, minMaxIndex for class 2, ...} 
	   minMaxIndex: {minimum index, maximum index} *)

    
	(* classificationDataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} 
	   outputValue: 0/1
	   Data set must be a classification data set, i.e. the output components must 0/1 code a class,
	   i.e. class 4 of 5 must be coded {0, 0, 0, 1, 0} *)
    classificationDataSet_
	    
	] :=
  
	Module[
    
		{
			i,
			k,
			numberOfOutputComponents,
			sortedClassificationDataSet,
			ioPair,
			output,
			position,
			classIndexMinMaxList,
			startIndex
		},

		numberOfOutputComponents = Length[classificationDataSet[[1, 2]]];
		sortedClassificationDataSet = Table[{}, {numberOfOutputComponents}];
		Do[
			ioPair = classificationDataSet[[i]];
			output = ioPair[[2]];
			position = First[Flatten[Position[N[output], 1.0]]];
			sortedClassificationDataSet[[position]] = Append[sortedClassificationDataSet[[position]], ioPair],
			
			{i, Length[classificationDataSet]}
		];
		sortedClassificationDataSet = Flatten[sortedClassificationDataSet, 1];
		
		classIndexMinMaxList = Table[{0, 0}, {numberOfOutputComponents}];
		startIndex = 1;
		Do[
			classIndexMinMaxList[[i, 1]] = startIndex;
			classIndexMinMaxList[[i, 2]] = startIndex - 1;
			Do[
				ioPair = sortedClassificationDataSet[[k]];
				output = ioPair[[2]];
				position = First[Flatten[Position[N[output], 1.0]]];
				If[position == i,
					
					classIndexMinMaxList[[i, 2]] = classIndexMinMaxList[[i, 2]] + 1,
					
					Break[]
				],
				
				{k, startIndex, Length[sortedClassificationDataSet]}
			];
			If[i < numberOfOutputComponents,
				startIndex = classIndexMinMaxList[[i, 2]] + 1
			],
						
			{i, numberOfOutputComponents}
		];
				
		Return[{sortedClassificationDataSet, classIndexMinMaxList}]
	];

SplitClassificationDataSet[

	(* Splits classification data set so that the number of classes in new data sets is most similar to that of original data set.

	   Returns:
	   splitInfo:        {dataSetSplitInfo1, dataSetSplitInfo2}
	   dataSetSplitInfo: {dataSet, inputComponentForSplit, {min of component, max of component}} *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   NOTE: inputs must be convertable into grayscale images *)
    dataSet_
    
    ] :=
  
	Module[
    
		{
			classNumber,
			inputComponentForSplit,
			i,
			k,
			input,
			output,
			inputs,
			outputs,
			median,
			distance,
			minDistance,
			relativeClassCountVector,
			relativeClassCountVector1,
			relativeClassCountVector2,
			classCountVector1,
			classCountVector2,
			count1,
			count2
		},

		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];
		outputs = CIP`Utility`GetOutputsOfDataSet[dataSet];
		relativeClassCountVector = N[CIP`Utility`GetClassCountVector[outputs]/Length[outputs]];
		minDistance = Infinity;
		Do[
			median = Median[inputs[[All, i]]];
			classCountVector1 = Table[0, {Length[outputs[[1]]]}];
			classCountVector2 = Table[0, {Length[outputs[[1]]]}];
			count1 = 0;
			count2 = 0;
			Do[
				input = inputs[[k]];
				output = outputs[[k]];
				If[input[[i]] < median,
					
					count1++;
					classNumber = CIP`Utility`GetPositionOfMaximumValue[output];
					classCountVector1[[classNumber]] = classCountVector1[[classNumber]] + 1,

					count2++;
					classNumber = CIP`Utility`GetPositionOfMaximumValue[output];
					classCountVector2[[classNumber]] = classCountVector2[[classNumber]] + 1
				],
				
				{k, Length[inputs]}
			];
			relativeClassCountVector1 = N[classCountVector1/count1];
			relativeClassCountVector2 = N[classCountVector2/count2];
			distance = (EuclideanDistance[relativeClassCountVector1, relativeClassCountVector] +  EuclideanDistance[relativeClassCountVector2, relativeClassCountVector])/2.0;
			If[distance < minDistance,
				minDistance = distance;
				inputComponentForSplit = i;
			],
			
			{i, Length[inputs[[1]]]}
		];

		Return[SplitDataSet[dataSet, inputComponentForSplit]]    
	];

SplitDataSet[

	(* Splits data set with specified input component. Used the median of the input component for splitting.

	   Returns:
	   splitInfo:        {dataSetSplitInfo1, dataSetSplitInfo2}
	   dataSetSplitInfo: {dataSet, inputComponentForSplit, {min of component, max of component}} *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   NOTE: inputs must be convertable into grayscale images *)
    dataSet_,
    
	(* Input component for split *)
    inputComponentForSplit_?IntegerQ
    
    ] :=
  
	Module[
    
		{
			i,
			newDataSet1,
			newDataSet2,
			ioPair,
			input,
			inputs,
			output,
			median,
			min,
			max
		},

		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];
		median = Median[inputs[[All, inputComponentForSplit]]];
		min = Min[inputs[[All, inputComponentForSplit]]];
		max = Max[inputs[[All, inputComponentForSplit]]];

		newDataSet1 = {};
		newDataSet2 = {};
		Do[
			ioPair = dataSet[[i]];
			input = ioPair[[1]];
			output = ioPair[[2]];
			If[input[[inputComponentForSplit]] < median,
				AppendTo[newDataSet1, ioPair],
				AppendTo[newDataSet2, ioPair]
			],
			
			{i, Length[dataSet]}
		];

		Return[
			{
				{newDataSet1, inputComponentForSplit, {min, median}}, 
				{newDataSet2, inputComponentForSplit, {median, max}} 
			}
		]    
	];

SplitDataSetByMinMeanMedianDifference[

	(* Splits data set with component that has the smallest difference between mean and median splitting.

	   Returns:
	   splitInfo:        {dataSetSplitInfo1, dataSetSplitInfo2}
	   dataSetSplitInfo: {dataSet, inputComponentForSplit, {min of component, max of component}} *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...}
	   NOTE: inputs must be convertable into grayscale images *)
    dataSet_
    
    ] :=
  
	Module[
    
		{
			inputComponentForSplit,
			i,
			k,
			input,
			inputs,
			median,
			medianCounter,
			mean,
			meanCounter,
			minDifference
		},

		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];
		minDifference = Infinity;
		Do[
			median = Median[inputs[[All, i]]];
			mean = Mean[inputs[[All, i]]];
			
			medianCounter = 0;
			meanCounter = 0;
			Do[
				input = inputs[[k]];
				If[input[[i]] < median,
					medianCounter += 1;
				];
				If[input[[i]] < mean,
					meanCounter += 1;
				],
				
				{k, Length[inputs]}
			];
			If[Abs[meanCounter - medianCounter] < minDifference,
				minDifference = Abs[meanCounter - medianCounter];
				inputComponentForSplit = i;
			],
			
			{i, Length[inputs[[1]]]}
		];

		Return[SplitDataSet[dataSet, inputComponentForSplit]]    
	];

TransformDataMatrixByExp[

	(* Calculates exponential of every element of matrix, see code.

	   Returns:
	   Transformed matrix with same structure as dataMatrix *)

	(* dataMatrix: {dataVector1, dataVector2, ...}
	   dataVector: {dataValue1, dataValue2, ...} *)
    dataMatrix_/;MatrixQ[dataMatrix, NumberQ]	    

	] :=
  
	Module[
    
		{},
    
		Return[N[Exp[dataMatrix]]]
	];

TransformDataMatrixByLog[

	(* Calculates natural logarithm of every element of matrix, see code.

	   Returns:
	   Transformed matrix with same structure as dataMatrix *)

	(* dataMatrix: {dataVector1, dataVector2, ...}
	   dataVector: {dataValue1, dataValue2, ...} *)
    dataMatrix_/;MatrixQ[dataMatrix, NumberQ]	    

	] :=
  
	Module[
    
		{},
    
		Return[N[Log[dataMatrix]]]
	];

TransformDataSetToMultipleDataSet[

	(* Transforms data set to multiple data set where each data set has only one output component.

	   Returns:
	   multipleDataSet: (dataSet1, dataSet2, ..., dataSet<NumberOfOutputComponents>) *)

    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ..., outputValue<NumberOfOutputComponents>} *)
    dataSet_
    
	] :=
  
	Module[
    
		{
			i,
			inputs,
			k,
			numberOfIOPairs,
			numberOfOutputComponents,
			outputs
		},
    
		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];
	    outputs = CIP`Utility`GetOutputsOfDataSet[dataSet];
	    numberOfIOPairs = Length[inputs];
	    numberOfOutputComponents = Length[outputs[[1]]];
    	
		Return[
			Table[
				Table[
					{inputs[[k]], {outputs[[k, i]]}},
					
					{k, numberOfIOPairs}
				],
				
				{i, numberOfOutputComponents}
			]
		]
	];
	
TransformLinear[
	
	(* Performs linear transformation of value x of interval [min1, max1] into interval [min2, max2]

	   Returns:
	   Linear transformed value *)

	
	x_?NumberQ, 
	
	min1_?NumberQ, 
	
	max1_?NumberQ, 
	
	min2_?NumberQ, 
	
	max2_?NumberQ
	
	] := max2 - (max2 - min2)/(max1 - min1)*(max1 - x);

TransformXYDataToDataSet[

	(* Transforms (x, y) data to data set.

	   Returns:
	   dataSet = {IOPair1, ..., IOPair<numberOfData>}
	   IOPair = {{x},{y}} *)


	(* {{x1, y1}, {x2, y2}, ...} 
	   numberOfData = Length[xyData] *)      
	xyData_/;MatrixQ[xyData, NumberQ]
      
	] :=
    
    Module[
      
		{i},
      
		Return[
			Table[
				{{xyData[[i, 1]]},{xyData[[i, 2]]}},
								
				{i, Length[xyData]}
			]
		];
	]

TransformXyErrorDataToDataSet[

	(* Transforms (x, y, error) data to data set.

	   Returns:
	   dataSet = {IOPair1, ..., IOPair<numberOfData>}
	   IOPair = {{x},{y}} *)


	(* {{x1, y1, error1}, {x2, y2, error2}, ...} 
	   numberOfData = Length[xyErrorData] *)      
	xyErrorData_/;MatrixQ[xyErrorData, NumberQ]
      
	] :=
    
    Module[
      
		{i},
      
		Return[
			Table[
				{{xyErrorData[[i, 1]]},{xyErrorData[[i, 2]]}},
								
				{i, Length[xyErrorData]}
			]
		];
	]

(* ::Section:: *)
(* End of Package *)

End[]

EndPackage[]
