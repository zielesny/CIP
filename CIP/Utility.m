(* ::Package:: *)

(*
-----------------------------------------------------------------------
Computational Intelligence Packages (CIP): Package Utility
Version 3.0 for Mathematica 11 or higher
-----------------------------------------------------------------------

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
-----------------------------------------------------------------------
*)


(* ::Section:: *)
(* Package and dependencies *)


BeginPackage["CIP`Utility`"]


(* ::Section:: *)
(* Off settings *)


Off[General::"spell1"]
Off[General::shdw]
Off[General::compat]


(* ::Section:: *)
(* Options *)


Options[UtilityOptionsValueDistribution] = 
{
    (* Number of intervals for frequency percentages *)
	UtilityOptionNumberOfIntervals -> 20
}

Options[UtilityOptionsTrainingSetOptimization] = 
{
    (* Number of optimization steps *)
	UtilityOptionOptimizationSteps -> 0,
	
    (* "AllClusterMax", "AllClusterMean", "SingleGlobalMax", "SingleGlobalMean" *)
	UtilityOptionDeviationCalculation -> "SingleGlobalMax",
	
    (* Length of black list *)
	UtilityOptionBlackListLength -> 0
}

Options[UtilityOptionsRandomInitialization] = 
{
	(* "Seed"  : Deterministic random sequence with SeedRandom[1] 
	   "NoSeed": Random random sequence with SeedRandom[] *)
	UtilityOptionRandomInitializationMode -> "Seed"
}

Options[UtilityOptionsIntermediateOutput] = 
{
	(* True : Intermediate output is generated 
	   False: Intermediate output is suppressed *)
	UtilityOptionIsIntermediateOutput -> False
}

Options[UtilityOptionsExclusion] = 
{
	(* Input component exclusion analysis: After each loop over all input components the number of input components specified in 
	   UtilityOptionExclusionsPerStep are excluded, i.e. with UtilityOptionExclusionsPerStep -> {20, 10, 5} 20 input components 
	   are excluded after the first loop, after the second loop 10 input components etc. *)
	UtilityOptionExclusionsPerStep -> {}
}

Options[UtilityOptionsInclusion] = 
{
	(* Input component inclusion analysis: After each loop over all input components the number of input components specified in 
	   UtilityOptionInclusionsPerStep are included, i.e. with UtilityOptionInclusionsPerStep -> {20, 10, 5} 20 input components 
	   are included after the first loop, after the second loop 10 input components etc. *)
	UtilityOptionInclusionsPerStep -> {},
	
	(* Input component inclusion analysis: Start list with indices of included input components. *)
	UtilityOptionInclusionStartList -> {}
}

Options[UtilityOptionsOptimization] = 
{
	(* "BestTestResult", "MinimumDeviation" *)
	UtilityOptionBestOptimization -> "BestTestResult"
}

Options[UtilityOptionsParallelization] =
{
	(* "SequentialCalculation"  : Default sequential calculation
	   "ParallelCalculation"    : Parallelized calculation with chosen number of kernels, see SetNumberOfParallelKernels *)
	UtilityOptionCalculationMode -> "SequentialCalculation"
}


(* ::Section:: *)
(* Declarations *)


CleanMatrix::usage = 
	"CleanMatrix[matrix]"

GetBestRegressOptimization::usage = 
	"GetBestRegressOptimization[trainOptimization, options]"

GetCentroidVectorFromIndexedInputs::usage = 
	"GetCentroidVectorFromIndexedInputs[inputs, indexList]"

GetCentroidVectorFromInputs::usage = 
	"GetCentroidVectorFromInputs[inputs]"

GetClassCountVector::usage = 
	"GetClassCountVector[outputs]"

GetConstantComponentList::usage = 
	"GetConstantComponentList[matrix]"

GetConstantComponentListOfDataSet::usage = 
	"GetConstantComponentListOfDataSet[dataSet]"

GetCorrectClassificationInPercent::usage = 
	"GetCorrectClassificationInPercent[dataSet, pureFunction]"

GetDescendingValuePositions::usage = 
	"GetDescendingValuePositions[list]"

GetDeviationMinMaxIndex::usage = 
	"GetDeviationMinMaxIndex[dataSetInputs, dataSetOutputs, machineOutputs]"

GetDeviationSortedDataSet::usage = 
	"GetDeviationSortedDataSet[dataSetInputs, dataSetOutputs, machineOutputs]"

GetDeviationSortedIndexListOfDataSet::usage = 
	"GetDeviationSortedIndexListOfDataSet[dataSetInputs, dataSetOutputs, machineOutputs]"

GetIndexOfIndexList::usage = 
	"GetIndexOfIndexList[singleIndex, indexLists]"
    	
GetInputsOfDataSet::usage = 
	"GetInputsOfDataSet[dataSet]"

GetMatchList::usage = 
	"GetMatchList[list1, list2]"

GetMaxDeviationIndex::usage = 
	"GetMaxDeviationIndex[dataSet, indexList, pureFunction]"

GetMaxDeviationWithIndex::usage = 
	"GetMaxDeviationWithIndex[dataSet, indexList, pureFunction]"

GetMaxDeviationToCentroidFromIndexedInputs::usage = 
	"GetMaxDeviationToCentroidFromIndexedInputs[inputs, indexList]"

GetMeanAndStandardDeviation::usage =
	"GetMeanAndStandardDeviation[matrix, indexOfComponent]"

GetMeanAndStandardDeviationList::usage =
	"GetMeanAndStandardDeviationList[matrix]"

GetMeanDeviationIndex::usage = 
	"GetMeanDeviationIndex[dataSet, indexList, pureFunction]"

GetMeanDeviationToCentroidFromIndexedInputs::usage = 
	"GetMeanDeviationToCentroidFromIndexedInputs[inputs, indexList]"

GetMeanDeviationToCentroidFromInputs::usage = 
	"GetMeanDeviationToCentroidFromInputs[inputs]"

GetMeanSquaredError::usage = 
	"GetMeanSquaredError[dataSet, pureFunction]"

GetMeanSquaredErrorList::usage = 
	"GetMeanSquaredErrorList[dataSet, pureFunction]"

GetMinList::usage = 
	"GetMinList[matrix]"

GetMinMax::usage = 
	"GetMinMax[matrix, indexOfComponent]"

GetMinMaxList::usage = 
	"GetMinMaxList[matrix]"
	
GetNextHigherEvenIntegerNumber::usage = 
	"GetNextHigherEvenIntegerNumber[number]"

GetNextHigherMultipleOfTen::usage = 
	"GetNextHigherMultipleOfTen[number]"
	
GetOutputsOfDataSet::usage = 
	"GetOutputsOfDataSet[dataSet]"

GetPositionOfMaximumValue::usage = 
	"GetPositionOfMaximumValue[list]"

GetPositionOfMinimumValue::usage = 
	"GetPositionOfMinimumValue[list]"

GetScaledFitnessSumList::usage = 
	"GetScaledFitnessSumList[fitnessList]"

GetValuesDistribution::usage = 
	"GetValuesDistribution[valueList, options]"

HasNonNumberComponent::usage = 
	"HasNonNumberComponent[vector]"

NormalizeVector::usage = 
	"NormalizeVector[vector]"

RemoveComponents::usage = 
	"RemoveComponents[matrix, inputComponentRemovalList]"

RoundTo::usage = 
	"RoundTo[value, numberOfDecimals]"
	
SelectChromosome::usage = 
	"SelectChromosome[population, scaledFitnessSumList]"

SelectNewTrainingAndTestSetIndexLists::usage = 
	"SelectNewTrainingAndTestSetIndexLists[dataSet, trainingSetIndexList, testSetIndexList, blackList, indexLists, pureOutputFunction, options]"

SetNumberOfParallelKernels::usage =
	"SetNumberOfParallelKernels[numberOfKernels]"


(* ::Section:: *)
(* Functions *)


Begin["`Private`"]

CleanMatrix[

	(* Cleans matrix from components that do have non-defined numbers and transforms all components to real values.
	
	   Returns:
	   cleanedMatrix with possibly different structure than matrix *)
    
    
    (* matrix : {vector1, vector2, ...}
       vector : {component1, ..., component<NumberOfComponentsInVectorOfMatrix>}
       All vectors in matrix must have the same length *)
    matrix_ 
    
	] :=
  
	Module[
    
    	{
    		cleanedMatrix,
    		cleanedVector,
    		inputComponentRemovalList,
    		i,
    		k,
    		vector
    	},

		inputComponentRemovalList = {};
		cleanedMatrix = {}; 
		Do[
			vector = matrix[[i]];
			cleanedVector = {};
			Do[
				If[NumberQ[vector[[k]]] && Length[Position[inputComponentRemovalList, k]] == 0,
					
					AppendTo[cleanedVector, N[vector[[k]]]],
					
					If[Length[Position[inputComponentRemovalList, k]] == 0,
						AppendTo[inputComponentRemovalList, k]
					]
				],
				
				{k, Length[vector]}
			];
			AppendTo[cleanedMatrix, cleanedVector],
			
			{i, Length[matrix]}
		];
		    
	    Return[cleanedMatrix]
	];

GetBestRegressOptimization[

	(* Returns best optimization result for regression.

	   Returns: 
	   Best index for regression *)


	(* trainOptimization = {trainingSetRmseList, testSetRmseList, trainingAndTestSetList, <FitType>InfoList}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set}
	   trainingAndTestSetList: List with {training set, test set}
	   trainingAndTestSetList[[i]] refers to optimization step i
	   <FitType>InfoList: List with <FitType>Info
	   <FitType>InfoList[[i]] refers to optimization step i *)
	trainOptimization_,

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			k,
			bestIndex,
			bestTestSetRmse,
			testSetRmseList,
			trainingSetRmseList,
			minimumTestSetRmseValue,
			bestOptimization,
			minimumDeviation,
			deviation
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    bestOptimization = UtilityOptionBestOptimization/.{opts}/.Options[UtilityOptionsOptimization];

		Switch[bestOptimization,

			(* ------------------------------------------------------------------------------- *)
			"BestTestResult",			
			testSetRmseList = trainOptimization[[2]];
			minimumTestSetRmseValue = Infinity;
			Do[
				If[testSetRmseList[[k, 2]] < minimumTestSetRmseValue,
					minimumTestSetRmseValue = testSetRmseList[[k, 2]];
					bestIndex = k
				],
				
				{k, Length[testSetRmseList]}
			],
			
			(* ------------------------------------------------------------------------------- *)			
			"MinimumDeviation",
			trainingSetRmseList = trainOptimization[[1]];
			testSetRmseList = trainOptimization[[2]];
			minimumDeviation = Infinity;
			Do[
				deviation = Abs[testSetRmseList[[k, 2]] - trainingSetRmseList[[k, 2]]];
				If[deviation < minimumDeviation || (deviation == minimumDeviation && testSetRmseList[[k, 2]] < bestTestSetRmse),
					minimumDeviation = deviation;
					bestTestSetRmse = testSetRmseList[[k, 2]];
					bestIndex = k
				],
				
				{k, Length[testSetRmseList]}
			]
		];

		Return[bestIndex]
	];

GetCentroidVectorFromIndexedInputs[

	(* Returns center of mass centroid vector from indexed inputs.

	   Returns :
	   Center of mass centroid vector *)


	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],

	(* Index list for inputs *)
	indexList_/;VectorQ[indexList, IntegerQ]
	
	] := 
	
	Module[
    
	    {
	    	i
		},

		(* Check if cluster contains only 1 vector *)
		If[Length[indexList] == 1,
			Return[inputs[[indexList[[1]]]]]
		];
		
		Return[
			Apply[Plus,
				Table[
					inputs[[indexList[[i]]]],
					
					{i, Length[indexList]}
				]
			]/Length[indexList]
		]
	];

GetCentroidVectorFromInputs[

	(* Returns center of mass centroid vector for inputs.

	   Returns :
	   Center of mass centroid vector *)


	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ]
	
	] := 
	
	Module[
    
	    {},

		(* Check if inputs contains only 1 vector *)
		If[Length[inputs] == 1,
			Return[inputs[[1]]]
		];
		
		Return[
			Apply[Plus, inputs]/Length[inputs]
		]
	];


GetClassCountVector[

	(* Returns vector with class counts.
	
	   Returns:
       Vector with class counts *)
    
    
    (* Outputs of classification data set *)
    outputs_/;VectorQ[outputs, NumberQ]
    
	] :=
  
	Module[
    
    	{
    		classNumber,
    		classCountVector,
    		k,
    		output
    	},

		classCountVector = Table[0, {Length[outputs[[1]]]}];    
		Do[
			output = outputs[[k]];
			classNumber = GetPositionOfMaximumValue[output];
			classCountVector[[classNumber]] = classCountVector[[classNumber]] + 1,
			
			{k, Length[outputs]}
		];
		Return[classCountVector]
	];

GetConstantComponentList[

	(* Returns list with components that all have the same value.
	
	   Returns:
	   constantComponentList: {constant component 1, constant component 2, ...} *)
    
    
    (* matrix : {vector1, vector2, ...}
       vector : {component1, ..., component<NumberOfComponentsInVectorOfMatrix>}
       All vectors in matrix must have the same length *)
    matrix_/;MatrixQ[matrix, NumberQ] 
    
	] :=
  
	Module[
    
    	{
    		constantComponentList,
    		counter,
    		componentList,
    		i,
    		k
    	},

		constantComponentList = {};
		Do[
			componentList = matrix[[All, i]];
			counter = 0;
			Do[
				If[componentList[[k]] == componentList[[k + 1]],
					counter += 1,
					Break[]
				],
				
				{k, Length[componentList] - 1}
			];
			If[counter == Length[componentList] - 1,
				AppendTo[constantComponentList, i]
			],
			
			{i, Length[matrix[[1]]]}
		];
		    
	    Return[constantComponentList]
	];

GetConstantComponentListOfDataSet[

	(* Returns list with components that all have the same value.
	
	   Returns:
	   constantComponentList: {constant component 1, constant component 2, ...} *)
    
    
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output: {outputComponent1, outputComponent2, ...} *)
	dataSet_ 
    
	] :=
  
	Module[
    
    	{},

	    Return[
	    	GetConstantComponentList[GetInputsOfDataSet[dataSet]]
		]
	];

GetCorrectClassificationInPercent[

	(* Returns correct classification in percent.

	   Returns: 
	   Correct classification in percent *)

   
	(* classificationDataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} 
	   outputValue: 0/1
	   Data set must be a classification data set, i.e. the output components must 0/1 code a class,
	   i.e. class 4 of 5 must be coded {0, 0, 0, 1, 0} *)
    classificationDataSet_,
    
	(* Pure function of inputs
	   inputs = {input1, input2, ...} 
	   input: {inputComponent1, inputComponent2, ...} *)
	pureFunction_
    
	] :=
  
	Module[
    
		{
			classNumber,
			classNumberDesired,
			classNumbers,
			correctPredictions,
			i,
			inputs,
			outputs
		},

    	inputs = GetInputsOfDataSet[classificationDataSet];
    	outputs = GetOutputsOfDataSet[classificationDataSet];
    	classNumbers = pureFunction[inputs];
    	
	    (* Analyze data set *)
	    correctPredictions = 0;
	    Do[
	    	classNumber = classNumbers[[i]];
			classNumberDesired = GetPositionOfMaximumValue[outputs[[i]]];
			If[classNumber == classNumberDesired,
				(* Correct classification *)
				correctPredictions++
			],
      
			{i, Length[classificationDataSet]}
		];

		Return[
			RoundTo[correctPredictions/Length[classificationDataSet]*100., 1]
		]
	];

GetDescendingValuePositions[

	(* Returns descending value positions, i.e. list = {2, 4, 1} returns {2, 1, 3}.
	
	   Returns:
       Descending value positions *)
    
    
    (* List *)
    list_/;VectorQ[list, NumberQ]
    
	] :=
  
	Module[
    
    	{
    		i,
    		sortList
    	},
    
    	If[Length[list] == 0, Return[{}]];
    	
    	sortList = 
    		Table[
    			{list[[i]], i},
    			
    			{i, Length[list]}
    		];
    	
	    Return[
	        Sort[sortList, #1[[1]] > #2[[1]] &][[All, 2]]
		];
	];

GetDeviationMinMaxIndex[

	(* Returns minimum and maximum index that corresponds to minimum and maximum deviation, see code.

	   Returns:
	   {minIndex, maxIndex} *)


    (* {input1, input2, ...}
	   input: {inputComponent1, inputComponent2, ...} *)
    dataSetInputs_?MatrixQ,

    (* {output1, output2, ...}
	   output: {outputComponent1, outputComponent2, ...} 
	   output[[i]] corresponds to input[[i]] *)
    dataSetOutputs_?MatrixQ,

    (* {machineOutput1, machineOutput2, ...}
	   machineOutput: {machineOutputComponent1, machineOutputComponent2, ...} 
	   machineOutput[[i]] corresponds to input[[i]] *)
    machineOutputs_?MatrixQ
    
    ] :=
  
	Module[
    
		{
			deviation,
			minDeviation,
			maxDeviation,
			minIndex,
			maxIndex,
			i
		},

		minDeviation = Norm[machineOutputs[[1]] - dataSetOutputs[[1]]];
		maxDeviation = minDeviation;
		minIndex = 1;
		maxIndex = 1;
		If[Length[dataSetInputs] > 1,
			Do[
				deviation = Norm[machineOutputs[[i]] - dataSetOutputs[[i]]];
				If[deviation < minDeviation,
					minDeviation = deviation;
					minIndex = i
				];
				If[deviation > maxDeviation,
					maxDeviation = deviation;
					maxIndex = i
				],
				
				{i, 2, Length[dataSetInputs]}
			]
		];
		
		Return[{minIndex, maxIndex}]    
	];

GetDeviationSortedDataSet[

	(* Returns data set sorted ascending according to deviation between dataSetOutputs and machineOutputs.

	   Returns:
	   dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output: {outputComponent1, outputComponent2, ...} *)

    
    (* {input1, input2, ...}
	   input: {inputComponent1, inputComponent2, ...} *)
    dataSetInputs_?MatrixQ,

    (* {output1, output2, ...}
	   output: {outputComponent1, outputComponent2, ...} 
	   output[[i]] corresponds to input[[i]] *)
    dataSetOutputs_?MatrixQ,

    (* {machineOutput1, machineOutput2, ...}
	   machineOutput: {machineOutputComponent1, machineOutputComponent2, ...} 
	   machineOutput[[i]] corresponds to input[[i]] *)
    machineOutputs_?MatrixQ
    
    ] :=
  
	Module[
    
		{
			sortedData,
			i
		},
    
		sortedData = Sort[
			Table[
				{Norm[machineOutputs[[i]] - dataSetOutputs[[i]]], {dataSetInputs[[i]], dataSetOutputs[[i]]}},
				
				{i, Length[dataSetInputs]}
			]
		];
		Return[sortedData[[All, 2]]]
	];

GetDeviationSortedIndexListOfDataSet[

	(* Returns indexList corresponding to data set IOPairs according to deviations between dataSetOutputs and machineOutputs sorted ascending.

	   Returns:
	   indexList: {index1, index2, ..., index<Length[dataSetInputs]>}
	   index[[i]] corresponds to dataSetInputs[[i]] *)

    
    (* {input1, input2, ...}
	   input: {inputComponent1, inputComponent2, ...} *)
    dataSetInputs_?MatrixQ,

    (* {output1, output2, ...}
	   output: {outputComponent1, outputComponent2, ...} 
	   output[[i]] corresponds to input[[i]] *)
    dataSetOutputs_?MatrixQ,

    (* {machineOutput1, machineOutput2, ...}
	   machineOutput: {machineOutputComponent1, machineOutputComponent2, ...} 
	   machineOutput[[i]] corresponds to input[[i]] *)
    machineOutputs_?MatrixQ
    
    ] :=
  
	Module[
    
		{
			sortedData,
			i
		},
    
		sortedData = Sort[
			Table[
				{Norm[machineOutputs[[i]] - dataSetOutputs[[i]]], i},
				
				{i, Length[dataSetInputs]}
			]
		];
		Return[sortedData[[All, 2]]]
	];

GetIndexOfIndexList[

	(* Returns index of index list in indexLists that contains singleIndex.

	   Returns:
	   Index of index list in indexLists that contains singleIndex. *)


	singleIndex_?IntegerQ,
	
    (* indexLists: {indexList1, indexList2, ..., indexList<numberOfIndexLists>} 
       indexList: {index1, index2, ...} *)
    indexLists_
    
    ] :=
  
	Module[
    
		{
			i
		},
    
		Do[
			If[MemberQ[indexLists[[i]], singleIndex],
				Return[i]
			],
			
			{i, Length[indexLists]}
        ]
	];

GetInputsOfDataSet[

	(* Returns inputs of data set

	   Returns:
	   {input1, input2, ...} *)

    
    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output: {outputComponent1, outputComponent2, ...} *)
    dataSet_ 
    
    ] :=
  
	Module[
    
		{},
    
		Return[
			dataSet[[All, 1]]
        ]
	];

GetMatchList[

	(* Returns list with all elements of list2 that are contained in list1.

	   Returns:
	   List with all elements of list2 that are contained in list1. *)


    list1_?VectorQ,

    list2_?VectorQ
    
    ] :=
  
	Module[
    
		{
			i,
			result
		},

		result = {};
		Do[
			If[MemberQ[list1, list2[[i]]], 
				AppendTo[result, list2[[i]]]
			], 
			
			{i, Length[list2]}
		];
		Return[result]    
	];

GetMaxDeviationIndex[

	(* Returns index with maximum deviation, see code.

	   Returns:
	   Index with maximum deviation *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_ ,

	(* {index1, index2, ...} *)
	indexList_/;VectorQ[indexList, IntegerQ],

	(* Pure function of single input
	   input: {inputComponent1, inputComponent2, ...} *)
	pureFunction_
    
    ] :=
  
	Module[
    
		{
			maxDeviation,
			maxDeviationIndex,
			i,
			ioPair,
			input,
			output,
			machineOutput,
			deviation
		},

		If[Length[indexList] == 1,
			
			maxDeviationIndex = indexList[[1]],
			
			maxDeviation = -1.0;
			Do[
				ioPair = dataSet[[ indexList[[i]] ]];
				input = ioPair[[1]];
				output = ioPair[[2]];
				machineOutput = pureFunction[input];
				deviation = Norm[machineOutput - output];
				If[deviation > maxDeviation,
					maxDeviation = deviation;
					maxDeviationIndex = indexList[[i]]
				],
				
				{i, Length[indexList]}
			]			
		];

		Return[maxDeviationIndex]
	];

GetMaxDeviationWithIndex[

	(* Returns maximum deviation with corresponding index in index list, see code.

	   Returns:
	   {maximum deviation, corresponding index} *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_ ,

	(* {index1, index2, ...} *)
	indexList_/;VectorQ[indexList, IntegerQ],

	(* Pure function of single input
	   input: {inputComponent1, inputComponent2, ...} *)
	pureFunction_
    
    ] :=
  
	Module[
    
		{
			maxDeviation,
			maxDeviationIndex,
			i,
			ioPair,
			input,
			output,
			machineOutput,
			deviation
		},

		maxDeviation = -1.0;
		Do[
			ioPair = dataSet[[ indexList[[i]] ]];
			input = ioPair[[1]];
			output = ioPair[[2]];
			machineOutput = pureFunction[input];
			deviation = Norm[machineOutput - output];
			If[deviation > maxDeviation,
				maxDeviation = deviation;
				maxDeviationIndex = indexList[[i]]
			],
			
			{i, Length[indexList]}
		];

		Return[{maxDeviation, maxDeviationIndex}]
	];

GetMaxDeviationToCentroidFromIndexedInputs[

	(* Returns maximum deviation of indexed inputs from their center of mass centroid vector.

	   Returns :
	   Maximum deviation of inputs from their center of mass centroid vector *)


	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],

	(* Index list for inputs *)
	indexList_/;VectorQ[indexList, IntegerQ]
	
	] := 
	
	Module[
    
	    {
	    	centroidVector,
	    	i
		},

		(* Check if inputs contains only 1 vector *)
		If[Length[indexList] <= 1,
			Return[0.0]
		];

		centroidVector = GetCentroidVectorFromIndexedInputs[inputs, indexList];
		Return[
			Max[
				Table[
					EuclideanDistance[centroidVector, inputs[[indexList[[i]]]]],
			
					{i, Length[indexList]}
				]
			]
		]
	];

GetMeanAndStandardDeviation[

	(* Returns mean and standard deviation for specified component of vectors of matrix.
	
	   Returns:
	   {mean, standard deviation} *)
    
    
    (* matrix : {vector1, vector2, ...}
       vector : {component1, ..., component<NumberOfComponentsInVectorOfMatrix>}
       All vectors in matrix must have the same length *)
    matrix_/;MatrixQ[matrix, NumberQ],
    
    (* Index of component of vectors in matrix *)
    indexOfComponent_?IntegerQ
    
	] :=
  
	Module[
    
    	{},
    
	    Return[
	    	{
	    		Mean[matrix[[All, indexOfComponent]]], 
	    		StandardDeviation[matrix[[All, indexOfComponent]]]
    		}
	    ]
	];

GetMeanAndStandardDeviationList[

	(* Returns list with mean and standard deviation for each component of vectors of matrix.
       
       Returns:
       {MeanAndStandardDeviation1, ..., MeanAndStandardDeviation<NumberOfComponentsInVectorOfMatrix>}
	   MeanAndStandardDeviation: {mean, standard deviation}
	   MeanAndStandardDeviation[[i]] corresponds to component [[i]] of vectors of matrix *)
    
    
    (* matrix : {vector1, vector2, ...}
       vector : {component1, ..., component<NumberOfComponentsInVectorOfMatrix>}
       All vectors in matrix must have the same length *)
    matrix_/;MatrixQ[matrix, NumberQ]
    
	] :=
  
	Module[
		
    	{i},
    
	    Return[
	        Table[
	        	{
		    		Mean[matrix[[All, i]]], 
		    		StandardDeviation[matrix[[All, i]]]
    			}, 
	        		
	        	{i, Length[matrix[[1]]]}
	        ]
		]
	];

GetMeanDeviationIndex[

	(* Returns index with mean deviation, see code.

	   Returns:
	   Index with mean deviation *)


	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_ ,

	(* {index1, index2, ...} *)
	indexList_/;VectorQ[indexList, IntegerQ],

	(* Pure function of single input
	   input: {inputComponent1, inputComponent2, ...} *)
	pureFunction_
    
    ] :=
  
	Module[
    
		{
			distance,
			minDistance,
			deviationList,
			minDeviation,
			maxDeviation,
			meanDeviation,
			meanDeviationIndex,
			i,
			ioPair,
			input,
			output,
			machineOutput,
			deviation
		},

		If[Length[indexList] == 1,
			
			meanDeviationIndex = indexList[[1]],
			
			deviationList = {};
			minDeviation = Infinity;
			maxDeviation = -Infinity;
			Do[
				ioPair = dataSet[[ indexList[[i]] ]];
				input = ioPair[[1]];
				output = ioPair[[2]];
				machineOutput = pureFunction[input];
				deviation = Norm[machineOutput - output];
				AppendTo[deviationList, {deviation, indexList[[i]]}];
				minDeviation = Min[deviation, minDeviation];
				maxDeviation = Max[deviation, maxDeviation],
				
				{i, Length[indexList]}
			];
			meanDeviation = (maxDeviation + minDeviation)/2.0;
			minDistance = Infinity;
			Do[
				distance = Abs[meanDeviation - deviationList[[i, 1]]];
				If[distance < minDistance,
					minDistance = distance;
					meanDeviationIndex = deviationList[[i, 2]]
				],
				
				{i, Length[deviationList]}
			]
		];

		Return[meanDeviationIndex]
	];

GetMeanDeviationToCentroidFromIndexedInputs[

	(* Returns mean deviation of indexed inputs from their center of mass centroid vector.

	   Returns :
	   Mean deviation of inputs from their center of mass centroid vector *)


	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],

	(* Index list for inputs *)
	indexList_/;VectorQ[indexList, IntegerQ]
	
	] := 
	
	Module[
    
	    {
	    	centroidVector,
	    	i
		},

		(* Check if inputs contains only 1 vector *)
		If[Length[indexList] <= 1,
			Return[0.0]
		];

		centroidVector = GetCentroidVectorFromIndexedInputs[inputs, indexList];
		Return[
			Mean[
				Table[
					EuclideanDistance[centroidVector, inputs[[indexList[[i]]]]],
			
					{i, Length[indexList]}
				]
			]
		]
	];

GetMeanDeviationToCentroidFromInputs[

	(* Returns mean deviation of inputs from their center of mass centroid vector.

	   Returns :
	   Mean deviation of inputs from their center of mass centroid vector *)


	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ]
	
	] := 
	
	Module[
    
	    {
	    	centroidVector,
	    	i
		},

		(* Check if inputs contains only 1 vector *)
		If[Length[inputs] <= 1,
			Return[0.0]
		];

		centroidVector = GetCentroidVectorFromInputs[inputs];
		Return[
			Mean[
				Table[
					EuclideanDistance[centroidVector, inputs[[i]]],
			
					{i, Length[inputs]}
				]
			]
		]
	];

GetMeanSquaredError[

	(* Returns mean squared error of data set.
	
	   Returns:
       Mean squared error of data set *)
    
    
    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...} *)
    dataSet_,

	(* Pure function of inputs
	   inputs = {input1, input2, ...} 
	   input: {inputComponent1, inputComponent2, ...} *)
	pureFunction_    
    
	] :=
  
	Module[
    
		{
			errors,
			inputs,
			machineOutputs,
			outputs,
			lengthOfSingleOutput
		},

		inputs = GetInputsOfDataSet[dataSet];
		outputs = GetOutputsOfDataSet[dataSet];
		
		lengthOfSingleOutput = Length[outputs[[1]]];
		
		machineOutputs = pureFunction[inputs];
        errors = machineOutputs - outputs;
        Return[
        	Apply[Plus, errors^2, {0, 1}]/(Length[dataSet]*lengthOfSingleOutput)
        ]
	];

GetMeanSquaredErrorList[

	(* Returns list of mean squared errors for every output component of data set.
	
	   Returns:
       List of mean squared errors for every output component of data set *)
    
    
    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...} *)
    dataSet_,

	(* Pure function of inputs
	   inputs = {input1, input2, ...} 
	   input: {inputComponent1, inputComponent2, ...} *)
	pureFunction_    
    
	] :=
  
	Module[
    
		{
			i,
			errors,
			singleOutputErrors,
			inputs,
			machineOutputs,
			outputs,
			lengthOfSingleOutput,
			numberOfIoPairs
		},

		inputs = GetInputsOfDataSet[dataSet];
		outputs = GetOutputsOfDataSet[dataSet];
		numberOfIoPairs = Length[dataSet];
		lengthOfSingleOutput = Length[outputs[[1]]];
		
		machineOutputs = pureFunction[inputs];
        errors = machineOutputs - outputs;
        Return[
        	Table[
        		singleOutputErrors = errors[[All, i]];
        		Apply[Plus, singleOutputErrors^2]/(numberOfIoPairs),
        		
        		{i, lengthOfSingleOutput}
        	]
        ]
	];

GetMinList[

	(* Returns min list for each component of vectors of matrix.
	
	   Returns:
       {MinComponent1, ..., MinComponent<NumberOfComponentsInVectorOfMatrix>}
	   MinComponent contains the minimum value of component
	   MinComponent[[i]] corresponds to component [[i]] of vectors of matrix *)
    
    
    (* matrix : {vector1, vector2, ...}
       vector : {component1, ..., component<NumberOfComponentsInVectorOfMatrix>}
       All vectors in matrix must have the same length *)
    matrix_/;MatrixQ[matrix, NumberQ] 
    
	] :=
  
	Module[
    
    	{i},
    
	    (* Determine min value of each component. Loop over all components for min values. *)
	    Return[
	        Table[
	        	Min[matrix[[All, i]]], 
	        		
	        	{i, Length[matrix[[1]]]}
	        ]
		]
	];

GetMinMax[

	(* Returns min-max for specified component of vectors of matrix.
	
	   Returns:
	   {MinComponent, MaxComponent} *)
    
    
    (* matrix : {vector1, vector2, ...}
       vector : {component1, ..., component<NumberOfComponentsInVectorOfMatrix>}
       All vectors in matrix must have the same length *)
    matrix_/;MatrixQ[matrix, NumberQ],
    
    (* Index of component of vectors in matrix *)
    indexOfComponent_?IntegerQ
    
	] :=
  
	Module[
    
    	{},
    
	    Return[{Min[matrix[[All, indexOfComponent]]], Max[matrix[[All, indexOfComponent]]]}]
	];

GetMinMaxList[

	(* Returns min-max list for each component of vectors of matrix.
	
	   Returns:
       {MinMaxComponent1, ..., MinMaxComponent<NumberOfComponentsInVectorOfMatrix>}
	   MinMaxComponent contains the minimum and maximum value of component
	   MinMaxComponent : {MinComponent, MaxComponent}
	   MinMaxComponent[[i]] corresponds to component [[i]] of vectors of matrix *)
    
    
    (* matrix : {vector1, vector2, ...}
       vector : {component1, ..., component<NumberOfComponentsInVectorOfMatrix>}
       All vectors in matrix must have the same length *)
    matrix_/;MatrixQ[matrix, NumberQ] 
    
	] :=
  
	Module[
    
    	{i},
    
	    (* Determine min-max pairs of each component. Loop over all components for min-max values. *)
	    Return[
	        Table[
	        	{Min[matrix[[All, i]]], Max[matrix[[All, i]]]}, 
	        		
	        	{i, Length[matrix[[1]]]}
	        ]
		]
	];

GetNextHigherEvenIntegerNumber[

	(* Returns next higher even integer number *)
	
	
	(* Integer number *)
	number_?IntegerQ
	
	] :=

	Module[
    
    	{},

	    If[!EvenQ[number], 
	    	Return[number + 1], 
	    	Return[number]
	    ]
	];

GetNextHigherMultipleOfTen[

	(* Returns next higher multiple of 10, e.g. 3 yields 10, 17 yields 100, 335 yield 1000 etc. 
	   10 is the minimum number to be returned. *)
	
	
	(* Number *)
	number_?NumberQ
	
	] :=

	Module[
    
    	{
    		multipleOfTen
    	},

		multipleOfTen = 10;
		While[multipleOfTen <= number,
			multipleOfTen *= 10;			
		];
		Return[multipleOfTen];		
	];

GetOutputsOfDataSet[

	(* Returns outputs of data set

	   Returns:
	   {output1, output2, ...} *)

      
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output: {outputComponent1, outputComponent2, ...} *)
	dataSet_ 
      
	] :=
    
	Module[
      
		{},
      
		Return[
			dataSet[[All, 2]]
		];
	];

GetPositionOfMaximumValue[

	(* Returns position of maximum value in specified list.
	
	   Returns:
       Position of maximum value in list or 0 if list is empty *)
    
    
    (* list *)
    list_/;VectorQ[list, NumberQ]
    
	] :=
  
	Module[
    
    	{},
    
    	If[Length[list] == 0, Return[0]];
    
	    Return[
	        First[First[Position[list, Max[list]]]]
		];
	];

GetPositionOfMinimumValue[

	(* Returns position of minimum value in specified list.
	
	   Returns:
       Position of minimum value in list or 0 if list is empty *)
    
    
    (* list *)
    list_/;VectorQ[list, NumberQ]
    
	] :=
  
	Module[
    
    	{},
    
    	If[Length[list] == 0, Return[0]];
    
	    Return[
	        First[First[Position[list, Min[list]]]]
		];
	];

GetScaledFitnessSumList[
    
	(* Calculates the scaled fitness sums for a fitness list by summing up the fitnesses of the individual chromosomes (scaling : Sum of all fitnesses = 1)

	   Returns:
	   Vector/list with scaled fitness sums *)


    fitnessList_/;VectorQ[fitnessList, NumberQ]
    
    ] :=
  
	Module[
    
	    {
	    	scaleFactor,
	    	sumList
	    },
    
		sumList = FoldList[Plus, First[fitnessList], Rest[fitnessList]];
		scaleFactor = 1./Last[sumList];
		Return[sumList*scaleFactor]
	];

GetValuesDistribution[

	(* Returns statistics for specified values.
	
	   Returns:
	   valuesStatistics: {numberOfIntervals, intervalPoints, statistics}
	   numberOfIntervals: Number of intervals
	   intervalPoints: {interval1, interval2, ...}
	   interval: {middle position, frequency in percent of interval}
	   statistics: {min, max, mean, median, standard deviation} of specified values *)


	(* valueList : {value1, value2, ...} *)
    valueList_/;VectorQ[valueList, NumberQ],

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	frequencyList,
	    	i,
	    	index,
	    	intervalLength,
	    	intervalMiddlePositionList,
	    	intervalPoints,
	    	max,
	    	mean,
	    	median,
	    	min,
	    	numberOfInputs,
	    	numberOfIntervals,
	    	standardDeviation
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    numberOfIntervals = UtilityOptionNumberOfIntervals/.{opts}/.Options[UtilityOptionsValueDistribution];

		numberOfInputs = Length[valueList];
		
		min = Min[valueList];
		max = Max[valueList];
		mean = Mean[valueList];
		median = Median[valueList];
		standardDeviation = StandardDeviation[valueList];

		intervalLength = (max - min)/numberOfIntervals;
		frequencyList = Table[0.0, {numberOfIntervals}];
		Do[
			index = Floor[(valueList[[i]] - min)/intervalLength] + 1;
			If[index > numberOfIntervals, index = numberOfIntervals];
			frequencyList[[index]] = frequencyList[[index]] + 1.0,
			
			{i, numberOfInputs}
		];		
		intervalMiddlePositionList = Table[min + intervalLength*(i - 0.5), {i, numberOfIntervals}];
		intervalPoints = Table[{intervalMiddlePositionList[[i]], frequencyList[[i]]/numberOfInputs*100.0}, {i, numberOfIntervals}];
					
		Return[
			{
				numberOfIntervals,
				intervalPoints,
				{min, max, mean, median, standardDeviation}
			}
		]
	];

HasNonNumberComponent[

	(* Returns if vector has a component which is not a number.
	
	   Returns:
	   True: Vector has a component which is not a number, false: Otherwise *)
    
    
    (* vector : {component1, component2, ...} *)
    vector_ 
    
	] :=
  
	Module[
    
    	{
    		i,
    		isNonNumber
    	},

		isNonNumber = False;
		Do[
			If[!NumberQ[vector[[i]]],
				isNonNumber = True;
				Break[]
			],
			
			{i, Length[vector]}
		];
		    
	    Return[isNonNumber]
	];
    
NormalizeVector[

	(* Returns normalized vector *)
	
	
	vector_?VectorQ 
	
	] := vector/Norm[vector];

RemoveComponents[

	(* Removes specified components of vectors of matrix

	   Returns:
	   Matrix with removed components *)

    
    (* matrix : {vector1, vector2, ...}
       vector : {component1, ..., component<NumberOfComponentsInVectorOfMatrix>}
       All vectors in matrix must have the same length *)
    matrix_, 
    
    (* {first input component to be removed, second input component to be removed, ...} *)
    componentRemovalList_/;VectorQ[componentRemovalList, IntegerQ]
    
	] :=
  
	Module[
    
		{
			i,
			k,
			vector,
			newVector
		},
    
		Return[
			Table[
				vector = matrix[[i]];
				newVector = {};
				Do[
					If[Length[Position[componentRemovalList, k]] == 0, 
						newVector = {newVector, vector[[k]]}
					],

					{k, Length[vector]}
				];
				Flatten[newVector],

				{i, Length[matrix]}
			]
		];
	];

RoundTo[

	(* Rounds value to specified number of decimals. *)


	value_?NumberQ,
	
	numberOfDecimals_?IntegerQ
	
	] := 

    Module[
      
		{factor},
      
      	If[numberOfDecimals < 0, 
      		
      		Return[value],
      		
	      	factor = 10.0^numberOfDecimals;
			Return[Round[N[value]*factor]/factor]
      	]
	];

SelectChromosome[

	(* Returns a chromosome of the specified population with a probability according to the corresponding scaled fitness sum (roulette-wheel selection)
	   NOTE: Use SeedRandom[] in superior method to get deterministic random selection

	   Returns:
	   Chromosome of the specified population *)

      
	(* {chromosome1, chromosome2, ...} *)
	population_,

	(* scaledFitnessSumList from GetScaledFitnessSumList *)
	scaledFitnessSumList_/;VectorQ[scaledFitnessSumList, NumberQ]
      
	] :=
    
    Module[
      
		{
			chromosome,
			indexOfChromosome,
			randomNumber
		},
      
		(* NOTE: Use SeedRandom[] in superior method to get deterministic random selection *)
		randomNumber = RandomReal[];
		
		chromosome = First[Select[scaledFitnessSumList, # >= randomNumber &, 1]];
		indexOfChromosome = First[Flatten[Position[scaledFitnessSumList, chromosome]]];
		Return[population[[indexOfChromosome]] ]
	];

SelectNewTrainingAndTestSetIndexLists[

	(* Selects new training and test set.

	   Returns:
	   {newTrainingSetIndexList, newTestSetIndexList, newBlackList} *)

      
	(* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} *)
    dataSet_,

	(* Index list for training set *)
	trainingSetIndexList_/;VectorQ[trainingSetIndexList, IntegerQ],

	(* Index list for test set *)
	testSetIndexList_/;VectorQ[testSetIndexList, IntegerQ],

	(* Black list for already used indices *)
	blackList_,	

	(* List of index lists for all clusters *)
	indexLists_,

	(* output = pureOutputFunction[input] *)
	pureOutputFunction_,
	
	(* Options *)
	opts___
      
	] :=
    
    Module[
      
		{
			deviationCalculationMethod,
			blackListLength,

			k,
			i,
			indexList,
			deviationIndex,
			maxDeviation,
			maxDeviationIndex,
			maxDeviationClusterIndex,
			deviationInfo,
			deviation,
			testIndex,
			newTrainingSetIndexList,
			newTestSetIndexList,
			newBlackList,
			meanDeviationIndex,
			testSetClusterMemberIndexList,
			memberOfTrainingSetIndex
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Training set optimization options *)
	    deviationCalculationMethod = UtilityOptionDeviationCalculation/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
	    blackListLength = UtilityOptionBlackListLength/.{opts}/.Options[UtilityOptionsTrainingSetOptimization];
		
		Switch[deviationCalculationMethod,

			(* ------------------------------------------------------------------------------- *)			
			"AllClusterMax",

			newTrainingSetIndexList = {};
			newTestSetIndexList = {};
			Do[
				indexList = indexLists[[k]];
				If[Length[indexList] == 1,
					
					AppendTo[newTrainingSetIndexList, indexList[[1]]],
					
					deviationIndex = GetMaxDeviationIndex[dataSet, indexList, pureOutputFunction];
					AppendTo[newTrainingSetIndexList, deviationIndex];
					AppendTo[newTestSetIndexList, Delete[indexList, First[Flatten[Position[indexList, deviationIndex]]]]]
				],
				
				{k, Length[indexLists]}
			];
			newTestSetIndexList = Flatten[newTestSetIndexList],
			
			(* ------------------------------------------------------------------------------- *)			
			"AllClusterMean",

			newTrainingSetIndexList = {};
			newTestSetIndexList = {};
			Do[
				indexList = indexLists[[k]];
				If[Length[indexList] == 1,
					
					AppendTo[newTrainingSetIndexList, indexList[[1]]],
					
					deviationIndex = GetMeanDeviationIndex[dataSet, indexList, pureOutputFunction];
					AppendTo[newTrainingSetIndexList, deviationIndex];
					AppendTo[newTestSetIndexList, Delete[indexList, First[Flatten[Position[indexList, deviationIndex]]]]]
				],
				
				{k, Length[indexLists]}
			];
			newTestSetIndexList = Flatten[newTestSetIndexList],

			(* ------------------------------------------------------------------------------- *)			
			"SingleGlobalMax",

			newTrainingSetIndexList = trainingSetIndexList;
			newTestSetIndexList = testSetIndexList;
			maxDeviation = -1.0;
			maxDeviationClusterIndex = 0;
			maxDeviationIndex = -1;
			If[blackListLength > 0,
				
				If[Length[blackList] > blackListLength,
					
					newBlackList = {},
					
					newBlackList = blackList
				],
				
				newBlackList = {}
			];
			Do[
				indexList = indexLists[[k]];
				If[Length[indexList] > 1,
					deviationInfo = GetMaxDeviationWithIndex[dataSet, indexList, pureOutputFunction];
					deviation = deviationInfo[[1]];
					deviationIndex = deviationInfo[[2]];
					If[deviation > maxDeviation && FreeQ[newTrainingSetIndexList, deviationIndex] && FreeQ[newBlackList, deviationIndex],
						maxDeviation = deviation;
						maxDeviationClusterIndex = k;
						maxDeviationIndex = deviationIndex
					]
				],
				
				{k, Length[indexLists]}
			];
			If[maxDeviationClusterIndex > 0,
				indexList = indexLists[[maxDeviationClusterIndex]];
				Do[
					testIndex = indexList[[k]];
					If[MemberQ[newTrainingSetIndexList, testIndex],
						newTrainingSetIndexList = Delete[newTrainingSetIndexList, First[Flatten[Position[newTrainingSetIndexList, testIndex]]]];
						AppendTo[newTrainingSetIndexList, maxDeviationIndex];
						newTestSetIndexList = Delete[newTestSetIndexList, First[Flatten[Position[newTestSetIndexList, maxDeviationIndex]]]];
						AppendTo[newTestSetIndexList, testIndex];
						newBlackList = Append[newBlackList, maxDeviationIndex];
						Break[]
					],
					
					{k, Length[indexList]}
				];
			],
			
			(* ------------------------------------------------------------------------------- *)			
			"SingleGlobalMean",

			newTrainingSetIndexList = trainingSetIndexList;
			newTestSetIndexList = testSetIndexList;
			maxDeviationClusterIndex = 0;
			maxDeviation = -1.0;
			If[blackListLength > 0,
				
				If[Length[blackList] > blackListLength,
					
					newBlackList = {},
					
					newBlackList = blackList
				],
				
				newBlackList = {}
			];
			Do[
				indexList = indexLists[[k]];
				If[Length[indexList] > 1,
					deviationInfo = GetMaxDeviationWithIndex[dataSet, indexList, pureOutputFunction];
					deviation = deviationInfo[[1]];
					deviationIndex = deviationInfo[[2]];
					If[deviation > maxDeviation && FreeQ[newTrainingSetIndexList, deviationIndex] && FreeQ[newBlackList, deviationIndex],
						testSetClusterMemberIndexList = {};
						Do[
							testIndex = indexList[[i]];
							If[MemberQ[newTrainingSetIndexList, testIndex],
								
								memberOfTrainingSetIndex = testIndex,
								
								AppendTo[testSetClusterMemberIndexList, testIndex]
							],
							
							{i, Length[indexList]}
						];
						meanDeviationIndex = GetMeanDeviationIndex[dataSet, testSetClusterMemberIndexList, pureOutputFunction];
						If[FreeQ[newBlackList, meanDeviationIndex],
							maxDeviation = deviation;
							maxDeviationClusterIndex = k
						]
					]
				],
				
				{k, Length[indexLists]}
			];
			If[maxDeviationClusterIndex > 0,
				indexList = indexLists[[maxDeviationClusterIndex]];
				testSetClusterMemberIndexList = {};
				Do[
					testIndex = indexList[[k]];
					If[MemberQ[newTrainingSetIndexList, testIndex],
						
						memberOfTrainingSetIndex = testIndex,
						
						AppendTo[testSetClusterMemberIndexList, testIndex]
					],
					
					{k, Length[indexList]}
				];
				meanDeviationIndex = GetMeanDeviationIndex[dataSet, testSetClusterMemberIndexList, pureOutputFunction];
				newTrainingSetIndexList = Delete[newTrainingSetIndexList, First[Flatten[Position[newTrainingSetIndexList, memberOfTrainingSetIndex]]]];
				AppendTo[newTrainingSetIndexList, meanDeviationIndex];
				newTestSetIndexList = Delete[newTestSetIndexList, First[Flatten[Position[newTestSetIndexList, meanDeviationIndex]]]];
				AppendTo[newTestSetIndexList, memberOfTrainingSetIndex];
				newBlackList = Append[newBlackList, meanDeviationIndex]
			]
		];

		Return[{newTrainingSetIndexList, newTestSetIndexList, newBlackList}]
	];

SetNumberOfParallelKernels[

	(* Set number of kernels for parallel calculation.
	   NOTE: Do NOT change the number of kernels during program/script execution!

	   Returns:
	   Number of launched kernels *)	

    (* Number of kernels to be launched  
       0: The number of kernels is defined by $ProcessorCount which gives the number of processor cores available on the computer system on which Mathematica is being run. *)      
	numberOfKernels_?IntegerQ

	] :=
	
	 Module[
      
		{
			currentKernelCount,
			newKernelCount
		},
		
    	currentKernelCount = Length[Kernels[]];
    	
		If[numberOfKernels == 0,
			 
			newKernelCount = $ProcessorCount,
			
			newKernelCount = numberOfKernels
		];
		
		(* If the current number of kernels is different from the choosen number, they will be closed and launched again with the new count *)
		If[currentKernelCount != newKernelCount,
			CloseKernels[];
			LaunchKernels[newKernelCount]
		];
		
		Return[newKernelCount]
	 ];



(* ::Section:: *)
(* End of Package *)


End[]

EndPackage[]
