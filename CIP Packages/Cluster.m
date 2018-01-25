(*
-----------------------------------------------------------------------
Computational Intelligence Packages (CIP): Package Cluster
Version 3.0 for Mathematica 11 or higher
-----------------------------------------------------------------------

Authors: Kolja Berger (parallelization for CIP 2.0), Achim Zielesny 

GNWI - Gesellschaft fuer naturwissenschaftliche Informatik mbH, 
Oer-Erkenschwick, Germany

Citation:
Achim Zielesny, Computational Intelligence Packages (CIP), Version 3.0, 
GNWI mbH (http://www.gnwi.de), Oer-Erkenschwick, Germany, 2018.

Code partially based on:
G. A. Carpenter, S. Grossberg, D. B. Rosen, ART 2-A: An Adaptive 
Resonance Algorithm for Rapid Category Learning and Recognition, 
Neural Networks 4, 493-504, 1991.
D. Wienke, Y. Xie, P. K. Hopke, An adaptive resonance theory based 
artificial neural network (ART -2a) for rapid identification of 
airborne particle shapes from their scanning electron microscopy 
images, Chemometrics and Intelligent Laboratory Systems 25, 367-387, 
1994.

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
(* Frequently used data structures *)

(*
-----------------------------------------------------------------------
Frequently used data structures
-----------------------------------------------------------------------
clusterInfo: {clusterMethod, info data structure for clusterMethod, classificationCentroids}

	   clusterMethod = "ART2a"
	   -----------------------
	   art2aInfo: {numberOfInputVectors, numberOfClusters, sortedClusteringResult, vigilanceParameter}
	   sortedClusteringResult: {singleCluster1, singleCluster2, ... singleCluster<numberOfClusters>}
	   NOTE: sortedClusteringResult is sorted in the following way:
	         1. The biggest cluster with the most input vectors
	         2. The next nearest cluster (with smallest angle derived from the scalar product with the biggest cluster)
	         etc.
	   numberOfClusters = Length[sortedClusteringResult]
	   singleCluster: 
	        {
	             Size in percentage of inputVectors, 
	             indexList, 
	             {Distance to center of mass of biggest cluster, centroidVector},
	             {Angle derived from the scalar product with the biggest cluster, art2aCentroidVector}
	        }
	   indexList: {index1, index2, ...} of input vectors in cluster
	   centroidVector: Center of mass centroid vector of cluster
	   art2aCentroidVector: ART2a centroid vector of cluster 

	   clusterMethod = "FindClusters"
	   ------------------------------
	   findClustersInfo: {numberOfInputVectors, numberOfClusters, sortedClusteringResult}
	   sortedClusteringResult: {singleCluster1, singleCluster2, ... singleCluster<numberOfClusters>}
	   NOTE: sortedClusteringResult is sorted in the following way:
	         1. The biggest cluster with the most input vectors
	         2. The next nearest cluster (with smallest euclidean distance from the biggest cluster)
	         etc.
	   numberOfClusters = Length[sortedClusteringResult]
	   singleCluster: 
	        {
	             Size in percentage of inputVectors, 
	             indexList, 
	             {Distance to center of mass of biggest cluster, centroidVector}
	        }
	   indexList: {index1, index2, ...} of input vectors in cluster
	   centroidVector: Center of mass centroid vector of cluster
	   
	   classificationCentroids: Optional
	   classificationCentroids: {centroids for class 1, centroids for class 2, ...}
-----------------------------------------------------------------------
*)

(* ::Section:: *)
(* Package and dependencies *)

BeginPackage["CIP`Cluster`", {"CIP`Utility`", "CIP`DataTransformation`", "CIP`Graphics`", "Combinatorica`"}]

(* ::Section:: *)
(* Off settings *)

Off[General::"spell1"]
Off[General::shdw]
Off[General::compat]

(* ::Section:: *)
(* Options *)

Options[ClusterOptionsMethod] = 
{
	(* "FindClusters", "ART2a" *)
    ClusterOptionMethod -> "FindClusters"
}

Options[ClusterOptionsCalculation] = 
{
	(* "True", "False" *)
    ClusterOptionCreateDistanceMatrix -> True
}

Options[ClusterOptionsFindClusters] = 
{
	(* Method: See Mathematica documentation *)
    ClusterOptionFindClustersMethod -> {"Optimize", "SignificanceTest" -> "Silhouette"}
}

Options[ClusterOptionsArt2a] = 
{
	(* Maximum number of epochs *)
    ClusterOptionMaximumNumberOfEpochs -> 100,
    
	(* Minimum treshold for value of scalar product: 0 (orthogonal unit vectors) < scalarProductMinimumTreshold <= 1.0 (parallel unit vectors) *)
	ClusterOptionScalarProductMinimumTreshold -> 0.99,
	
	(* Vigilance parameter : 0 (rough clustering) < vigilanceParameter <= 1 (fine clustering) *)
	ClusterOptionVigilanceParameter -> 0.1,
	
	(* True: Only number of clusters is returned with faster performance, False: Normal mode *)
	ClusterOptionIsScan-> False,
	
	(* Maximum number of trial steps to adjust vigilance parameter for the desired cluster number *)
	ClusterOptionMaximumTrialSteps -> 50
}

Options[ClusterOptionsComponentStatistics] = 
{
    (* Number of intervals for frequency percentages *)
	ClusterOptionNumberOfIntervals -> 20
}

(* ::Section:: *)
(* Declarations *)

CalculateClusterClassNumber::usage = 
	"CalculateClusterClassNumber[input, clusterInfo]"

CalculateClusterClassNumbers::usage = 
	"CalculateClusterClassNumbers[inputs, clusterInfo]"

FitCluster::usage = 
	"FitCluster[classificationDataSet, options]"

GetClusterRepresentatives::usage = 
	"GetClusterRepresentatives[inputs, numberOfRepresentatives, options]"

GetClusterRepresentativesRelatedIndexLists::usage = 
	"GetClusterRepresentativesRelatedIndexLists[dataSet, trainingFraction, options]"

GetClusterBasedTrainingAndTestSet::usage = 
	"GetClusterBasedTrainingAndTestSet[dataSet, trainingFraction, options]"

GetClusterBasedTrainingAndTestSetIndexList::usage = 
	"GetClusterBasedTrainingAndTestSetIndexList[dataSet, trainingFraction, options]"

GetClusterIndex::usage = 
	"GetClusterIndex[singleInput, inputs, clusterInfo]"

GetClusterMembers::usage = 
	"GetClusterMembers[singleInput, inputs, clusterInfo]"

GetClusterOccupancies::usage = 
	"GetClusterOccupancies[inputsList, inputs, clusterInfo]"

GetClusterProperty::usage = 
	"GetClusterProperty[namedPropertyList, clusterInfo]"

GetClusters::usage = 
	"GetClusters[inputs, options]"

GetComponentStatistics::usage = 
	"GetComponentStatistics[inputs, indexOfComponent, options]"

GetFixedNumberOfClusters::usage = 
	"GetFixedNumberOfClusters[inputs, numberOfClusters, options]"

GetIndexListOfCluster::usage = 
	"GetIndexListOfCluster[indexOfCluster, clusterInfo]"

GetIndexLists::usage = 
	"GetIndexLists[clusterInfo]"
	
GetInputsOfCluster::usage = 
	"GetInputsOfCluster[inputs, indexOfCluster, clusterInfo]"
	
GetNearestVigilanceParameterForClusterNumber::usage = 
	"GetNearestVigilanceParameterForClusterNumber[inputs, numberOfClusters, options]"

GetRandomRepresentatives::usage = 
	"GetRandomRepresentatives[inputs, numberOfRepresentatives, options]"

GetRandomTrainingAndTestSet::usage = 
	"GetRandomTrainingAndTestSet[dataSet, trainingFraction, options]"

GetRepresentativesIndexList::usage = 
	"GetRepresentativesIndexList[inputs, clusterInfo]"

GetSilhouettePlotPoints::usage = 
	"GetSilhouettePlotPoints[inputs, minimumNumberOfClusters, maximumNumberOfClusters, options]"

GetSilhouetteStatistics::usage = 
	"GetSilhouetteStatistics[inputs, clusterInfo]"

GetSilhouetteStatisticsForClusters::usage = 
	"GetSilhouetteStatisticsForClusters[inputs, clusterInfo]"

GetSilhouetteWidthsForClusters::usage = 
	"GetSilhouetteWidthsForClusters[inputs, clusterInfo]"

GetVigilanceParameterScan::usage = 
	"GetVigilanceParameterScan[inputs, minimumVigilanceParameter, maximumVigilanceParameter, numberOfScanPoints, options]"

GetWhiteSpots::usage = 
	"GetWhiteSpots[clusterOccupancies, inputsIndex, threshold]"

ScanClassTrainingWithCluster::usage = 
	"ScanClassTrainingWithCluster[dataSet, trainingFractionList, options]"

ShowClusterClassificationResult::usage = 
	"ShowClusterClassificationResult[namedPropertyList, trainingAndTestSet, clusterInfo]"

ShowClusterSingleClassification::usage = 
	"ShowClusterSingleClassification[namedPropertyList, dataSet, clusterInfo]"

ShowClusterClassificationScan::usage = 
	"ShowClusterClassificationScan[clusterClassificationScan, options]"

ShowClusterResult::usage = 
	"ShowClusterResult[namedPropertyList, clusterInfo]"

ShowClusterOccupancies::usage = 
	"ShowClusterOccupancies[clusterOccupancies, options]"

ShowComponentStatistics::usage = 
	"ShowComponentStatistics[inputs, indexOfComponentList, options]"

ShowSilhouettePlot::usage = 
	"ShowSilhouettePlot[silhouettePlotPoints2D]"

ShowSilhouetteWidthsForCluster::usage = 
	"ShowSilhouetteWidthsForCluster[silhouetteStatisticsForClusters, indexOfCluster]"
 	
ShowVigilanceParameterScan::usage = 
	"ShowVigilanceParameterScan[art2aScanInfo]"

(* ::Section:: *)
(* Functions *)

Begin["`Private`"]

CalculateClusterClassNumber[

	(* Returns class number for specified input for clustering based classification.

	   Returns:
	   Class number of input *)

    
    (* Input in original units: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    input_/;VectorQ[input, NumberQ],

  	(* See "Frequently used data structures" *)
	clusterInfo_
    
	] :=
  
	Module[
    
		{
			classificationCentroids,
			i,
			distances
		},

		classificationCentroids = clusterInfo[[3]];
		distances = 
			Table[
				EuclideanDistance[input, classificationCentroids[[i]]], 
					
				{i, Length[classificationCentroids]}
			];
		Return[CIP`Utility`GetPositionOfMinimumValue[distances]]
	];

CalculateClusterClassNumbers[

	(* Returns class numbers for specified inputs for clustering based classification.

	   Returns:
	   {class number of input1, class number of input2, ...} *)

    
    (* {inputsInOriginalUnit1, inputsInOriginalUnit2, ...}
        inputsInOriginalUnit: {inputValueInOriginalUnit1, inputValueInOriginalUnit2, ...} *)
    inputs_/;MatrixQ[inputs, NumberQ],
    
  	(* See "Frequently used data structures" *)
	clusterInfo_
    
	] :=
  
	Module[
    
		{
			i
		},

		Return[
			Table[
				CalculateClusterClassNumber[inputs[[i]], clusterInfo],
				
				{i, Length[inputs]}
			]
		]
	];

CreateRepresentativesAndRestIndexList[

	(* Returns representatives and rest index lists.

	   Returns :
	   {representativesIndexList, restIndexList, indexLists} 
	   Join[representativesIndexList, restIndexList] would yield all indexes from 1 to number of inputs 
	   indexLists: {indexList1, indexList2, ..., indexList<numberOfClusters>}*)

	
	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],
	
	(* See definition of clusterInfo *)
	sortedClusteringResult_
	
	] := 
	
	Module[
    
	    {
	    	centroidVector,
	    	currentDistance,
	    	currentIndexList,
	    	i,
	    	k,
	    	indexLists,
	    	minimumDistance,
	    	minimumDistanceIndex,
	    	minimumPosition,
			numberOfClusters,
			representativesIndexList,
			restIndexList,
			singleCluster
		},

		numberOfClusters = Length[sortedClusteringResult];

		representativesIndexList = {};
		restIndexList = {};
		indexLists = {};
		Do[
			singleCluster = sortedClusteringResult[[i]];
			currentIndexList = singleCluster[[2]];
			centroidVector = singleCluster[[3, 2]];
			minimumPosition = 1;
			minimumDistanceIndex = 1;
			minimumDistance = Infinity;
			Do[
				currentDistance = EuclideanDistance[centroidVector, inputs[[currentIndexList[[k]]]]];
				If[currentDistance < minimumDistance,
					minimumDistance = currentDistance;
					minimumDistanceIndex = currentIndexList[[k]];
					minimumPosition = k
				],
				
				{k, Length[currentIndexList]}
			];
			AppendTo[representativesIndexList, minimumDistanceIndex];
			restIndexList = Join[restIndexList, Drop[currentIndexList, {minimumPosition}]];
			AppendTo[indexLists, currentIndexList],
			
			{i, numberOfClusters}
		];
		
		Return[{representativesIndexList, restIndexList, indexLists}]
	];

FitCluster[

	(* Returns clusterInfo with classificationCentroids.

	   Returns :
	   clusterInfo with classificationCentroids  *)

	
	(* classificationDataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} 
	   outputValue: 0/1
	   Data set must be a classification data set, i.e. the output components must 0/1 code a class,
	   i.e. class 4 of 5 must be coded {0, 0, 0, 1, 0} *)
    classificationDataSet_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	i,
	    	k,
	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
	    	targetInterval,
	    	sortResult,
	    	sortedClassificationDataSet,
	    	classIndexMinMaxList,
	    	numberOfClasses,
	    	inputs,
	    	clusterInfo,
	    	clusterOccupancies,
	    	centroids,
	    	classificationCentroids,
	    	classIndex,
	    	classIndexList
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		sortResult = CIP`DataTransformation`SortClassificationDataSet[classificationDataSet];
		sortedClassificationDataSet = sortResult[[1]];
		classIndexMinMaxList = sortResult[[2]];
		numberOfClasses = Length[classIndexMinMaxList];
		inputs = CIP`Utility`GetInputsOfDataSet[sortedClassificationDataSet];
		clusterInfo = 
			GetFixedNumberOfClusters[
				inputs,
				numberOfClasses,
				ClusterOptionMethod -> clusterMethod,
				ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				UtilityOptionRandomInitializationMode -> randomValueInitialization,
				DataTransformationOptionTargetInterval -> targetInterval
			];
		centroids = GetCentroidVectors[clusterInfo];
		clusterOccupancies = GetClusterOccupancies[classIndexMinMaxList, clusterInfo];
		classificationCentroids = Table[{},{numberOfClasses}];
		Do[
			classIndexList =  CIP`Utility`GetDescendingValuePositions[clusterOccupancies[[i]]];
			Do[
				classIndex = classIndexList[[k]];
				If[Length[classificationCentroids[[classIndex]]] == 0,
					classificationCentroids[[classIndex]] = centroids[[i]];
					Break[]
				],
				
				{k, Length[classIndexList]}
			],
			
			{i, numberOfClasses}
		];
		AppendTo[clusterInfo, classificationCentroids];
		Return[clusterInfo]
	];

GetArt2aInfo[

	(* Returns ART-2a info from result of method GetClustersWithArt2a[].

	   Returns :
	   art2aInfo: {numberOfInputVectors, numberOfClusters, sortedClusteringResult, vigilanceParameter}
	   sortedClusteringResult: {singleCluster1, singleCluster2, ... singleCluster<numberOfClusters>}
	   NOTE: sortedClusteringResult is sorted in the following way:
	         1. The biggest cluster with the most input vectors
	         2. The next nearest cluster (with smallest angle derived from the scalar product with the biggest cluster)
	         etc.
	   numberOfClusters = Length[sortedClusteringResult]
	   singleCluster: 
	        {
	             Size in percentage of inputVectors, 
	             indexList, 
	             {Distance to center of mass of biggest cluster, centroidVector},
	             {Angle derived from the scalar product with the biggest cluster, art2aCentroidVector}
	        }
	   indexList: {index1, index2, ...} of input vectors in cluster
	   centroidVector: Center of mass centroid vector of cluster
	   art2aCentroidVector: ART2a centroid vector of cluster *)


	(* Index lists of clusters
	   indexLists[[i]] corresponds to art2aCentroidVectors[[i]] *)
	indexLists_,
	
	(* ART2a centroid vectors of clusters 	   
	   indexLists[[i]] corresponds to art2aCentroidVectors[[i]] *)
	art2aCentroidVectors_,
	
	(* Center of mass centroid vectors of clusters 	   
	   indexLists[[i]] corresponds to centroidVectors[[i]] *)
	centroidVectors_,
	
	(* Vigilance parameter *)
	vigilanceParameter_?NumberQ
	
	] := 
	
	Module[
    
	    {
	    	angle,
	    	art2aCentroidVectorOfBiggestCluster,
	    	centroidVectorOfBiggestCluster,
	    	dataToSort,
	    	i,
	    	indexOfBiggestCluster,
	    	numberOfInputVectors,
	    	numberOfClusters,
	    	scalarProduct,
	    	sortedClusteringResult
		},

		numberOfInputVectors = Apply[Plus, Map[Length, indexLists]];
		numberOfClusters = Length[art2aCentroidVectors];
		
		indexOfBiggestCluster = CIP`Utility`GetPositionOfMaximumValue[Map[Length, indexLists]];
		art2aCentroidVectorOfBiggestCluster = art2aCentroidVectors[[indexOfBiggestCluster]];
		centroidVectorOfBiggestCluster = centroidVectors[[indexOfBiggestCluster]];
		
		dataToSort = {};
		Do[
			If[i == indexOfBiggestCluster,
				
				(* Biggest cluster *) 
				AppendTo[dataToSort, 
					{
						0.0, 
						Length[indexLists[[i]]]/numberOfInputVectors*100.0, 
						Sort[indexLists[[i]]], 
						{0.0, centroidVectors[[i]]},
						{0.0, art2aCentroidVectors[[i]]}
					}
				],
				
				(* Other clusters *) 
				scalarProduct = art2aCentroidVectorOfBiggestCluster.art2aCentroidVectors[[i]];
				angle = ArcCos[scalarProduct]*180.0/Pi;
				AppendTo[dataToSort, 
					{
						angle, 
						Length[indexLists[[i]]]/numberOfInputVectors*100.0, 
						Sort[indexLists[[i]]], 
						{EuclideanDistance[centroidVectorOfBiggestCluster, centroidVectors[[i]]], centroidVectors[[i]]},
						{angle, art2aCentroidVectors[[i]]}
					}
				]
			],
			
			{i, Length[art2aCentroidVectors]}
		];
		(* Sort according to angle, then drop angle *)
		sortedClusteringResult = Map[Rest, Sort[dataToSort]];
		Return[
			{
				numberOfInputVectors,
				numberOfClusters,
				sortedClusteringResult,
				vigilanceParameter
			}
		]
	];

GetCentroidVectors[

	(* Returns center of mass centroid vectors.
	
	   Returns:
	   Center of mass centroid vectors *)


  	(* See "Frequently used data structures" *)
	clusterInfo_
    
	] :=
  
	Module[
    
	    {
	    	infoDataStructure,
	    	singleCluster,
	    	i,
	    	sortedClusteringResult
		},

		(* NOTE: This method is applicable for all clustering methods *)
		infoDataStructure = clusterInfo[[2]];
		sortedClusteringResult = infoDataStructure[[3]];
		Return[
			Table[
				singleCluster = sortedClusteringResult[[i]];
				singleCluster[[3, 2]],
				
				{i, Length[sortedClusteringResult]}
			]
		]
	];

GetClusterRepresentatives[

	(* Returns specified number of representatives of the inputs.

	   Returns :
	   Representatives: {input1, input2, ...}  *)

	
	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],
	
	(* Number of representatives *)
	numberOfRepresentatives_?IntegerQ, 
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	maximumNumberOfTrialSteps,
	    	scalarProductMinimumTreshold,
	    	randomValueInitialization,
	    	targetInterval
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		Switch[clusterMethod,
			
			(* NOTE: UtilityOptionRandomInitializationMode is NOT taken into account *)
			"FindClusters", 
			Return[
				GetClusterRepresentativesWithFindClusters[
					inputs,
					numberOfRepresentatives
				]
			],
			
			"ART2a",
			Return[
				GetClusterRepresentativesWithArt2a[
					inputs,
					numberOfRepresentatives,
					ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
					ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
					ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					DataTransformationOptionTargetInterval -> targetInterval
				]
			]
		]		
	];

GetClusterRepresentativesIndexListWithArt2a[

	(* Returns representatives index list of desired size and rest index list on the basis of method GetFixedNumberOfClustersWithArt2a[].

	   Returns :
	   {representativesIndexList, restIndexList, indexLists} 
	   Join[representativesIndexList, restIndexList] would yield all indexes from 1 to number of inputs 
	   indexLists: {indexList1, indexList2, ..., indexList<numberOfClusters>} *)

	
	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],
	
	(* Number of representatives *)
	numberOfRepresentatives_?IntegerQ, 
	
	(* Options *)
	opts___
	
	] := 
	
	Module[
    
	    {
	    	art2aInfo,
	    	maximumNumberOfTrialSteps,
	    	maximumNumberOfEpochs,
			randomValueInitialization,
			scalarProductMinimumTreshold,
	    	i,
			sortedClusteringResult,
			targetInterval
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		If[numberOfRepresentatives >= Length[inputs], 
			Return[
				{
					Table[i, {i, Length[inputs]}], 
					{}
				}
			]
		];

		art2aInfo = Last[
			GetFixedNumberOfClustersWithArt2a[
				inputs,
				numberOfRepresentatives,
				ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				UtilityOptionRandomInitializationMode -> randomValueInitialization,
				DataTransformationOptionTargetInterval -> targetInterval
			]
		];
			
		sortedClusteringResult = art2aInfo[[3]];

		Return[CreateRepresentativesAndRestIndexList[inputs, sortedClusteringResult]]
	];

GetClusterRepresentativesIndexListWithFindClusters[

	(* Returns representatives index list of desired size and rest index list on the basis of method GetFixedNumberOfClustersWithFindClusters[].

	   Returns :
	   {representativesIndexList, restIndexList, indexLists} 
	   Join[representativesIndexList, restIndexList] would yield all indexes from 1 to number of inputs 
	   indexLists: {indexList1, indexList2, ..., indexList<numberOfClusters>} *)

	
	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],
	
	(* Number of representatives *)
	numberOfRepresentatives_?IntegerQ
	
	] := 
	
	Module[
    
	    {
	    	findClustersInfo,
	    	i,
			sortedClusteringResult
		},

		If[numberOfRepresentatives >= Length[inputs], 
			Return[
				{
					Table[i, {i, Length[inputs]}], 
					{}
				}
			]
		];

		findClustersInfo = Last[
			GetFixedNumberOfClustersWithFindClusters[
				inputs,
				numberOfRepresentatives
			]
		];
			
		sortedClusteringResult = findClustersInfo[[3]];

		Return[CreateRepresentativesAndRestIndexList[inputs, sortedClusteringResult]]
	];

GetClusterRepresentativesRelatedIndexLists[

	(* Returns representatives related index lists.

	   Returns :
	   {representativesIndexList, restIndexList, indexLists} 
	   Join[representativesIndexList, restIndexList] would yield all indexes from 1 to number of I/O pairs of data set 
	   indexLists: {indexList1, indexList2, ..., indexList<numberOfClusters>} *)

	
    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...} *)
	dataSet_,
	
	(* 0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFraction_?NumberQ, 
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	maximumNumberOfTrialSteps,
	    	scalarProductMinimumTreshold,
	    	randomValueInitialization,
	    	targetInterval,
	    	numberOfIoPairs,
	    	numberOfRepresentatives,
	    	inputs
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];
		numberOfIoPairs = Length[dataSet];
		numberOfRepresentatives = Floor[numberOfIoPairs*trainingFraction];

		Switch[clusterMethod,
			
			(* NOTE: UtilityOptionRandomInitializationMode is NOT taken into account *)
			"FindClusters", 
			Return[
				GetClusterRepresentativesIndexListWithFindClusters[
					inputs, 
					numberOfRepresentatives
				]
			],
			
			"ART2a",
			Return[
				GetClusterRepresentativesIndexListWithArt2a[
					inputs,
					numberOfRepresentatives,
					ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
					ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
					ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					DataTransformationOptionTargetInterval -> targetInterval
				]
			]
		]		
	];

GetClusterRepresentativesWithArt2a[

	(* Returns specified number of representatives of the inputs with GetFixedNumberOfClustersWithArt2a[] method.

	   Returns :
	   Representatives: {input1, input2, ...}  *)

	
	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],
	
	(* Number of representatives *)
	numberOfRepresentatives_?IntegerQ, 
	
	(* Options *)
	opts___
	
	] := 
	
	Module[
    
	    {
	    	i,
	    	maximumNumberOfTrialSteps,
	    	maximumNumberOfEpochs,
			randomValueInitialization,
			scalarProductMinimumTreshold,
			representativesIndexList,
			representatives,
			targetInterval
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		If[numberOfRepresentatives >= Length[inputs], 
			Return[inputs]
		];

		representativesIndexList = First[
			GetClusterRepresentativesIndexListWithArt2a[
				inputs,
				numberOfRepresentatives,
				ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				UtilityOptionRandomInitializationMode -> randomValueInitialization,
				DataTransformationOptionTargetInterval -> targetInterval
			]
		];
			
		representatives = {};
		Do[
			AppendTo[representatives, inputs[[representativesIndexList[[i]]]]],
			
			{i, Length[representativesIndexList]}
		];
		
		Return[representatives]
	];

GetClusterRepresentativesWithFindClusters[

	(* Returns specified number of representatives of the inputs on the basis of the GetFixedNumberOfClustersWithFindClusters[] method.

	   Returns :
	   Representatives: {input1, input2, ...}  *)

	
	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],
	
	(* Number of representatives *)
	numberOfRepresentatives_?IntegerQ
	
	] := 
	
	Module[
    
	    {
	    	i,
	    	representatives,
	    	representativesIndexList
		},

		If[numberOfRepresentatives >= Length[inputs], 
			Return[inputs]
		];
		
		representativesIndexList = First[
			GetClusterRepresentativesIndexListWithFindClusters[
				inputs, 
				numberOfRepresentatives
			]
		];		
		representatives = {};
		Do[
			AppendTo[representatives, inputs[[representativesIndexList[[i]]]]],
			
			{i, Length[representativesIndexList]}
		];
		
		Return[representatives]
	];

GetClusterBasedTrainingAndTestSet[

	(* Returns training and test set of desired size based on cluster representatives.

	   Returns :
	   trainingAndTestSet: {trainingSet, testSet} 
	   trainingSet and testSet have the same structure as dataSet *)

	
    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...} *)
	dataSet_,
	
	(* 0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFraction_?NumberQ, 
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	maximumNumberOfTrialSteps,
	    	scalarProductMinimumTreshold,
	    	randomValueInitialization,
	    	targetInterval
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		Switch[clusterMethod,
			
			(* NOTE: UtilityOptionRandomInitializationMode is NOT taken into account *)
			"FindClusters", 
			Return[
				GetClusterBasedTrainingAndTestSetWithFindClusters[
					dataSet,
					trainingFraction
				]
			],
			
			"ART2a",
			Return[
				GetClusterBasedTrainingAndTestSetWithArt2a[
					dataSet,
					trainingFraction,
					ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
					ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
					ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					DataTransformationOptionTargetInterval -> targetInterval
				]
			]
		]		
	];

GetClusterBasedTrainingAndTestSetWithArt2a[

	(* Returns training and test set of desired size on the basis of an ART-2a clustering.

	   Returns :
	   trainingAndTestSet: {trainingSet, testSet} 
	   trainingSet and testSet have the same structure as dataSet *)

	
    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...} *)
	dataSet_,
	
	(* 0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFraction_?NumberQ, 
	
	(* Options *)
	opts___
	
	] := 
	
	Module[
    
	    {
	    	i,
	    	indexLists,
	    	maximumNumberOfEpochs,
	    	maximumNumberOfTrialSteps,
			randomValueInitialization,
			result,
			scalarProductMinimumTreshold,
			restIndexList,
			representativesIndexList,
			testSet,
			trainingSet,
			targetInterval
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];

		result = 
			GetClusterBasedTrainingAndTestSetIndexListWithArt2a[
				dataSet,
				trainingFraction,
				ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				UtilityOptionRandomInitializationMode -> randomValueInitialization,
				DataTransformationOptionTargetInterval -> targetInterval
			];
		representativesIndexList = result[[1]];
		restIndexList = result[[2]];
		indexLists =  result[[3]];
		
		trainingSet = {};
		Do[
			AppendTo[trainingSet, dataSet[[representativesIndexList[[i]]]]],
			
			{i, Length[representativesIndexList]}
		];
		
		testSet = {};
		Do[
			AppendTo[testSet, dataSet[[restIndexList[[i]]]]],
			
			{i, Length[restIndexList]}
		];
		
		Return[{trainingSet, testSet}]
	];

GetClusterBasedTrainingAndTestSetWithFindClusters[

	(* Returns training and test set of desired size on the basis of method GetFixedNumberOfClustersWithFindClusters[].

	   Returns :
	   trainingAndTestSet: {trainingSet, testSet} 
	   trainingSet and testSet have the same structure as dataSet *)

	
    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...} *)
	dataSet_,
	
	(* 0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFraction_?NumberQ
	
	] := 
	
	Module[
    
	    {
	    	i,
	    	indexLists,
			representativesIndexList,
			restIndexList,
			result,
			testSet,
			trainingSet
		},

		result = 
			GetClusterBasedTrainingAndTestSetIndexListWithFindClusters[
				dataSet, 
				trainingFraction
			];
		representativesIndexList = result[[1]];
		restIndexList = result[[2]];
		indexLists =  result[[3]];
		
		trainingSet = {};
		Do[
			AppendTo[trainingSet, dataSet[[representativesIndexList[[i]]]]],
			
			{i, Length[representativesIndexList]}
		];
		
		testSet = {};
		If[Length[restIndexList] > 0,
			Do[
				AppendTo[testSet, dataSet[[restIndexList[[i]]]]],
				
				{i, Length[restIndexList]}
			]
		];
		
		Return[{trainingSet, testSet}]
	];

GetClusterBasedTrainingAndTestSetIndexList[

	(* Returns training and test set index list of desired size.

	   Returns :
	   {trainingIndexList, testIndexList, indexLists}
	   trainingIndexList: trainingSet[[i]] = dataSet[[trainingIndexList[[i]]]]
	   testIndexList: testSet[[i]] = dataSet[[testIndexList[[i]]]]
	   Join[trainingIndexList, testIndexList] would yield all indices from 1 to Length[dataSet]
	   indexLists: {indexList1, indexList2, ..., indexList<numberOfClusters>} 
	   Join[indexList1, indexList2, ...] would yield all indices from 1 to Length[dataSet] *)

	
    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...} *)
	dataSet_,
	
	(* 0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFraction_?NumberQ, 
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	maximumNumberOfTrialSteps,
	    	scalarProductMinimumTreshold,
	    	randomValueInitialization,
	    	targetInterval
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		Switch[clusterMethod,
			
			(* NOTE: UtilityOptionRandomInitializationMode is NOT taken into account *)
			"FindClusters", 
			Return[
				GetClusterBasedTrainingAndTestSetIndexListWithFindClusters[
					dataSet,
					trainingFraction
				]
			],
			
			"ART2a",
			Return[
				GetClusterBasedTrainingAndTestSetIndexListWithArt2a[
					dataSet,
					trainingFraction,
					ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
					ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
					ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					DataTransformationOptionTargetInterval -> targetInterval
				]
			]
		]		
	];

GetClusterBasedTrainingAndTestSetIndexListWithArt2a[

	(* Returns training and test set index list of desired size on the basis of an ART-2a clustering.

	   Returns :
	   {trainingIndexList, testIndexList, indexLists}
	   trainingIndexList: trainingSet[[i]] = dataSet[[trainingIndexList[[i]]]]
	   testIndexList: testSet[[i]] = dataSet[[testIndexList[[i]]]]
	   Join[trainingIndexList, testIndexList] would yield all indices from 1 to Length[dataSet]
	   indexLists: {indexList1, indexList2, ..., indexList<numberOfClusters>} 
	   Join[indexList1, indexList2, ...] would yield all indices from 1 to Length[dataSet] *)

	
    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...} *)
	dataSet_,
	
	(* 0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFraction_?NumberQ, 
	
	(* Options *)
	opts___
	
	] := 
	
	Module[
    
	    {
	    	i,
	    	inputs,
	    	maximumNumberOfEpochs,
	    	maximumNumberOfTrialSteps,
			numberOfIoPairs,
			numberOfTrainingSetIoPairs,
			randomValueInitialization,
			scalarProductMinimumTreshold,
			targetInterval
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];

		numberOfIoPairs = Length[dataSet];
		numberOfTrainingSetIoPairs = Floor[numberOfIoPairs*trainingFraction];
		If[numberOfTrainingSetIoPairs == numberOfIoPairs, 
			Return[
				{
					Table[i, {i, Length[dataSet]}], 
					{}
				}
			]
		];

		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];
		
		Return[
			GetClusterRepresentativesIndexListWithArt2a[
				inputs,
				numberOfTrainingSetIoPairs,
				ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				UtilityOptionRandomInitializationMode -> randomValueInitialization,
				DataTransformationOptionTargetInterval -> targetInterval
			]
		]
	];

GetClusterBasedTrainingAndTestSetIndexListWithFindClusters[

	(* Returns training and test set index list of desired size on the basis of method GetFixedNumberOfClustersWithFindClusters[].

	   Returns :
	   {trainingIndexList, testIndexList, indexLists}
	   trainingIndexList: trainingSet[[i]] = dataSet[[trainingIndexList[[i]]]]
	   testIndexList: testSet[[i]] = dataSet[[testIndexList[[i]]]]
	   Join[trainingIndexList, testIndexList] would yield all indices from 1 to Length[dataSet]
	   indexLists: {indexList1, indexList2, ..., indexList<numberOfClusters>} 
	   Join[indexList1, indexList2, ...] would yield all indices from 1 to Length[dataSet] *)

	
    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...} *)
	dataSet_,
	
	(* 0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFraction_?NumberQ
	
	] := 
	
	Module[
    
	    {
	    	i,
	    	inputs,
			numberOfIoPairs,
			numberOfTrainingSetIoPairs
		},

		numberOfIoPairs = Length[dataSet];
		numberOfTrainingSetIoPairs = Floor[numberOfIoPairs*trainingFraction];
		If[numberOfTrainingSetIoPairs == numberOfIoPairs, 
			Return[
				{
					Table[i, {i, Length[dataSet]}], 
					{}
				}
			]
		];

		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];

		Return[
			GetClusterRepresentativesIndexListWithFindClusters[
				inputs, 
				numberOfTrainingSetIoPairs
			]
		]
	];

GetClusterIndex[

	(* Returns cluster index for specified input vector singleInput.
	
	   Returns:
	   Cluster index or -1 if no cluster was found. *)

	(* singleInput = vector : {component1, ..., component<NumberOfComponentsOfVector>} *)
    singleInput_/;VectorQ[singleInput, NumberQ],

	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],

  	(* See "Frequently used data structures" *)
	clusterInfo_
    
	] :=
  
	Module[
    
	    {
	    	i,
	    	infoDataStructure,
	    	inputsOfClusterList,
	    	numberOfClusters
		},

		(* NOTE: This method is applicable for all clustering methods *)
		infoDataStructure = clusterInfo[[2]];
		numberOfClusters = infoDataStructure[[2]];

		inputsOfClusterList = 
			Table[
				GetInputsOfCluster[inputs, i, clusterInfo],
				
				{i, numberOfClusters}
			];

		Do[
			If[MemberQ[inputsOfClusterList[[i]], singleInput],
				Return[i]
			],
			
			{i, numberOfClusters}
		];
		
		Return[-1]
	];

GetClusterMembers[

	(* Returns all input vectors of cluster of specified input vector singleInput.
	
	   Returns:
	   Cluster members: {vector1 of cluster, vector2 of cluster, ...} 
	   or 
	   {} if no cluster was found *)

	(* singleInput = vector : {component1, ..., component<NumberOfComponentsOfVector>} *)
    singleInput_/;VectorQ[singleInput, NumberQ],

	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],

  	(* See "Frequently used data structures" *)
	clusterInfo_
    
	] :=
  
	Module[
    
	    {
	    	i,
	    	infoDataStructure,
	    	inputsOfClusterList,
	    	numberOfClusters
		},

		(* NOTE: This method is applicable for all clustering methods *)
		infoDataStructure = clusterInfo[[2]];
		numberOfClusters = infoDataStructure[[2]];

		inputsOfClusterList = 
			Table[
				GetInputsOfCluster[inputs, i, clusterInfo],
				
				{i, numberOfClusters}
			];

		Do[
			If[MemberQ[inputsOfClusterList[[i]], singleInput],
				Return[inputsOfClusterList[[i]]]
			],
			
			{i, numberOfClusters}
		];
		
		Return[{}]
	];

GetClusterOccupancies[

	(* Returns cluster occupancies.
	
	   Returns:
	   clusterOccupancies: {clusterOccupancy1, clusterOccupancy1, ..., clusterOccupancy1<numberOfClusters>} 
	   clusterOccupancy: {Percent for inputs1, percent for inputs2, ..., percent for inputs<Length[inputsIndexMinMaxList]>} 
	   clusterOccupancy[[i]] refers to cluster[[i]] in clusterInfo *)

	(* {minMaxIndexOfInputs1, minMaxIndexOfInputs2, ...}
	   minMaxIndexOfInputs : {minIndexOfInputs, maxIndexOfInputs} *)
    inputsIndexMinMaxList_/;MatrixQ[inputsIndexMinMaxList, NumberQ],

  	(* See "Frequently used data structures" *)
	clusterInfo_
    
	] :=
  
	Module[
    
	    {
	    	clusterOccupancies,
	    	i,
	    	indexList,
	    	infoDataStructure,
	    	k,
	    	l,
	    	numberOfClusters,
	    	numberOfInputsList,
	    	inputsFrequency,
	    	singleCluster,
	    	singleIndex,
	    	sortedClusteringResult,
	    	sortedInputsIndexMinMax,
	    	sortedInputsIndexMinMaxList
		},

		(* NOTE: This method is applicable for all clustering methods *)
		infoDataStructure = clusterInfo[[2]];
		sortedClusteringResult = infoDataStructure[[3]];
		numberOfClusters = Length[sortedClusteringResult];

		sortedInputsIndexMinMaxList = Sort[inputsIndexMinMaxList];
		numberOfInputsList=
			Table[
				sortedInputsIndexMinMax = sortedInputsIndexMinMaxList[[i]];
				sortedInputsIndexMinMax[[2]] - sortedInputsIndexMinMax[[1]] + 1,
				
				{i, Length[sortedInputsIndexMinMaxList]}
			];
		
		clusterOccupancies = {};
		Do[
			inputsFrequency = Table[0, {Length[sortedInputsIndexMinMaxList]}];
			singleCluster = sortedClusteringResult[[i]];
			indexList = singleCluster[[2]];
			Do[
				singleIndex = indexList[[k]];
				Do[
					If[singleIndex <= sortedInputsIndexMinMaxList[[l, 2]],
						inputsFrequency[[l]] = inputsFrequency[[l]] + 1;
						Break[]		
					],
					
					{l, Length[sortedInputsIndexMinMaxList]}
				],

				{k, Length[indexList]}
			];
			AppendTo[clusterOccupancies, 
				Table[
					N[CIP`Utility`RoundTo[inputsFrequency[[k]]/numberOfInputsList[[k]]*100, 1]], 
					
					{k, Length[numberOfInputsList]}
				]
			],
			
			{i, numberOfClusters}
		];
		
		Return[clusterOccupancies]
	];

GetClusterProperty[

	(* Returns list with named cluster properties.
	
	   Returns:
	   List with named cluster properties *)

	(* Properties, full list: 
	   {
	       "NumberOfInputVectors",
		   "NumberOfClusters",
		   "CentroidVectors"
	    } *)
 	namedPropertyList_,

  	(* See "Frequently used data structures" *)
	clusterInfo_
    
	] :=

	Module[
    
    	{
    		i,
    		namedProperty,
    		result
    	},
    	
    	result = {};
    	Do[
    		namedProperty = namedPropertyList[[i]];
    		AppendTo[result, GetSingleClusterProperty[namedProperty, clusterInfo]],
    		
    		{i, Length[namedPropertyList]}
    	];
    	
    	Return[result];
	];

GetClusters[

	(* Performs clustering of inputs.
	
	   Returns:
	   clusterInfo (see "Frequently used data structures" above) *)


	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	clusterMethod,
	    	findClustersMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	vigilanceParameter,
	    	isScan,
	    	randomValueInitialization,
	    	targetInterval
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    findClustersMethod = ClusterOptionFindClustersMethod/.{opts}/.Options[ClusterOptionsFindClusters];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    vigilanceParameter = ClusterOptionVigilanceParameter/.{opts}/.Options[ClusterOptionsArt2a];
	    isScan = ClusterOptionIsScan/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		Switch[clusterMethod,
			
			(* NOTE: UtilityOptionRandomInitializationMode is NOT taken into account *)
			"FindClusters", 
			Return[
				GetClustersWithFindClusters[
					inputs,
					ClusterOptionFindClustersMethod -> findClustersMethod
				]
			],
			
			"ART2a",
			Return[
				GetClustersWithArt2a[
					inputs, 
					ClusterOptionVigilanceParameter -> vigilanceParameter,
					ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
					ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
		    		ClusterOptionIsScan-> isScan,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					DataTransformationOptionTargetInterval -> targetInterval
				]
			]
		]		
	];

GetClustersWithArt2a[

	(* Performs ART2a clustering of inputs.
	
	   Returns:
	   ClusterOptionIsScan= False:
	   ---------------------
	   clusterInfo: {"ART2a", art2aInfo}
	   art2aInfo: {numberOfInputVectors, numberOfClusters, sortedClusteringResult, vigilanceParameter}
	   sortedClusteringResult: {singleCluster1, singleCluster2, ... singleCluster<numberOfClusters>}
	   NOTE: sortedClusteringResult is sorted in the following way:
	         1. The biggest cluster with the most input vectors
	         2. The next nearest cluster (with smallest angle derived from the scalar product with the biggest cluster)
	         etc.
	   numberOfClusters = Length[sortedClusteringResult]
	   singleCluster: 
	        {
	             Size in percentage of inputVectors, 
	             indexList, 
	             {Distance to center of mass of biggest cluster, centroidVector},
	             {Angle derived from the scalar product with the biggest cluster, art2aCentroidVector}
	        }
	   indexList: {index1, index2, ...} of input vectors in cluster
	   centroidVector: Center of mass centroid vector of cluster
	   art2aCentroidVector: ART2a centroid vector of cluster
	   
	   ClusterOptionIsScan= True:
	   --------------------
	   Number of detected clusters
	   NOTE: Empty clusters are NOT checked/removed for faster performance *)


	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
			art2aCentroidVectors,
			art2aCentroidVectorsOld,
			centroidVectors,
			contrastEnhancedRandomVector,
			dataMatrixScaleInfo,
			emptyClusterIndexList,
			i,
			indexLists,
			indexWinner,
			isScan,
			k,
			learningRate,
			maximumNumberOfEpochs,
			numberOfEpochs,
			randomValueInitialization,
			rhoCompare,
			rhoList,
			rhoWinner,
			scalarProductMinimumTreshold,
			scaledInputs,
			scalingFactor,
			targetInterval,
			thresholdForContrastEnhancement,
			trainingList,
			vigilanceParameter
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    vigilanceParameter = ClusterOptionVigilanceParameter/.{opts}/.Options[ClusterOptionsArt2a];
	    isScan = ClusterOptionIsScan/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];


		(* ----------------------------------------------------------------------------------------------------
		   Initialization
		   ---------------------------------------------------------------------------------------------------- *)
		(* Set seed for random numbers if necessary *)
		If[randomValueInitialization == "Seed", SeedRandom[1], SeedRandom[]];

		(* Scale Data *)
		dataMatrixScaleInfo = CIP`DataTransformation`GetDataMatrixScaleInfo[inputs, targetInterval];
		scaledInputs = CIP`DataTransformation`ScaleDataMatrix[inputs, dataMatrixScaleInfo];

		(* Definitions *)
		thresholdForContrastEnhancement = 1.0/Sqrt[Length[scaledInputs[[1]]] + 1.0];
		scalingFactor = thresholdForContrastEnhancement;
		learningRate = 0.01;

		(* Initialize *)
		art2aCentroidVectors = {};
		art2aCentroidVectorsOld = {};
		indexLists = {};
		numberOfEpochs = 0;

		(* ---------------------------------------------------------------------------------------------------------------------- *)
		(* Train ART2a network *)
		(* ---------------------------------------------------------------------------------------------------------------------- *)
		While[!HasConverged[art2aCentroidVectors, art2aCentroidVectorsOld, scalarProductMinimumTreshold] && numberOfEpochs < maximumNumberOfEpochs,

			numberOfEpochs++;

			(* Save cluster centroid vectors of previous training epoch *)
			art2aCentroidVectorsOld = art2aCentroidVectors;

			(* Create random training list *)
			trainingList = Combinatorica`RandomPermutation[Length[scaledInputs]];

			(* Initialize *)
			If [!isScan && Length[art2aCentroidVectors] > 0,
				indexLists = Table[{}, {Length[art2aCentroidVectors]}]
			];
      
			(* Loop over all vectors *)
			Do[
				(* Normalize random vector *)
				contrastEnhancedRandomVector = 
					GetContrastEnhancedVector[
						CIP`Utility`NormalizeVector[scaledInputs[[trainingList[[i]]]]], 
						thresholdForContrastEnhancement
					];

				If[Length[art2aCentroidVectors] == 0,
          
					(* There are no detected clusters *)
					art2aCentroidVectors = {contrastEnhancedRandomVector};
					If[!isScan, indexLists = {{trainingList[[i]]}}],
          
					(* Detected clusters exist *)
					rhoList = art2aCentroidVectors.contrastEnhancedRandomVector;
					indexWinner = First[Ordering[rhoList, -1]];
					rhoWinner = Max[rhoList];
					rhoCompare = scalingFactor*Apply[Plus, contrastEnhancedRandomVector];
					If[rhoCompare > rhoWinner,

						(* Increase number of clusters *)
						AppendTo[art2aCentroidVectors, contrastEnhancedRandomVector];
						If[!isScan, AppendTo[indexLists, {trainingList[[i]]}]],

						(* Compare to vigilanceParameter *)
						If[rhoWinner < vigilanceParameter,

							(* Increase number of clusters *)
							AppendTo[art2aCentroidVectors, contrastEnhancedRandomVector];
							If[!isScan, AppendTo[indexLists, {trainingList[[i]]}]],

							(* Modify existing cluster *)
							art2aCentroidVectors[[indexWinner]] = 
								CIP`Utility`NormalizeVector[
									(1.0 - learningRate)*art2aCentroidVectors[[indexWinner]] + 
										(learningRate*
											CIP`Utility`NormalizeVector[
			                          			Table[
													If[art2aCentroidVectors[[indexWinner, k]] > thresholdForContrastEnhancement, 
														
														(* True *)
														contrastEnhancedRandomVector[[k]], 
														
														(* False *)
														0.0
													],
													
													{k, Length[contrastEnhancedRandomVector]}
												]
											]
										)
								];
							If[!isScan, AppendTo[indexLists[[indexWinner]], trainingList[[i]]]]
						]
					]
				],
        
				{i, Length[trainingList]}
        	];(* End Do *)
        	
        	(* Check indexLists for empty clusters if NOT in scan mode *)
        	If[!isScan, 
	        	emptyClusterIndexList = {};
	        	Do[
	        		If[Length[indexLists[[i]]] == 0,
						AppendTo[emptyClusterIndexList, i]
	        		],
	        		
	        		{i, Length[indexLists]}
	        	];
	        	(* Remove empty clusters *)
	        	If[Length[emptyClusterIndexList] > 0,
	        		Do[
	        			Drop[indexLists, {emptyClusterIndexList[[i]]}];
	        			Drop[art2aCentroidVectors, {emptyClusterIndexList[[i]]}],
	        			
	        			{i, Length[emptyClusterIndexList]}
	        		]
	        	]
        	]
		];(* End While *)
    
    	If[!isScan,
    		
    		centroidVectors = 
    			Table[
    				CIP`Utility`GetCentroidVectorFromIndexedInputs[inputs, indexLists[[i]]],
    				
    				{i, Length[indexLists]}
    			];
			Return[
				{
					"ART2a",
					GetArt2aInfo[
						indexLists, 
						art2aCentroidVectors,
						centroidVectors,
						vigilanceParameter
					]
				}
			],
			
			Return[Length[art2aCentroidVectors]]
    	]
	];

GetClustersWithFindClusters[

	(* Performs clustering of inputs with FindClusters. NOTE: UtilityOptionRandomInitializationMode is NOT taken into account.
	
	   Returns:
	   clusterInfo: {"FindClusters", findClustersInfo}
	   findClustersInfo: {numberOfInputVectors, numberOfClusters, sortedClusteringResult}
	   sortedClusteringResult: {singleCluster1, singleCluster2, ... singleCluster<numberOfClusters>}
	   NOTE: sortedClusteringResult is sorted in the following way:
	         1. The biggest cluster with the most input vectors
	         2. The next nearest cluster (with smallest euclidean distance from the biggest cluster)
	         etc.
	   numberOfClusters = Length[sortedClusteringResult]
	   singleCluster: 
	        {
	             Size in percentage of inputVectors, 
	             indexList, 
	             {Distance to center of mass of biggest cluster, centroidVector}
	        }
	   indexList: {index1, index2, ...} of input vectors in cluster
	   centroidVector: Center of mass centroid vector of cluster *)


	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	centroidVectors,
	    	indexLists,
	    	CurveFitOptionMethod
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    CurveFitOptionMethod = ClusterOptionFindClustersMethod/.{opts}/.Options[ClusterOptionsFindClusters];

		indexLists = 
			FindClusters[
				inputs -> Range[Length[inputs]],
				Method -> CurveFitOptionMethod
			];

		centroidVectors = 
			Table[
				CIP`Utility`GetCentroidVectorFromIndexedInputs[inputs, indexLists[[i]]],
				
				{i, Length[indexLists]}
			];
			
		Return[
			{
				"FindClusters",
				GetFindClustersInfo[
					indexLists, 
					centroidVectors
				]
			}
		]
	];

GetComponentStatistics[

	(* Returns statistics for specified component of input vectors.
	
	   Returns:
	   componentStatistics: {numberOfIntervals, intervalPoints, statistics}
	   numberOfIntervals: Number of intervals
	   intervalPoints: {interval1, interval2, ...}
	   interval: {middle position, frequency in percent of interval}
	   statistics: {min, max, mean, median} of specified component *)


	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],

    (* Index of component of vectors in inputs *)
	indexOfComponent_?IntegerQ,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	frequencyList,
	    	i,
	    	index,
	    	inputVector,
	    	intervalLength,
	    	intervalMiddlePositionList,
	    	intervalPoints,
	    	max,
	    	mean,
	    	median,
	    	min,
	    	numberOfInputs,
	    	numberOfIntervals,
	    	valueList
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    numberOfIntervals = ClusterOptionNumberOfIntervals/.{opts}/.Options[ClusterOptionsComponentStatistics];

		numberOfInputs = Length[inputs];
		
		valueList = inputs[[All, indexOfComponent]];
		min = Min[valueList];
		max = Max[valueList];
		mean = Mean[valueList];
		median = Median[valueList];

		intervalLength = (max - min)/numberOfIntervals;
		frequencyList = Table[0.0, {numberOfIntervals}];
		Do[
			inputVector = inputs[[i]];
			index = Floor[(inputVector[[indexOfComponent]] - min)/intervalLength] + 1;
			If[index > numberOfIntervals, index = numberOfIntervals];
			frequencyList[[index]] = frequencyList[[index]] + 1.0,
			
			{i, Length[inputs]}
		];		
		intervalMiddlePositionList = Table[min + intervalLength*(i - 0.5), {i, numberOfIntervals}];
		intervalPoints = Table[{intervalMiddlePositionList[[i]], frequencyList[[i]]/numberOfInputs*100.0}, {i, numberOfIntervals}];
					
		Return[
			{
				numberOfIntervals,
				intervalPoints,
				{min, max, mean, median}
			}
		]
	];

GetContrastEnhancedVector[

	(* Returns contrast enhanced vector.

	   Returns :
	   Contrast enhanced vector of same structure as argument vector *)

	
	vector_,
	
	thresholdForContrastEnhancement_
	
	] := CIP`Utility`NormalizeVector[Map[If[# > thresholdForContrastEnhancement, #, 0.0] &, vector]];    

GetFixedNumberOfClusters[

	(* Performs clustering of inputs with an a priori fixed number of resulting clusters.
	
	   Returns:
	   clusterInfo (see "Frequently used data structures" above) *)


	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],
	
	(* Number of clusters *)
	numberOfClusters_?IntegerQ, 
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	maximumNumberOfTrialSteps,
	    	scalarProductMinimumTreshold,
	    	randomValueInitialization,
	    	targetInterval
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		Switch[clusterMethod,
			
			(* NOTE: UtilityOptionRandomInitializationMode is NOT taken into account *)
			"FindClusters", 
			Return[
				GetFixedNumberOfClustersWithFindClusters[
					inputs,
					numberOfClusters
				]
			],
			
			"ART2a",
			Return[
				GetFixedNumberOfClustersWithArt2a[
					inputs,
					numberOfClusters,
					ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
					ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
					ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					DataTransformationOptionTargetInterval -> targetInterval
				]
			]
		]		
	];

GetFixedNumberOfClustersWithArt2a[

	(* Clusters inputs with ART2a into specified number of clusters if possible.

	   Returns:
	   clusterInfo: {"ART2a", art2aInfo}
	   art2aInfo: {numberOfInputVectors, numberOfClusters, sortedClusteringResult, vigilanceParameter}
	   sortedClusteringResult: {singleCluster1, singleCluster2, ... singleCluster<numberOfClusters>}
	   NOTE: sortedClusteringResult is sorted in the following way:
	         1. The biggest cluster with the most input vectors
	         2. The next nearest cluster (with smallest angle derived from the scalar product with the biggest cluster)
	         etc.
	   numberOfClusters = Length[sortedClusteringResult]
	   singleCluster: 
	        {
	             Size in percentage of inputVectors, 
	             indexList, 
	             {Distance to center of mass of biggest cluster, centroidVector},
	             {Angle derived from the scalar product with the biggest cluster, art2aCentroidVector}
	        }
	   indexList: {index1, index2, ...} of input vectors in cluster
	   centroidVector: Center of mass centroid vector of cluster
	   art2aCentroidVector: ART2a centroid vector of cluster *)

	
	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],
	
	(* Number of clusters *)
	numberOfClusters_?IntegerQ, 
	
	(* Options *)
	opts___
	
	] := 
	
	Module[
    
	    {
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
			targetInterval,
			vigilanceParameterInfo,
			vigilanceParameter
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		vigilanceParameterInfo = 
			GetNearestVigilanceParameterForClusterNumber[
				inputs, 
				numberOfClusters,
				ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
				UtilityOptionRandomInitializationMode -> randomValueInitialization,
				DataTransformationOptionTargetInterval -> targetInterval
			];
		vigilanceParameter = vigilanceParameterInfo[[1]];
		Return[ 
			GetClustersWithArt2a[
				inputs, 
				ClusterOptionVigilanceParameter -> vigilanceParameter,
				ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
	    		ClusterOptionIsScan-> False,
				UtilityOptionRandomInitializationMode -> randomValueInitialization,
				DataTransformationOptionTargetInterval -> targetInterval
			]
		]
	];

GetFixedNumberOfClustersWithFindClusters[

	(* Clusters inputs with FindClusters into specified number of clusters. NOTE: UtilityOptionRandomInitializationMode is NOT taken into account.
	
	   Returns:
	   clusterInfo: {"FindClusters", findClustersInfo}
	   findClustersInfo: {numberOfInputVectors, numberOfClusters, sortedClusteringResult}
	   sortedClusteringResult: {singleCluster1, singleCluster2, ... singleCluster<numberOfClusters>}
	   NOTE: sortedClusteringResult is sorted in the following way:
	         1. The biggest cluster with the most input vectors
	         2. The next nearest cluster (with smallest euclidean distance from the biggest cluster)
	         etc.
	   numberOfClusters = Length[sortedClusteringResult]
	   singleCluster: 
	        {
	             Size in percentage of inputVectors, 
	             indexList, 
	             {Distance to center of mass of biggest cluster, centroidVector}
	        }
	   indexList: {index1, index2, ...} of input vectors in cluster
	   centroidVector: Center of mass centroid vector of cluster *)


	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],
	
	(* Number of clusters *)
	numberOfClusters_?IntegerQ
    
	] :=
  
	Module[
    
	    {
	    	centroidVectors,
	    	indexLists
		},

		indexLists = 
			FindClusters[
				inputs -> Range[Length[inputs]],
				numberOfClusters
			];

		centroidVectors = 
			Table[
				CIP`Utility`GetCentroidVectorFromIndexedInputs[inputs, indexLists[[i]]],
				
				{i, Length[indexLists]}
			];
			
		Return[
			{
				"FindClusters",
				GetFindClustersInfo[
					indexLists, 
					centroidVectors
				]
			}
		]
	];

GetFindClustersInfo[

	(* Returns findClusters info from result of method GetClustersWithFindClusters[].

	   Returns :
	   findClustersInfo: {numberOfInputVectors, numberOfClusters, sortedClusteringResult}
	   sortedClusteringResult: {singleCluster1, singleCluster2, ... singleCluster<numberOfClusters>}
	   NOTE: sortedClusteringResult is sorted in the following way:
	         1. The biggest cluster with the most input vectors
	         2. The next nearest cluster (with smallest euclidean distance from the biggest cluster)
	         etc.
	   numberOfClusters = Length[sortedClusteringResult]
	   singleCluster: 
	        {
	             Size in percentage of inputVectors, 
	             indexList, 
	             {Distance to center of mass of biggest cluster, centroidVector}
	        }
	   indexList: {index1, index2, ...} of input vectors in cluster
	   centroidVector: Center of mass centroid vector of cluster *)


	(* Index lists of clusters
	   indexLists[[i]] corresponds to centroidVectors[[i]] *)
	indexLists_,
	
	(* Center of mass centroid vectors of clusters 	   
	   indexLists[[i]] corresponds to centroidVectors[[i]] *)
	centroidVectors_
	
	] := 
	
	Module[
    
	    {
	    	centroidVectorOfBiggestCluster,
	    	dataToSort,
	    	euclideanDistance,
	    	i,
	    	indexOfBiggestCluster,
	    	numberOfInputVectors,
	    	numberOfClusters,
	    	sortedClusteringResult
		},

		numberOfInputVectors = Apply[Plus, Map[Length, indexLists]];
		numberOfClusters = Length[centroidVectors];
		
		indexOfBiggestCluster = CIP`Utility`GetPositionOfMaximumValue[Map[Length, indexLists]];
		centroidVectorOfBiggestCluster = centroidVectors[[indexOfBiggestCluster]];
		
		dataToSort = {};
		Do[
			If[i == indexOfBiggestCluster,
				
				(* Biggest cluster *) 
				AppendTo[dataToSort, 
					{
						0.0, 
						Length[indexLists[[i]]]/numberOfInputVectors*100.0, 
						Sort[indexLists[[i]]], 
						{0.0, centroidVectors[[i]]}
					}
				],
				
				(* Other clusters *) 
				euclideanDistance = EuclideanDistance[centroidVectorOfBiggestCluster, centroidVectors[[i]]];
				AppendTo[dataToSort, 
					{
						euclideanDistance, 
						Length[indexLists[[i]]]/numberOfInputVectors*100.0, 
						Sort[indexLists[[i]]], 
						{euclideanDistance, centroidVectors[[i]]}
					}
				]
			],
			
			{i, Length[centroidVectors]}
		];
		(* Sort according to euclidean distance, then drop euclidean distance *)
		sortedClusteringResult = Map[Rest, Sort[dataToSort]];
		Return[
			{
				numberOfInputVectors,
				numberOfClusters,
				sortedClusteringResult
			}
		]
	];

GetIndexListOfCluster[

	(* Returns index list of specified cluster.
	
	   Returns:
	   Index list of cluster *)

	(* Index of cluster in sortedClusteringResult *)
	indexOfCluster_?IntegerQ,

  	(* See "Frequently used data structures" *)
	clusterInfo_
    
	] :=
  
	Module[
    
	    {
	    	indexList,
	    	infoDataStructure,
	    	singleCluster,
			sortedClusteringResult
		},

		(* NOTE: This method is applicable for all clustering methods *)
		infoDataStructure = clusterInfo[[2]];

		sortedClusteringResult = infoDataStructure[[3]];
		singleCluster = sortedClusteringResult[[indexOfCluster]];
		indexList = singleCluster[[2]]; 
		
		Return[indexList]
    ];

GetIndexLists[

	(* Returns list with all index lists in the sequence of clusters in clusterInfo.
	
	   Returns:
	   indexLists: {indexList1, indexList2, ..., indexList<numberOfClusters>} 
	   indexList[[i]]: Index list of cluster i *)

  	(* See "Frequently used data structures" *)
	clusterInfo_
    
	] :=
  
	Module[
    
	    {
	    	i,
	    	indexList,
	    	indexLists,
	    	infoDataStructure,
	    	singleCluster,
			sortedClusteringResult,
			numberOfClusters
		},

		(* NOTE: This method is applicable for all clustering methods *)
		infoDataStructure = clusterInfo[[2]];
		sortedClusteringResult = infoDataStructure[[3]];
		numberOfClusters = Length[sortedClusteringResult];
		
		indexLists = 
			Table[
				singleCluster = sortedClusteringResult[[i]];
				indexList = singleCluster[[2]],
				
				{i, numberOfClusters}
			];
		
		Return[indexLists]
    ];

GetInputsOfCluster[

	(* Returns inputs of specified cluster.
	
	   Returns:
	   Inputs of cluster *)

	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsInVectorOfDataMatrix>}
	   NOTE : component >= 0
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],

	(* Index of cluster in sortedClusteringResult *)
	indexOfCluster_?IntegerQ,

  	(* See "Frequently used data structures" *)
	clusterInfo_
    
	] :=
  
	Module[
    
	    {
	    	indexList,
	    	infoDataStructure,
	    	i,
	    	inputsOfCluster,
	    	singleCluster,
			sortedClusteringResult
		},

		(* NOTE: This method is applicable for all clustering methods *)
		infoDataStructure = clusterInfo[[2]];

		sortedClusteringResult = infoDataStructure[[3]];
		singleCluster = sortedClusteringResult[[indexOfCluster]];
		indexList = singleCluster[[2]]; 
		
		inputsOfCluster = {};
		Do[
			AppendTo[inputsOfCluster, inputs[[indexList[[i]]]]],
						
			{i, Length[indexList]}
		];
		
		Return[inputsOfCluster]
    ];

GetNearestVigilanceParameterForClusterNumber[

	(* Returns nearest lower vigilance parameter for specified number of clusters.
	
	   Returns:
	   vigilanceParameterAdjustmentInfo: {nearest lower vigilanceParameter, detected numberOfClusters} *)


	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsInVectorOfDataMatrix>}
	   NOTE : component >= 0
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],

    (* Desired number of clusters (> 1) *)
	numberOfClusters_?IntegerQ, 
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	currentNumberOfClusters,
	    	currentVigilanceParameter,
	    	lowerVigilanceParameter,
	    	maximumNumberOfEpochs,
	    	maximumNumberOfTrialSteps,
	    	numberOfTrialSteps,
	    	randomValueInitialization,
	    	returnNumberOfClusters,
	    	returnVigilanceParameter,
	    	scalarProductMinimumTreshold,
	    	upperVigilanceParameter,
			targetInterval
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		lowerVigilanceParameter = 0.0;
		upperVigilanceParameter = 1.0;
		numberOfTrialSteps = 1;
		While[numberOfTrialSteps < maximumNumberOfTrialSteps,
			currentVigilanceParameter = (upperVigilanceParameter + lowerVigilanceParameter)/2.0;
			currentNumberOfClusters = 
				GetClustersWithArt2a[
					inputs,
					ClusterOptionVigilanceParameter -> currentVigilanceParameter,
					ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
					ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
					ClusterOptionIsScan-> True,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					DataTransformationOptionTargetInterval -> targetInterval
				];
				
			If[currentNumberOfClusters == numberOfClusters, 
				Return[{currentVigilanceParameter, currentNumberOfClusters}]
			];
			
			If[currentNumberOfClusters < numberOfClusters,
				
				(* currentNumberOfClusters < numberOfClusters *)
				lowerVigilanceParameter = currentVigilanceParameter;
				returnVigilanceParameter = currentVigilanceParameter;
				returnNumberOfClusters = currentNumberOfClusters,
				
				(* currentNumberOfClusters > numberOfClusters *)
				upperVigilanceParameter = currentVigilanceParameter
			];
			numberOfTrialSteps++;
		];
		Return[{returnVigilanceParameter, returnNumberOfClusters}]
	];

GetRandomRepresentatives[

	(* Returns specified number of random representatives of the inputs.

	   Returns :
	   Representatives: {input1, input2, ...}  *)

	
	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],
	
	(* Number of representatives *)
	numberOfRepresentatives_?IntegerQ,
	
	(* Options *)
	opts___
	
	] := 
	
	Module[
    
	    {
	    	i,
	    	randomValueInitialization,
			representativesIndexList,
			representatives
		},


		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];

		If[numberOfRepresentatives >= Length[inputs], 
			Return[inputs]
		];

		representativesIndexList = First[ 
			GetRandomRepresentativesIndexList[
				Table[i, {i, Length[inputs]}], 
				numberOfRepresentatives, 
				UtilityOptionRandomInitializationMode -> randomValueInitialization
			]];
		
		representatives = {};
		Do[
			AppendTo[representatives, inputs[[representativesIndexList[[i]]]]],
			
			{i, Length[representativesIndexList]}
		];
		
		Return[representatives]
	];

GetRandomRepresentativesIndexList[

	(* Returns random element list for list of integer numbers and the corresponding rest list.

	   Returns:
	   {randomList, restList} 
	   Join[randomList, restList] would yield integerNumberList *)


	(* List of integer numbers *)
	integerNumberList_/;VectorQ[integerNumberList, IntegerQ],
	
	numberOfRandomElements_?IntegerQ,
    
    (* Options *)
    opts___
      
	] :=
    
    Module[
      
		{
			randomValueInitialization,
			counter,
			restList,
			randomList,
			randomPosition
        },

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
        randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    
		If[randomValueInitialization == "Seed", 

			SeedRandom[1], 
			
			SeedRandom[]
		];

		If[numberOfRandomElements >= Length[integerNumberList],
			Return[integerNumberList]
		];

		restList = integerNumberList;
		randomList = {};
		counter = 1;
		While[counter <= numberOfRandomElements,
			randomPosition = RandomInteger[{1, Length[restList]}];
			AppendTo[randomList, restList[[randomPosition]]];
			restList = Drop[restList, {randomPosition}];
			counter++
		];

		Return[{randomList, restList}]
	];

GetRandomTrainingAndTestSet[

	(* Returns random training and test set of desired size.

	   Returns :
	   {trainingSet, testSet} 
	   trainingSet and testSet have the same structure as dataSet *)

	
    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...} *)
	dataSet_,
	
	(* 0.0 < trainingFraction <= 1.0 
	   trainingFraction = 1.0: Test set is empty *)
	trainingFraction_?NumberQ,
	
    (* Options *)
    opts___
	
	] := 
	
	Module[
    
	    {
	    	i,
			numberOfIoPairs,
			numberOfTrainingSetIoPairs,
			representativesIndexList,
			restIndexList,
			result,
			testSet,
			trainingSet,
			randomValueInitialization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
        randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];

		numberOfIoPairs = Length[dataSet];
		numberOfTrainingSetIoPairs = Floor[numberOfIoPairs*trainingFraction];
		If[numberOfTrainingSetIoPairs >= numberOfIoPairs, 
			Return[{dataSet, {}}]
		];
		
		result = 
			GetRandomRepresentativesIndexList[
				Table[i, {i, numberOfIoPairs}], 
				numberOfTrainingSetIoPairs,
				UtilityOptionRandomInitializationMode -> randomValueInitialization
			];
		representativesIndexList = result[[1]];
		restIndexList = result[[2]];

		trainingSet = {};
		Do[
			AppendTo[trainingSet, dataSet[[representativesIndexList[[i]]]]],
			
			{i, Length[representativesIndexList]}
		];
		
		testSet = {};
		If[Length[restIndexList] > 0,
			Do[
				AppendTo[testSet, dataSet[[restIndexList[[i]]]]],
				
				{i, Length[restIndexList]}
			]
		];
		
		Return[{trainingSet, testSet}]
	];

GetRepresentativesIndexList[

	(* Returns list with representatives in the sequence of clusters in clusterInfo.

	   Returns :
	   representativesIndexList: {index of representative of cluster 1, index of representative of cluster 2, ...} *)

	
	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],
	
  	(* See "Frequently used data structures" *)
	clusterInfo_
	
	] := 
	
	Module[
    
	    {
	    	infoDataStructure,
	    	sortedClusteringResult,
	    	centroidVector,
	    	currentDistance,
	    	currentIndexList,
	    	i,
	    	k,
	    	minimumDistance,
	    	minimumDistanceIndex,
	    	minimumPosition,
			numberOfClusters,
			representativesIndexList,
			singleCluster
		},

		(* NOTE: This method is applicable for all clustering methods *)
		infoDataStructure = clusterInfo[[2]];
		sortedClusteringResult = infoDataStructure[[3]];
		numberOfClusters = Length[sortedClusteringResult];

		representativesIndexList = {};
		Do[
			singleCluster = sortedClusteringResult[[i]];
			currentIndexList = singleCluster[[2]];
			centroidVector = singleCluster[[3, 2]];
			minimumPosition = 1;
			minimumDistanceIndex = 1;
			minimumDistance = Infinity;
			Do[
				currentDistance = EuclideanDistance[centroidVector, inputs[[currentIndexList[[k]]]]];
				If[currentDistance < minimumDistance,
					minimumDistance = currentDistance;
					minimumDistanceIndex = currentIndexList[[k]];
					minimumPosition = k
				],
				
				{k, Length[currentIndexList]}
			];
			AppendTo[representativesIndexList, minimumDistanceIndex],
			
			{i, numberOfClusters}
		];
		
		Return[representativesIndexList]
	];

GetSilhouettePlotPoints[

	(* Returns silhouette plot points.
	
	   Returns:
	   {plotPoint1, plotPoint2, ...}
	   plotPoint: {number of clusters, mean of corresponding silhouette widths} *)

	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],

  	(* Minimum number of clusters *)
	minimumNumberOfClusters_?IntegerQ,

  	(* Maximum number of clusters *)
	maximumNumberOfClusters_?IntegerQ,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	maximumNumberOfTrialSteps,
	    	scalarProductMinimumTreshold,
	    	randomValueInitialization,
	    	targetInterval,
	    	parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
   	    (* Parallelization options *)
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				GetSilhouettePlotPointsPC[
					inputs,
					minimumNumberOfClusters,
					maximumNumberOfClusters,
					ClusterOptionMethod -> clusterMethod,
					ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
					ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
					ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					DataTransformationOptionTargetInterval -> targetInterval
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				GetSilhouettePlotPointsSC[
					inputs,
					minimumNumberOfClusters,
					maximumNumberOfClusters,
					ClusterOptionMethod -> clusterMethod,
					ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
					ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
					ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					DataTransformationOptionTargetInterval -> targetInterval
				]
			]
		]
	];

GetSilhouettePlotPointsSC[

	(* Returns silhouette plot points.
	
	   Returns:
	   {plotPoint1, plotPoint2, ...}
	   plotPoint: {number of clusters, mean of corresponding silhouette widths} *)

	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],

  	(* Minimum number of clusters *)
	minimumNumberOfClusters_?IntegerQ,

  	(* Maximum number of clusters *)
	maximumNumberOfClusters_?IntegerQ,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	maximumNumberOfTrialSteps,
	    	scalarProductMinimumTreshold,
	    	randomValueInitialization,
	    	targetInterval,
	    	clusterInfo,
	    	i,
	    	silhouetteStatistics
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		Return[
			Table[
				clusterInfo = 
					GetFixedNumberOfClusters[
						inputs,
						i,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
						DataTransformationOptionTargetInterval -> targetInterval
					];
				silhouetteStatistics = GetSilhouetteStatistics[inputs, clusterInfo];
				{i, silhouetteStatistics[[2]]},
				
				{i, minimumNumberOfClusters, maximumNumberOfClusters}
			]
		]
	];
	
GetSilhouettePlotPointsPC[

	(* Returns silhouette plot points.
	
	   Returns:
	   {plotPoint1, plotPoint2, ...}
	   plotPoint: {number of clusters, mean of corresponding silhouette widths} *)

	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],

  	(* Minimum number of clusters *)
	minimumNumberOfClusters_?IntegerQ,

  	(* Maximum number of clusters *)
	maximumNumberOfClusters_?IntegerQ,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	maximumNumberOfTrialSteps,
	    	scalarProductMinimumTreshold,
	    	randomValueInitialization,
	    	targetInterval,
	    	clusterInfo,
	    	i,
	    	silhouetteStatistics
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		ParallelNeeds[{"CIP`Cluster`", "CIP`DataTransformation`", "CIP`Utility`", "Combinatorica`"}];
		DistributeDefinitions[clusterMethod, maximumNumberOfEpochs, scalarProductMinimumTreshold, maximumNumberOfTrialSteps, randomValueInitialization, targetInterval];
		
		
		Return[
			ParallelTable[
				clusterInfo = 
					GetFixedNumberOfClusters[
						inputs,
						i,
						ClusterOptionMethod -> clusterMethod,
						ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
						ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
						ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
						UtilityOptionRandomInitializationMode -> randomValueInitialization,
						DataTransformationOptionTargetInterval -> targetInterval
					];
				silhouetteStatistics = GetSilhouetteStatistics[inputs, clusterInfo];
				{i, silhouetteStatistics[[2]]},
				
				{i, minimumNumberOfClusters, maximumNumberOfClusters}
			]
		]
	];

GetSilhouetteStatistics[

	(* Returns silhouette statistics.
	
	   Returns:
	   {Minimum silhouette with, mean silhouette with, maximum silhouette with} *)

	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],

  	(* See "Frequently used data structures" *)
	clusterInfo_
    
	] :=
  
	Module[
    
	    {
	    	silhouetteWidths
		},

		silhouetteWidths = Flatten[GetSilhouetteWidthsForClusters[inputs, clusterInfo]];
		Return[
			{
				Min[silhouetteWidths],
				Mean[silhouetteWidths],
				Max[silhouetteWidths]
			}
		]
	];

GetSilhouetteStatisticsForClusters[

	(* Returns silhouette statistics for clusters.
	
	   Returns:
	   {Silhouette statistics for cluster 1, Silhouette statistics for cluster 2, ...}
	   Silhouette statistics for cluster: {Statistics for cluster, Silhouette widths for cluster}
	   Statistics for cluster: {Minimum silhouette width of cluster, mean silhouette width of cluster, maximum silhouette width of cluster} 
	   Silhouette widths for cluster (NOT sorted): {Silhouette width 1, Silhouette width 2, ...} *)

	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],

  	(* See "Frequently used data structures" *)
	clusterInfo_
    
	] :=
  
	Module[
    
	    {
	    	i,
	    	silhouetteWidthsForClusters
		},

		silhouetteWidthsForClusters = GetSilhouetteWidthsForClusters[inputs, clusterInfo];
		Return[
			Table[
				{
					{
						N[CIP`Utility`RoundTo[Min[silhouetteWidthsForClusters[[i]]], 2]],
						N[CIP`Utility`RoundTo[Mean[silhouetteWidthsForClusters[[i]]], 2]],
						N[CIP`Utility`RoundTo[Max[silhouetteWidthsForClusters[[i]]], 2]]
					},
					silhouetteWidthsForClusters[[i]]
				},
			
				{i, Length[silhouetteWidthsForClusters]}
			]
		]
	];

GetSilhouetteWidthsForClusters[

	(* Returns silhouette widths for all input vectors of all clusters of clusterInfo.
	
	   Returns:
	   {silhouette widths for cluster 1, silhouette widths for cluster 1, ...}
	   silhouette widths for cluster: {silhouette width for input 1 of cluster, silhouette width for input 1 of cluster,, ...} *)

	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],

  	(* See "Frequently used data structures" *)
	clusterInfo_,

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	a,
	    	b,
	    	i,
	    	j,
	    	k,
	    	l,
	    	infoDataStructure,
	    	numberOfClusters,
	    	distanceMatrix,
	    	sortedClusteringResult,
	    	indexLists,
	    	singleCluster,
	    	silhouetteWidths,
	    	clusterSilhouetteWidths,
	    	createDistanceMatrix
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		createDistanceMatrix = ClusterOptionCreateDistanceMatrix/.{opts}/.Options[ClusterOptionsCalculation];

		(* NOTE: This method is applicable for all clustering methods *)
		infoDataStructure = clusterInfo[[2]];
		numberOfClusters = infoDataStructure[[2]];
		sortedClusteringResult = infoDataStructure[[3]];

		(* Set indexLists: indexLists[[i]] = index list of cluster i *)
		indexLists = 
			Table[
				singleCluster = sortedClusteringResult[[i]];
				singleCluster[[2]],
				{i, numberOfClusters}
			];

		(* Set distance matrix if specified *)
		If[createDistanceMatrix,
			distanceMatrix = 
				Table[
					0.0,
					{i, Length[inputs]}, {k, Length[inputs]}
				];
			Do[
				Do[
					distanceMatrix[[i, k]] = EuclideanDistance[inputs[[i]], inputs[[k]]];
					distanceMatrix[[k, i]] = distanceMatrix[[i, k]],
					{k, 1, i}
				],
				{i, 2, Length[inputs]}
			]
		];

		(* Set silhouette widths *)
		silhouetteWidths = {};
		Do[
			clusterSilhouetteWidths =
				Table[
					If[createDistanceMatrix,

						a = Mean[
								Table[
									distanceMatrix[[indexLists[[i, j]], indexLists[[i, l]]]],
									
									{l, Length[indexLists[[i]]]}
								]
							],
							
						a = Mean[
								Table[
									EuclideanDistance[inputs[[indexLists[[i, j]]]], inputs[[indexLists[[i, l]]]]],
									
									{l, Length[indexLists[[i]]]}
								]
							]
					];
					b = Infinity;
					Do[
						If[k != i,
							b = 
								Min[
									b,
									If[createDistanceMatrix,
									 
										Mean[
											Table[
												distanceMatrix[[indexLists[[i, j]], indexLists[[k, l]]]],
												
												{l, Length[indexLists[[k]]]}
											]
										],
										
										Mean[
											Table[
												EuclideanDistance[inputs[[indexLists[[i, j]]]], inputs[[indexLists[[k, l]]]]],
												
												{l, Length[indexLists[[k]]]}
											]
										]
									]
								]
						],
						
						{k, numberOfClusters}
					];
					If[a < b,
						
						1.0 - a/b,
						
						If[b < a,
							
							b/a - 1.0,
							
							0.0
						]
					],
					
					{j, Length[indexLists[[i]]]}
				];
			AppendTo[silhouetteWidths, clusterSilhouetteWidths],
			
			{i, numberOfClusters}
		];

		Return[silhouetteWidths]
	];

GetSingleClusterProperty[

	(* Returns named cluster property.
	
	   Returns:
	   Named cluster property *)

	(* Properties, full list: 
	   {
	       "NumberOfInputVectors",
		   "NumberOfClusters",
		   "CentroidVectors"
	    } *)
 	namedProperty_,

  	(* See "Frequently used data structures" *)
	clusterInfo_
    
	] :=
  
	Module[
    
	    {
	    	infoDataStructure
		},

		infoDataStructure = clusterInfo[[2]];

		Switch[namedProperty,
			
			(* -------------------------------------------------------------------------------- *)
			"NumberOfInputVectors",
			Return[infoDataStructure[[1]]],
			
			(* -------------------------------------------------------------------------------- *)
			"NumberOfClusters",
			Return[infoDataStructure[[2]]],

			(* -------------------------------------------------------------------------------- *)
			"CentroidVectors",
			Return[GetCentroidVectors[clusterInfo]]
		]
	];

GetVigilanceParameterScan[

	(* Returns scan of vigilance parameter
	
	   Returns:
	   art2aScanInfo: {numberOfInputVectors, dimensionOfInputVector, scanPoints}
	   scanPoints: {{vigilanceParameter1, numberOfClusters1}, ...} *)


	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsInVectorOfDataMatrix>}
	   NOTE : component >= 0
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],

    (* Minimum vigilance parameter (> 0) *)
	minimumVigilanceParameter_?NumberQ, 
	
    (* Maximum vigilance parameter (< 1) *)
	maximumVigilanceParameter_?NumberQ,
	
    (* Number of scan points *)
    numberOfScanPoints_?IntegerQ,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	scanPoints,
	    	maximumNumberOfEpochs,
	    	numberOfClusters,
	    	scalarProductMinimumTreshold,
	    	randomValueInitialization,
	    	stepSize,
			targetInterval,
	    	vigilanceParameter
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		stepSize = (maximumVigilanceParameter - minimumVigilanceParameter)/(numberOfScanPoints - 1);
		vigilanceParameter = minimumVigilanceParameter;
		scanPoints = {};
		While[vigilanceParameter <= maximumVigilanceParameter,
			numberOfClusters = GetClustersWithArt2a[
				inputs,
				ClusterOptionVigilanceParameter -> vigilanceParameter,
				ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
				ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
				ClusterOptionIsScan-> True,
				UtilityOptionRandomInitializationMode -> randomValueInitialization,
				DataTransformationOptionTargetInterval -> targetInterval
			];
			AppendTo[scanPoints, {vigilanceParameter, numberOfClusters}];
			vigilanceParameter += stepSize
		];
		
		Return[
			{
				Length[inputs],
				Length[inputs[[1]]],
				scanPoints
			}
		]
	];

GetWhiteSpots[

	(* Returns white spots of specified cluster, i.e. if difference in occupancy of specified cluster is less than/equal to threshold to all other clusters.
	
	   Returns:
	   White Spots: {White spot cluster index 1, White spot cluster index 1, ...} *)


	(* clusterOccupancies: {clusterOccupancy1, clusterOccupancy1, ..., clusterOccupancy1<numberOfClusters>} 
	   clusterOccupancy: {Percent for inputs1, percent for inputs2, ..., percent for inputs<Length[inputsList]>} *)
    clusterOccupancies_/;MatrixQ[clusterOccupancies, NumberQ],
    
    (* Index of inputs for white spot detection *)
    inputsIndex_?IntegerQ,
    
    (* Threshold in percent for white spot detection *)
    threshold_?NumberQ
    
	] :=
  
	Module[
    
	    {
	    	differenceInPercent,
	    	i,
	    	k,
	    	singleClusterOccupancy,
	    	percentOfSpecifiedInputs,
	    	whiteSpots
		},

		whiteSpots = {};
		Do[
			singleClusterOccupancy = clusterOccupancies[[i]];
			percentOfSpecifiedInputs = singleClusterOccupancy[[inputsIndex]];
			Do[
				If[k != inputsIndex,
					differenceInPercent = singleClusterOccupancy[[k]] - percentOfSpecifiedInputs;
					If[differenceInPercent > 0.0,
						If[100.0 - percentOfSpecifiedInputs/differenceInPercent*100.0 >= threshold,
							AppendTo[whiteSpots, i];
							Break[]
						]
					]
				],
				
				{k, Length[singleClusterOccupancy]}
			],
			
			{i, Length[clusterOccupancies]}
		];
		
		Return[whiteSpots]
	];

HasConverged[

	(* Returns if ART-2a clustering converged.

	   Returns :
	   True: Convergence achieved, False: Otherwise *)

	
	(* Matrix with ART2a centroid vectors of clusters of this epoch *)
	art2aCentroidVectors_,
	
	(* Matrix with ART2a centroid vectors of clusters of previous epoch *)
	art2aCentroidVectorsOld_,
	
	(* Minimum treshold for value of scalar product: 0 (orthogonal unit vectors) < scalarProductMinimumTreshold <= 1.0 (parallel unit vectors) *)
	scalarProductMinimumTreshold_
	
	] := 
	
	Module[
    
	    {
	    	i
		},

		If[Length[art2aCentroidVectorsOld] == 0, 
			Return[False]
		];
		
		If[Length[art2aCentroidVectorsOld] != Length[art2aCentroidVectors], 
			Return[False]
		];

		Do[
			(* Check for empty clusters first *)
			If[Length[art2aCentroidVectors[[i]]] > 0 && Length[art2aCentroidVectorsOld[[i]]] > 0,
				
				If[art2aCentroidVectors[[i]].art2aCentroidVectorsOld[[i]] < scalarProductMinimumTreshold,
					Return[False]
				],
				
				Return[False]
			],
			
			{i, Length[art2aCentroidVectors]}
		];
		
		Return[True]		
	];

ScanClassTrainingWithCluster[

	(* Scans training and test set for different training fractions based on method FitCluster, see code.
	
	   Returns:
	   clusterClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, clusterInfo1 with classificationCentroids}, {trainingAndTestSet2, clusterInfo2 with classificationCentroids}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, classification result in percent for training set}, {trainingFraction, classification result in percent for test set}}
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
	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
	    	targetInterval,
			parallelization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];
   	    (* Parallelization options *)
		parallelization = UtilityOptionCalculationMode/.{opts}/.Options[UtilityOptionsParallelization];
		
		Switch[parallelization,
			
			(* ------------------------------------------------------------------------------- *)
			"ParallelCalculation",
			Return[
				ScanClassTrainingWithClusterPC[
					dataSet,
					trainingFractionList,
					ClusterOptionMethod -> clusterMethod,
					ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
					ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
					ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					DataTransformationOptionTargetInterval -> targetInterval
				]
			],
			
			(* ------------------------------------------------------------------------------- *)
			"SequentialCalculation",
			Return[
				ScanClassTrainingWithClusterSC[
					dataSet,
					trainingFractionList,
					ClusterOptionMethod -> clusterMethod,
					ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
					ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
					ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					DataTransformationOptionTargetInterval -> targetInterval
				]
			]
		]
	];
	
ScanClassTrainingWithClusterSC[

	(* Scans training and test set for different training fractions based on method FitCluster, see code.
	
	   Returns:
	   clusterClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, clusterInfo1 with classificationCentroids}, {trainingAndTestSet2, clusterInfo2 with classificationCentroids}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, classification result in percent for training set}, {trainingFraction, classification result in percent for test set}}
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
			i,
	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
	    	targetInterval,
			scanReport,
			trainingAndTestSetsInfo,
			currentTrainingAndTestSet,
			currentTrainingSet,
			currentTestSet,
			currentClusterInfo,
			pureFunction,
			trainingSetCorrectClassificationInPercent,
			testSetCorrectClassificationInPercent
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		scanReport = {};
		trainingAndTestSetsInfo = {};
		Do[
			currentTrainingAndTestSet = 
				GetClusterBasedTrainingAndTestSet[
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
			currentClusterInfo = 
				FitCluster[
					currentTrainingSet,
					ClusterOptionMethod -> clusterMethod,
					ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
					ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
					ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					DataTransformationOptionTargetInterval -> targetInterval
				];
			pureFunction = Function[inputs, CalculateClusterClassNumbers[inputs, currentClusterInfo]];
			trainingSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTrainingSet, pureFunction];
			testSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTestSet, pureFunction];
			AppendTo[trainingAndTestSetsInfo, {currentTrainingAndTestSet, currentClusterInfo}];
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

ScanClassTrainingWithClusterPC[

	(* Scans training and test set for different training fractions based on method FitCluster, see code.
	
	   Returns:
	   clusterClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, clusterInfo1 with classificationCentroids}, {trainingAndTestSet2, clusterInfo2 with classificationCentroids}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, classification result in percent for training set}, {trainingFraction, classification result in percent for test set}}
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
			i,
	    	clusterMethod,
	    	maximumNumberOfEpochs,
	    	scalarProductMinimumTreshold,
	    	maximumNumberOfTrialSteps,
	    	randomValueInitialization,
	    	targetInterval,
			scanReport,
			trainingAndTestSetsInfo,
			currentTrainingAndTestSet,
			currentTrainingSet,
			currentTestSet,
			currentClusterInfo,
			pureFunction,
			trainingSetCorrectClassificationInPercent,
			testSetCorrectClassificationInPercent,
			listOfTrainingAndTestSetsInfoAndScanReport
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		clusterMethod = ClusterOptionMethod/.{opts}/.Options[ClusterOptionsMethod];
	    maximumNumberOfEpochs = ClusterOptionMaximumNumberOfEpochs/.{opts}/.Options[ClusterOptionsArt2a];
	    scalarProductMinimumTreshold = ClusterOptionScalarProductMinimumTreshold/.{opts}/.Options[ClusterOptionsArt2a];
	    maximumNumberOfTrialSteps = ClusterOptionMaximumTrialSteps/.{opts}/.Options[ClusterOptionsArt2a];
	    randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
   	    targetInterval = DataTransformationOptionTargetInterval/.{opts}/.Options[DataTransformationOptions];

		ParallelNeeds[{"CIP`Cluster`", "CIP`DataTransformation`", "CIP`Utility`", "Combinatorica`"}];
		DistributeDefinitions[clusterMethod, maximumNumberOfEpochs, scalarProductMinimumTreshold, maximumNumberOfTrialSteps, randomValueInitialization, targetInterval];
		
		listOfTrainingAndTestSetsInfoAndScanReport = ParallelTable[
			currentTrainingAndTestSet = 
				GetClusterBasedTrainingAndTestSet[
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
			currentClusterInfo = 
				FitCluster[
					currentTrainingSet,
					ClusterOptionMethod -> clusterMethod,
					ClusterOptionMaximumNumberOfEpochs -> maximumNumberOfEpochs,
					ClusterOptionScalarProductMinimumTreshold -> scalarProductMinimumTreshold,
					ClusterOptionMaximumTrialSteps -> maximumNumberOfTrialSteps,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					DataTransformationOptionTargetInterval -> targetInterval
				];
			pureFunction = Function[inputs, CalculateClusterClassNumbers[inputs, currentClusterInfo]];
			trainingSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTrainingSet, pureFunction];
			testSetCorrectClassificationInPercent = CIP`Utility`GetCorrectClassificationInPercent[currentTestSet, pureFunction];
			{
				{currentTrainingAndTestSet, currentClusterInfo},
			 
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
	
ShowClusterClassificationResult[

	(* Shows result of clustering based classification for training and test set according to named property list.

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
	clusterInfo_,
    
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
		ShowClusterSingleClassification[
			namedPropertyList,
			trainingSet, 
			clusterInfo,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionMinMaxIndex -> minMaxIndex
		];
		
		(* Analyze test set *)
		If[Length[testSet] > 0,
			Print["Test Set:"];
			ShowClusterSingleClassification[
				namedPropertyList,
				testSet, 
				clusterInfo,
				GraphicsOptionImageSize -> imageSize,
				GraphicsOptionMinMaxIndex -> minMaxIndex
			];
		]
	];

ShowClusterSingleClassification[

	(* Shows result of clustering based classification for data set according to named property list.

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
	clusterInfo_,
    
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

   		pureFunction = Function[inputs, CalculateClusterClassNumbers[inputs, clusterInfo]];
		CIP`Graphics`ShowClassificationResult[
			namedPropertyList,
			dataSet, 
			pureFunction,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionMinMaxIndex -> minMaxIndex
		]
	];

ShowClusterClassificationScan[

	(* Shows result of clustering based classification scan of clustered training sets.

	   Returns: Nothing *)


	(* clusterClassificationScan: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, clusterInfo1 with classificationCentroids}, {trainingAndTestSet2, clusterInfo2 with classificationCentroids}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, classification result in percent for training set}, {trainingFraction, classification result in percent for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)
	clusterClassificationScan_,
	
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
			clusterClassificationScan,
			GraphicsOptionImageSize -> imageSize,
			GraphicsOptionDisplayFunction -> displayFunction
		]
	];

ShowClusterResult[

	(* Shows clustering results according to named property list.
	
	   Returns:
	   Nothing *)


	(* Properties, full list: 
	   {
	       "NumberOfInputVectors",
		   "NumberOfClusters",
	       "VigilanceParameter",
	       "ART2aDistanceDiagram",
	       "EuclideanDistanceDiagram",
	       "ClusterStatistics"
	       "ART2aClusterStatistics"    
	    } *)
 	namedPropertyList_,

  	(* See "Frequently used data structures" *)
	clusterInfo_
    
	] :=

	Module[
    
    	{
    		i,
    		namedProperty
    	},
    	
    	Do[
    		namedProperty = namedPropertyList[[i]];
    		ShowSingleClusterResult[namedProperty, clusterInfo],
    		
    		{i, Length[namedPropertyList]}
    	]
	];

ShowClusterOccupancies[

	(* Shows cluster occupancies.
	
	   Returns:
	   Nothing *)


	(* clusterOccupancies: {clusterOccupancy1, clusterOccupancy1, ..., clusterOccupancy1<numberOfClusters>} 
	   clusterOccupancy: {Percent for inputs1, percent for inputs2, ..., percent for inputs<Length[inputsList]>} *)
    clusterOccupancies_/;MatrixQ[clusterOccupancies, NumberQ],

	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	i,
	    	labels,
	    	yLabel,
	    	plotLabel,
	    	chartLabels,
	    	singleBarLabels,
	    	barGroupLabels,
	    	maxClusterIndex,
	    	minClusterIndex,
	    	minMaxIndex
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    minMaxIndex = GraphicsOptionMinMaxIndex/.{opts}/.Options[GraphicsOptionsIndex];

		If[Length[minMaxIndex] == 0,
			
			minClusterIndex = 1;
			maxClusterIndex = Length[clusterOccupancies],
			
			minClusterIndex = minMaxIndex[[1]];
			maxClusterIndex = minMaxIndex[[2]]
		];

		yLabel = "%";
		plotLabel = "Cluster occupancies";
		singleBarLabels = 
			Table[
				ToString[i],
				
				{i, Length[clusterOccupancies[[1]]]}
			];
		barGroupLabels = 
			Table[
				StringJoin["Cluster ", ToString[i]],
				
				{i, minClusterIndex, maxClusterIndex, 1}
			];
		chartLabels = {singleBarLabels, barGroupLabels};
		labels = {chartLabels, yLabel, plotLabel};
		
		Print[
			CIP`Graphics`PlotGroupedBarChart[
				Take[clusterOccupancies, {minClusterIndex, maxClusterIndex}],
				labels
			]
		]
	];
	
ShowComponentStatistics[

	(* Shows statistics for specified component of input vectors.
	
	   Returns: Nothing *)


	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>}
	   All vectors in inputs must have the same length *)
    inputs_/;MatrixQ[inputs, NumberQ],

    (* List of indices of components of vectors in inputs *)
	indexOfComponentList_/;VectorQ[indexOfComponentList, IntegerQ],
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	argumentRange2D,
	    	componentStatistics,
	    	displayFunction,
	    	i,
	    	labels,
	    	numberOfIntervals,
			functionValueRange2D,
	    	pointColor,
	    	pointSize
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    numberOfIntervals = ClusterOptionNumberOfIntervals/.{opts}/.Options[ClusterOptionsComponentStatistics];
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	    argumentRange2D = GraphicsOptionArgumentRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];
	    functionValueRange2D = GraphicsOptionFunctionValueRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];

		Do[
			componentStatistics = 
				GetComponentStatistics[
					inputs, 
					indexOfComponentList[[i]],
					ClusterOptionNumberOfIntervals -> numberOfIntervals
				];
			labels = 
				{
					"Component value",
					"Percentage in interval",
					StringJoin["In ", ToString[indexOfComponentList[[i]]], " : Distribution approximated by ", ToString[componentStatistics[[1]]], " intervals"]
				};
			Print[
				CIP`Graphics`Plot2dLineWithOptionalPoints[
					componentStatistics[[2]], 
					componentStatistics[[2]], 
					labels,
					GraphicsOptionDisplayFunction -> displayFunction,
					GraphicsOptionPointSize -> pointSize,
					GraphicsOptionPointColor -> pointColor,
					GraphicsOptionArgumentRange2D -> argumentRange2D,
					GraphicsOptionFunctionValueRange2D -> functionValueRange2D
				]
			];
			Print["Min  / Max    = ", ScientificForm[componentStatistics[[3, 1]], 3], " / ", ScientificForm[componentStatistics[[3, 2]], 3]];
			Print["Mean / Median = ", ScientificForm[componentStatistics[[3, 3]], 3], " / ", ScientificForm[componentStatistics[[3, 4]], 3]],
			
			{i, Length[indexOfComponentList]}
		]
	];

ShowSilhouettePlot[

	(* Shows silhouette plot.
	
	   Returns:
	   Nothing *)


	(* Silhouette plot points *)
 	silhouettePlotPoints2D_
 	    
	] :=

	Module[
    
    	{
    		labels
    	},
    	
    	labels = {"Number of clusters", "Mean silhouette width", "Silhouette plot"};
    	Print[
    		CIP`Graphics`Plot2dLineWithOptionalPoints[silhouettePlotPoints2D, silhouettePlotPoints2D, labels]
    	];
	];

ShowSilhouetteWidthsForCluster[

	(* Shows silhouette widths for specified cluster.
	
	   Returns:
	   Nothing *)


	(* {Silhouette statistics for cluster 1, Silhouette statistics for cluster 2, ...}
	   Silhouette statistics for cluster: {Statistics for cluster, Silhouette widths for cluster}
	   Statistics for cluster: {Minimum silhouette width of cluster, mean silhouette width of cluster, maximum silhouette width of cluster} 
	   Silhouette widths for cluster (NOT sorted): {Silhouette width 1, Silhouette width 2, ...} *)
    silhouetteStatisticsForClusters_,

	(* Index of cluster to be shown *)
	indexOfCluster_?IntegerQ
	    
	] :=
  
	Module[
    
	    {
	    	labels
		},

		labels = 
			{
				"Sorted index of input", 
				"Silhouette width", 
				StringJoin["Cluster ", ToString[indexOfCluster], " with mean ", ToString[silhouetteStatisticsForClusters[[indexOfCluster, 1, 2]]]]
			};

		Print[
			CIP`Graphics`PlotSilhouetteWidths[
				silhouetteStatisticsForClusters[[indexOfCluster, 2]], 
				silhouetteStatisticsForClusters[[indexOfCluster, 1, 2]], 
				labels,
				GraphicsOptionPointSize -> 0.01
			]
		]
	];

ShowSingleClusterResult[

	(* Shows single clustering result according to named property.
	
	   Returns:
	   Nothing *)


	(* Properties, full list: 
	   {
	       "NumberOfInputVectors",
		   "NumberOfClusters",
	       "VigilanceParameter",
	       "ART2aDistanceDiagram",
	       "EuclideanDistanceDiagram",
	       "ClusterStatistics"
	       "ART2aClusterStatistics"       
	    } *)
 	namedProperty_,

  	(* See "Frequently used data structures" *)
	clusterInfo_
    
	] :=
  
	Module[
    
	    {
	    	art2aInfo,
	    	clusterMethod,
	    	singleCluster,
			i,
			infoDataStructure,
			numberOfClusters,
			numberOfInputVectors,
			points,
			sortedClusteringResult,
			vigilanceParameter
		},

		clusterMethod = clusterInfo[[1]];
		infoDataStructure = clusterInfo[[2]];

		numberOfInputVectors = infoDataStructure[[1]];
		numberOfClusters = infoDataStructure[[2]];
		sortedClusteringResult = infoDataStructure[[3]];

		Switch[namedProperty,
			
			(* -------------------------------------------------------------------------------- *)
			"NumberOfInputVectors",
			Print["Number of inputs = ", numberOfInputVectors],

			(* -------------------------------------------------------------------------------- *)
			"NumberOfClusters",
			Print["Number of clusters = ", numberOfClusters],
			
			(* -------------------------------------------------------------------------------- *)
			"VigilanceParameter",
			If[clusterMethod == "ART2a",
				art2aInfo = infoDataStructure;
				vigilanceParameter = art2aInfo[[4]];
				Print["Vigilance parameter = ", vigilanceParameter]
			],
			
			(* -------------------------------------------------------------------------------- *)
			"ART2aDistanceDiagram",
			If[clusterMethod == "ART2a",
				points = {};
				Do[
					singleCluster = sortedClusteringResult[[i]];
					AppendTo[points, {singleCluster[[4, 1]], singleCluster[[1]]}],
								
					{i, Length[sortedClusteringResult]}
				];
				Print[
					CIP`Graphics`Plot2dLineWithOptionalPointsAndMaximumXValue[
					    points,
					    points,
					    90.0,
					    {
					    	"Angle between clusters (0: Identical, 90: Orthogonal)", 
					    	"Percentage in cluster", 
					    	StringJoin[
					    		ToString[numberOfClusters], 
					    		" clusters for ",
					    		ToString[numberOfInputVectors],
					    		" inputs"
					    	]
					    }
					]
				]
			],
			
			(* -------------------------------------------------------------------------------- *)
			"EuclideanDistanceDiagram",
			points = {};
			Do[
				singleCluster = sortedClusteringResult[[i]];
				AppendTo[points, {singleCluster[[3, 1]], singleCluster[[1]]}],
							
				{i, Length[sortedClusteringResult]}
			];
			Print[
				CIP`Graphics`Plot2dLineWithOptionalPoints[
				    points,
				    points,
				    {
				    	"Euclidean distance", 
				    	"Percentage in cluster", 
				    	StringJoin[
				    		ToString[numberOfClusters], 
				    		" clusters for ",
				    		ToString[numberOfInputVectors],
				    		" inputs"
				    	]
				    }
				]
			],
			
			(* -------------------------------------------------------------------------------- *)
			"ClusterStatistics",
			Do[
				singleCluster = sortedClusteringResult[[i]];
				Print["Cluster ", i, " : ", Length[singleCluster[[2]]], " members (", singleCluster[[1]],"%) with distance = ", singleCluster[[3, 1]]],
							
				{i, Length[sortedClusteringResult]}
			],
			
			
			(* -------------------------------------------------------------------------------- *)
			"ART2aClusterStatistics",
			Do[
				singleCluster = sortedClusteringResult[[i]];
				Print["Cluster ", i, " : ", Length[singleCluster[[2]]], " members (", singleCluster[[1]],"%) with angle = ", singleCluster[[4, 1]]],
							
				{i, Length[sortedClusteringResult]}
			]
		]
    ];

ShowVigilanceParameterScan[

	(* Shows scan of vigilance parameter.
	
	   Returns:
	   Nothing *)


	(* art2aScanInfo: {numberOfInputVectors, dimensionOfInputVector, scanPoints}
	   scanPoints: {{vigilanceParameter1, numberOfClusters1}, ...} *)
    art2aScanInfo_
    
	] :=
  
	Module[
    
	    {
	    	dimensionOfInputVector,
		    numberOfInputVectors,
		    scanPoints
		},

		numberOfInputVectors = art2aScanInfo[[1]];
		dimensionOfInputVector = art2aScanInfo[[2]];
		scanPoints = art2aScanInfo[[3]];

		Print[
			CIP`Graphics`Plot2dLineWithOptionalPoints[
				scanPoints,
				scanPoints,
				{
					"Vigilance parameter", 
					"Number of detected clusters", 
					StringJoin[ToString[numberOfInputVectors]," inputs with ", ToString[dimensionOfInputVector], " dimensions"]
				}
			]
		]
	];

(* ::Section:: *)
(* End of Package *)

End[]

EndPackage[]
