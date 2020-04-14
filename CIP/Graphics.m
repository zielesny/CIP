(*
-----------------------------------------------------------------------
Computational Intelligence Packages (CIP): Package Graphics
Version 3.1 for Mathematica 11 or higher
-----------------------------------------------------------------------

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
-----------------------------------------------------------------------
*)

(* ::Section:: *)
(* Package and dependencies *)

BeginPackage["CIP`Graphics`", {"CIP`Utility`", "CIP`DataTransformation`"}]

(* ::Section:: *)
(* Off settings *)

Off[General::"spell1"]
Off[General::shdw]
Off[General::compat]

(* ::Section:: *)
(* Options *)

Options[GraphicsOptionsLinePlotStyle] = 
{
	(* Plot style *)
	GraphicsOptionLinePlotStyle -> {Thickness[0.005], Black}
}

Options[GraphicsOptionsResidualsDistribution] = 
{
    (* Number of intervals for frequency percentages *)
	GraphicsOptionNumberOfIntervals -> 20
};

Options[GraphicsOptionsImageSize] = 
{
	(* Image size *)
	GraphicsOptionImageSize -> 300
}

Options[GraphicsOptionsDisplayFunction] = 
{
	(* DisplayFunction *)
	GraphicsOptionDisplayFunction -> $DisplayFunction
}

Options[GraphicsOptionsGraphics3D] = 
{
	(* NumberOfPlotPoints (in one dimension) *)
	GraphicsOptionNumberOfPlotPoints -> 50,
	
	(* Color function *)
	GraphicsOptionColorFunction -> Automatic,

	(* True: Mesh, False: No mesh *)
	GraphicsOptionIsMesh -> False,
	
	(* Mesh style *)
	GraphicsOptionMeshStyle -> RGBColor[0, 0.7, 1],

	(* Plot style *)
	GraphicsOptionPlotStyle3D -> Directive[Green, Specularity[White, 40], Opacity[0.8]],
	(* Alternative settings:
	   GraphicsOptionPlotStyle3D -> Directive[Green, Specularity[White, 20], Opacity[0.8]]
	   GraphicsOptionPlotStyle3D -> Directive[Green, Specularity[White, 40], Opacity[0.8]]
	   GraphicsOptionPlotStyle3D -> Directive[Red, Specularity[White, 40], Opacity[0.8]]
	   GraphicsOptionPlotStyle3D -> Directive[Red, Opacity[0.8]]
	*)
	
	(* Region function *)
	GraphicsOptionRegionFunction -> True,
	
	(* View Point *)
	GraphicsOptionViewPoint3D -> {1.3, -2.4, 2.0}
}

Options[GraphicsOptionsPlotRange2D] = 
{
	(* Argument range for 2D graphics *)
	GraphicsOptionArgumentRange2D -> {},

	(* Plot range for 2D graphics *)
	GraphicsOptionFunctionValueRange2D -> {}
}

Options[GraphicsOptionsPlotRange3D] = 
{
	(* Range for argument 1 *)
	GraphicsOptionArgument1Range3D -> {},

	(* Range for argument 2 *)
	GraphicsOptionArgument2Range3D -> {}
}

Options[GraphicsOptionsPoint] = 
{
	(* Option for point size *)
	GraphicsOptionPointSize -> 0.03,
	
	(* Option for point color *)
	GraphicsOptionPointColor -> RGBColor[0, 0, 1, 0.8],
	
	(* Number of points for display *)
	GraphicsOptionNumberOfPointsForDisplay -> 100
}

Options[GraphicsOptionsIndex] = 
{
	(* GraphicsOptionMinMaxIndex[[1]]: First index
 	   GraphicsOptionMinMaxIndex[[2]]: Last index
 	   Default is no setting: All *)
	GraphicsOptionMinMaxIndex -> {}
}

(* ::Section:: *)
(* Declarations *)

GetClassRelevantComponents::usage = 
    "GetClassRelevantComponents[inputComponentRelevanceListForClassification, numberOfComponents]"

GetRegressRelevantComponents::usage = 
    "GetRegressRelevantComponents[inputComponentRelevanceListForRegression, numberOfComponents]"

GetSingleRegressionResult::usage = 
	"GetSingleRegressionResult[namedProperty, dataSet, pureFunction]"

PlotFunction2D::usage = 
	"PlotFunction2D[pureFunction, argumentRange, functionValueRange, labels, options]"

PlotFunctions2D::usage = 
	"PlotFunctions2D[pureFunctions, argumentRange, functionValueRange, plotStyle, labels, options]"

PlotLine2DWithOptionalPoints::usage = 
	"PlotLine2DWithOptionalPoints[points2D, optionalPoints2D, labels, options]"

PlotLine2DWithOptionalPointsAndMaximumXValue::usage = 
	"PlotLine2DWithOptionalPointsAndMaximumXValue[points2D, optionalPoints2D, maximumXValue, labels, options]"
	
PlotPoints2D::usage = 
	"PlotPoints2D[points2D, labels, options]"

PlotPoints2DAboveDiagonal::usage = 
	"PlotPoints2DAboveDiagonal[points2D, labels, options]"
	
PlotPoints2DAboveFunction::usage = 
	"PlotPoints2DAboveFunction[points2D, pureFunction, labels, options]"

PlotPoints2DAboveFunctionWithMaximumYValue::usage = 
	"PlotPoints2DAboveFunctionWithMaximumYValue[points2D, pureFunction, maximumYValue, labels, options]"

PlotPoints2DAboveMultipleFunctions::usage = 
	"PlotPoints2DAboveMultipleFunctions[points2D, pureFunctions, argumentRange, functionValueRange, plotStyle, labels, options]"

PlotPoints2DWithOptionalPoints::usage = 
	"PlotPoints2DWithOptionalPoints[points2D, optionalPoints2D, labels, options]"

PlotFunction3D::usage = 
	"PlotFunction3D[pureFunction, xRange, yRange, labels, options]"

PlotPoints3D::usage = 
	"PlotPoints3D[points3D, labels, options]"

Plot3dDataSet::usage = 
	"Plot3dDataSet[dataSet3D, labels, options]"

PlotPoints3DWithFunction::usage = 
	"PlotPoints3DWithFunction[points3D, pureFunction, labels, options]"

Plot3dDataSetWithFunction::usage = 
	"Plot3dDataSetWithFunction[dataSet3D, pureFunction, labels, options]"

PlotBarChart::usage = 
	"PlotBarChart[barValues, labels, options]"

PlotGroupedBarChart::usage = 
	"PlotGroupedBarChart[barValues, labels, options]"
	
PlotIndexedPoints2D::usage = 
	"PlotIndexedPoints2D[yList, x, labels, options]"
	
PlotIndexedLine2D::usage = 
	"PlotIndexedLine2D[yList, x, labels, options]"

PlotMultipleLines2D::usage = 
	"PlotMultipleLines2D[points2DWithPlotStyleList, labels, options]"

PlotMultiplePoints2D::usage = 
	"PlotMultiplePoints2D[points2DWithPlotStyleList, labels, options]"

PlotResiduals::usage = 
	"PlotResiduals[residuals, labels, options]"

PlotSilhouetteWidths::usage = 
	"PlotSilhouetteWidths[silhouetteWidths, meanSilhouetteWidth, labels, options]"
	
PlotTwoIndexedLines2D::usage = 
	"PlotTwoIndexedLines2D[yList1, yList2, x, labels, options]"
	
PlotUpToFourLines2D::usage = 
	"PlotUpToFourLines2D[points2DRed, points2DGreen, points2DBlue, points2DBlack, labels, options]"
	
PlotUpToFourPoint2DSets::usage = 
	"PlotUpToFourPoint2DSets[points2DRed, points2DGreen, points2DBlue, points2DBlack, labels, options]"

PlotXyErrorData::usage = 
	"PlotXyErrorData[xyErrorData, labels, options]"

PlotXyErrorDataAboveFunction::usage = 
	"PlotXyErrorDataAboveFunction[xyErrorData, pureFunction, labels, options]"

PlotXyErrorDataAboveFunctions::usage = 
	"PlotXyErrorDataAboveFunctions[xyErrorData, pureFunctions, argumentRange, functionValueRange, plotStyle, labels, options]"

ShowClassificationResult::usage = 
	"ShowClassificationResult[namedPropertyList, dataSet, pureFunction]"

ShowClassificationScan::usage = 
	"ShowClassificationScan[classificationTrainingScanResult, options]"

ShowDataSetInfo::usage = 
	"ShowDataSetInfo[namedPropertyList, dataSet]"

ShowInputRelevanceClass::usage = 
	"ShowInputRelevanceClass[inputComponentRelevanceListForClassification, options]"

ShowInputRelevanceRegress::usage = 
	"ShowInputRelevanceRegress[inputComponentRelevanceListForRegression, options]"

ShowInputsInfo::usage = 
	"ShowInputsInfo[namedPropertyList, inputs]"

ShowRegressionResult::usage = 
	"ShowRegressionResult[namedPropertyList, dataSet, pureFunction, options]"

ShowRegressionScan::usage = 
	"ShowRegressionScan[regressionTrainingScanResult, options]"

ShowSingleRegressionResult::usage = 
	"ShowSingleRegressionResult[namedProperty, dataSet, pureFunction, options]"

ShowTrainOptimization::usage = 
	"ShowTrainOptimization[trainingSetOptimizationResult, options]"

(* ::Section:: *)
(* Functions *)
	
Begin["`Private`"]

GetClassRelevantComponents[

	(* Returns most-to-least-relevance sorted components from inputComponentRelevanceListForClassification.

	   Returns: Most-to-least-relevance sorted components *)


	(* inputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, info data structure}
	   trainingSetResult: {numberOfRemovedInputs, (best) correct classification in percent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best correct classification in percent of test set} *)
	inputComponentRelevanceListForClassification_,
	
	(* Number of components to return *)
	numberOfComponents_?IntegerQ
    
	] :=
  
	Module[
    
		{
			longestInputComponentList 
		},

		longestInputComponentList = inputComponentRelevanceListForClassification[[Length[inputComponentRelevanceListForClassification], 3]];
		If[Length[longestInputComponentList] >= numberOfComponents,
			
			Return[Take[longestInputComponentList, numberOfComponents]],
			
			Return[longestInputComponentList]
		]
	];

GetRegressRelevantComponents[

	(* Returns most-to-least-relevance sorted components from inputComponentRelevanceListForRegression.

	   Returns: Most-to-least-relevance sorted components *)


	(* inputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, info data structure}
	   trainingSetResult: {numberOfRemovedInputs, (best) RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)
	inputComponentRelevanceListForRegression_,
	
	(* Number of components to return *)
	numberOfComponents_?IntegerQ
    
	] :=
  
	Module[
    
		{
			longestInputComponentList
		},

		longestInputComponentList = inputComponentRelevanceListForRegression[[Length[inputComponentRelevanceListForRegression], 3]];
		If[Length[longestInputComponentList] >= numberOfComponents,
			
			Return[Take[longestInputComponentList, numberOfComponents]],
			
			Return[longestInputComponentList]
		]
	];

GetResidualsDistribution[

	(* Returns distribution for residuals.
	
	   Returns:
	   intervalPoints: {interval1, interval2, ...}
	   interval: {middle position, frequency in percent of interval} *)


    (* List with residuals *)
    residuals_/;VectorQ[residuals, NumberQ],
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
	    {
	    	frequencyList,
	    	index,
	    	intervalLength,
	    	intervalMiddlePositionList,
	    	intervalPoints,
	    	i,
	    	numberOfResiduals,
	    	numberOfIntervals,
	    	upperBound,
	    	lowerBound
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    numberOfIntervals = GraphicsOptionNumberOfIntervals/.{opts}/.Options[GraphicsOptionsResidualsDistribution];

		numberOfResiduals = Length[residuals];
		
		upperBound = Max[Abs[Min[residuals]],Abs[Max[residuals]]];
		lowerBound = -upperBound;

		intervalLength = (upperBound - lowerBound)/numberOfIntervals;
		frequencyList = Table[0.0, {numberOfIntervals}];
		Do[
			index = Floor[(residuals[[i]] - lowerBound)/intervalLength] + 1;
			If[index > numberOfIntervals, index = numberOfIntervals];
			frequencyList[[index]] = frequencyList[[index]] + 1.0,
			
			{i, numberOfResiduals}
		];		
		intervalMiddlePositionList = Table[lowerBound + intervalLength*(i - 0.5), {i, numberOfIntervals}];
		intervalPoints = 
			Table[
				{intervalMiddlePositionList[[i]], frequencyList[[i]]/numberOfResiduals*100.0},
				
				{i, numberOfIntervals}
			];
					
		Return[intervalPoints]
	];
	
GetSingleRegressionResult[
	
	(* Returns single regression result according to named property list.

	   Returns :
	   Single regression result according to named property *)

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

	(* Pure function of inputs
	   inputs = {input1, input2, ...} 
	   input: {inputComponent1, inputComponent2, ...} *)
	pureFunction_,
	
	(* Options *)
	opts___

    
	] :=
  
	Module[
    
    	{
    		absoluteResiduals,
    		absoluteDeviations,
    		componentOutput,
    		deviationsInPercent,
    		deviationsOfComponent,
    		displayDataList,
    		displayDataMatrix,
    		i,
    		inputs,
    		k,
    		machineComponentOutput,
    		machineDisplayDataList,
    		machineOutputs,
    		mse,
    		numberOfOutputComponents,
    		outputs,
    		relativeResiduals,
    		singleRegressionResult,
    		sortedDisplayDataMatrix,
    		numberOfIntervals,
    		mseList
    	},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    numberOfIntervals = GraphicsOptionNumberOfIntervals/.{opts}/.Options[GraphicsOptionsResidualsDistribution];

		inputs = CIP`Utility`GetInputsOfDataSet[dataSet];
		outputs = CIP`Utility`GetOutputsOfDataSet[dataSet];
    	numberOfOutputComponents = Length[outputs[[1]]];
    	machineOutputs = pureFunction[inputs];
		
		absoluteDeviations = outputs - machineOutputs;
		
		Switch[namedProperty,
			
			(* -------------------------------------------------------------------------------- *)
			"RMSE",
			mse = CIP`Utility`GetMeanSquaredError[dataSet, pureFunction];
			Return[Sqrt[mse]],

			(* -------------------------------------------------------------------------------- *)
			"SingleOutputRMSE",
			mseList = CIP`Utility`GetMeanSquaredErrorList[dataSet, pureFunction];
			Return[Sqrt[mseList]],

			(* -------------------------------------------------------------------------------- *)
			"AbsoluteResidualsStatistics",
			singleRegressionResult = {};
		    Do[
				deviationsOfComponent = Flatten[absoluteDeviations[[All, k]]];
				AppendTo[
					singleRegressionResult, 
					{
						Mean[Abs[deviationsOfComponent]], 
						Median[Abs[deviationsOfComponent]], 
						Max[Abs[deviationsOfComponent]]
					}
				],
		    	
				{k, numberOfOutputComponents}
			];
			Return[singleRegressionResult],
			
			(* -------------------------------------------------------------------------------- *)
			"RelativeResidualsStatistics",
			deviationsInPercent = 100.0*(outputs - machineOutputs)/outputs;
			singleRegressionResult = {};
		    Do[
				deviationsOfComponent = Flatten[deviationsInPercent[[All, k]]];
				AppendTo[
					singleRegressionResult, 
					{
						Mean[Abs[deviationsOfComponent]], 
						Median[Abs[deviationsOfComponent]], 
						Max[Abs[deviationsOfComponent]]
					}
				],
		    	
				{k, numberOfOutputComponents}
		    ];
			Return[singleRegressionResult],
			
			(* -------------------------------------------------------------------------------- *)
			"ModelVsData",
			singleRegressionResult = {};
		    Do[
				machineComponentOutput = Flatten[machineOutputs[[All, k]]];
				componentOutput = Flatten[outputs[[All, k]]];
				displayDataMatrix =
					Table[
						{machineComponentOutput[[i]], componentOutput[[i]]},
						
						{i, Length[machineComponentOutput]}
					];	
				AppendTo[singleRegressionResult, displayDataMatrix],
		    	
				{k, numberOfOutputComponents}
		    ];
			Return[singleRegressionResult],
			
			(* -------------------------------------------------------------------------------- *)
			"CorrelationCoefficient",
			singleRegressionResult = {};
		    Do[
				machineComponentOutput = Flatten[machineOutputs[[All, k]]];
				componentOutput = Flatten[outputs[[All, k]]];	
				AppendTo[singleRegressionResult, Correlation[machineComponentOutput, componentOutput]],
		    	
				{k, numberOfOutputComponents}
		    ];
			Return[singleRegressionResult],
			
			(* -------------------------------------------------------------------------------- *)
			"SortedModelVsData",
			singleRegressionResult = {};
		    Do[
				machineComponentOutput = Flatten[machineOutputs[[All, k]]];
				componentOutput = Flatten[outputs[[All, k]]];
				displayDataMatrix =
					Table[
						{machineComponentOutput[[i]], componentOutput[[i]]},
						
						{i, Length[machineComponentOutput]}
					];	
				sortedDisplayDataMatrix = Sort[displayDataMatrix];
				machineDisplayDataList = sortedDisplayDataMatrix[[All, 1]];
				displayDataList = sortedDisplayDataMatrix[[All, 2]];
				AppendTo[singleRegressionResult, {machineDisplayDataList, displayDataList}],
		    	
				{k, numberOfOutputComponents}
		    ];
			Return[singleRegressionResult],
			
			(* -------------------------------------------------------------------------------- *)
			"AbsoluteSortedResiduals",
			singleRegressionResult = {};
		    Do[
				machineComponentOutput = Flatten[machineOutputs[[All, k]]];
				componentOutput = Flatten[outputs[[All, k]]];
				displayDataMatrix =
					Table[
						{machineComponentOutput[[i]], componentOutput[[i]]},
						
						{i, Length[machineComponentOutput]}
					];	
				sortedDisplayDataMatrix = Sort[displayDataMatrix];
				machineDisplayDataList = sortedDisplayDataMatrix[[All, 1]];
				displayDataList = sortedDisplayDataMatrix[[All, 2]];
				absoluteResiduals = displayDataList - machineDisplayDataList;
				AppendTo[singleRegressionResult, absoluteResiduals],
		    	
				{k, numberOfOutputComponents}
		    ];
			Return[singleRegressionResult],
			
			(* -------------------------------------------------------------------------------- *)
			"RelativeSortedResiduals",
			singleRegressionResult = {};
		    Do[
				machineComponentOutput = Flatten[machineOutputs[[All, k]]];
				componentOutput = Flatten[outputs[[All, k]]];
				displayDataMatrix =
					Table[
						{machineComponentOutput[[i]], componentOutput[[i]]},
						
						{i, Length[machineComponentOutput]}
					];	
				sortedDisplayDataMatrix = Sort[displayDataMatrix];
				machineDisplayDataList = sortedDisplayDataMatrix[[All, 1]];
				displayDataList = sortedDisplayDataMatrix[[All, 2]];
				relativeResiduals = 100.0*(displayDataList - machineDisplayDataList)/displayDataList;
				AppendTo[singleRegressionResult, relativeResiduals],
		    	
				{k, numberOfOutputComponents}
		    ];
			Return[singleRegressionResult],
			
			(* -------------------------------------------------------------------------------- *)
			"AbsoluteResidualsDistribution",
			singleRegressionResult = {};
		    Do[
				deviationsOfComponent = Flatten[absoluteDeviations[[All, k]]];
				AppendTo[singleRegressionResult, GetResidualsDistribution[deviationsOfComponent, GraphicsOptionNumberOfIntervals -> numberOfIntervals]],
		    	
				{k, numberOfOutputComponents}
		    ];
			Return[singleRegressionResult],
			
			(* -------------------------------------------------------------------------------- *)
			"RelativeResidualsDistribution",
			deviationsInPercent = 100.0*(outputs - machineOutputs)/outputs;
			singleRegressionResult = {};
		    Do[
				deviationsOfComponent = Flatten[deviationsInPercent[[All, k]]];
				AppendTo[singleRegressionResult, GetResidualsDistribution[deviationsOfComponent, GraphicsOptionNumberOfIntervals -> numberOfIntervals]],
		    	
				{k, numberOfOutputComponents}
		    ];
			Return[singleRegressionResult]
		]
	];

PlotFunction2D[

	(* Displays function y=f(x).

	   Returns : Graphics *)


	(* Pure function *)      
	pureFunction_,

	(* {xMin, xMax} *)
	argumentRange_/;VectorQ[argumentRange, NumberQ],

	(* {yMin, yMax} *)
	functionValueRange_/;VectorQ[functionValueRange, NumberQ],
      
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
      
	] :=
    
	Module[
      
		{
			displayFunction,
			imageSize,
			xOffset, 
			xMin, 
			xMax,
			x,
			yOffset, 
			yMin, 
			yMax,
			plotStyle
		},
      
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    plotStyle = GraphicsOptionLinePlotStyle/.{opts}/.Options[GraphicsOptionsLinePlotStyle];
	
		xMin = argumentRange[[1]];
		xMax = argumentRange[[2]];
		xOffset = (xMax - xMin)/20.0;
		xMin = xMin - xOffset;
		xMax = xMax + xOffset;

		yMin = functionValueRange[[1]];
		yMax = functionValueRange[[2]];
		yOffset = (yMax - yMin)/20.0;
		yMin = yMin - yOffset;
		yMax = yMax + yOffset;
      
      	Return[
			Plot[
				Evaluate[pureFunction[x]], 
				{x, xMin, xMax}, 
				PlotRange -> {yMin, yMax},
				Axes -> False, 
				Frame -> True, 
				FrameTicks -> Automatic, 
				FrameLabel -> {labels[[1]], labels[[2]]}, 
				PlotLabel -> labels[[3]], 
				PlotStyle -> plotStyle, 
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
      	]
	];

PlotFunctions2D[

	(* Displays multiple functions y=f(x).

	   Returns : Graphics *)


	(* Pure functions: {PureFunction1, PureFunction2, ...} *)      
	pureFunctions_,

	(* {xMin, xMax} *)
	argumentRange_/;VectorQ[argumentRange, NumberQ],

	(* {yMin, yMax} *)
	functionValueRange_/;VectorQ[functionValueRange, NumberQ],
      
    (* Plot style *)
	plotStyle_,
      
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
      
	] :=
    
	Module[
      
		{
			displayFunction,
			imageSize,
			i,
			xOffset, 
			xMin, 
			xMax,
			x,
			yOffset, 
			yMin, 
			yMax
		},
      
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	
		xMin = argumentRange[[1]];
		xMax = argumentRange[[2]];
		xOffset = (xMax - xMin)/20.0;
		xMin = xMin - xOffset;
		xMax = xMax + xOffset;

		yMin = functionValueRange[[1]];
		yMax = functionValueRange[[2]];
		yOffset = (yMax - yMin)/20.0;
		yMin = yMin - yOffset;
		yMax = yMax + yOffset;
      
      	Return[
			Plot[
				Evaluate[Table[pureFunctions[[i]][x], {i, Length[pureFunctions]}]], 
				{x, xMin, xMax}, 
				PlotRange -> {yMin, yMax},
				Axes -> False, 
				Frame -> True, 
				FrameTicks -> Automatic, 
				FrameLabel -> {labels[[1]], labels[[2]]}, 
				PlotLabel -> labels[[3]], 
				PlotStyle -> plotStyle, 
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
      	]
	];

PlotLine2DWithOptionalPoints[

	(* Displays joined points2D (in black) and optional points2D (in blue)

	   Returns : Graphics *)

    
    (* points2D: {point1, point2, ...}
       point: {x, y} *)
    points2D_/;MatrixQ[points2D, NumberQ],
    
    (* If not {} then coordinate of optionalPoints2D that will be displayed in blue
	   points2D : {point1, point2, ...}
	   point : {x, y} *)
    optionalPoints2D_,
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
    	{
    		argumentRange,
    		displayFunction,
    		imageSize,
    		xOffset, 
    		yOffset, 
    		xMin, 
    		xMax, 
    		yMin, 
    		yMax, 
    		epilog,
			functionValueRange,
    		pointSize,
    		pointColor,
    		plotStyle
    	},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	    argumentRange = GraphicsOptionArgumentRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];
	    functionValueRange = GraphicsOptionFunctionValueRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];
	    plotStyle = GraphicsOptionLinePlotStyle/.{opts}/.Options[GraphicsOptionsLinePlotStyle];
	
		If[Length[argumentRange] == 0,

			xMin = Min[points2D[[All, 1]]];
		    xMax = Max[points2D[[All, 1]]],
		    
			xMin = argumentRange[[1]];
		    xMax = argumentRange[[2]]
		];
	    xOffset = (xMax - xMin)/20.0;
	    xMin = xMin - xOffset;
	    xMax = xMax + xOffset;

		If[Length[functionValueRange] == 0,

		    yMin = Min[points2D[[All, 2]]];
		    yMax = Max[points2D[[All, 2]]],
		    
		    yMin = functionValueRange[[1]];
		    yMax = functionValueRange[[2]]
		];
	    yOffset = (yMax - yMin)/20.0;
	    yMin = yMin - yOffset;
	    yMax = yMax + yOffset;
    
	    If[Length[optionalPoints2D] == 0,
	    	
			(* No optionalPoints2D defined *)
			epilog = {},
			
			(* Display optionalPoints2D *)
			epilog = {PointSize[pointSize], pointColor, Point[optionalPoints2D]}
		];
    
    	Return[
			ListLinePlot[
				points2D, 
				Axes -> False, 
				Frame -> True, 
				FrameTicks -> Automatic, 
				FrameLabel -> {labels[[1]], labels[[2]]}, 
				PlotLabel -> labels[[3]], 
				PlotStyle -> plotStyle, 
				PlotRange -> {{xMin, xMax}, {yMin, yMax}}, 
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				Epilog -> epilog, 
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
    	]
	];

PlotLine2DWithOptionalPointsAndMaximumXValue[

	(* Displays joined points2D (in black) and optional points2D (in blue). The maximum value of the x-axis is defined.

	   Returns : Graphics *)

    
    (* points2D: {point1, point2, ...}
       point: {x, y} *)
    points2D_/;MatrixQ[points2D, NumberQ],
    
    (* If not {} then coordinate of optionalPoints2D that will be displayed in blue
	   points2D : {point1, point2, ...}
	   point : {x, y} *)
    optionalPoints2D_,

    (* Maximum x-value *)
	maximumXValue_,
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
    	{
    		displayFunction,
    		imageSize,
    		xOffset, 
    		yOffset, 
    		xMin, 
    		xMax, 
    		yMin, 
    		yMax, 
    		epilog,
    		pointSize,
    		pointColor,
    		plotStyle
    	},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	    plotStyle = GraphicsOptionLinePlotStyle/.{opts}/.Options[GraphicsOptionsLinePlotStyle];
	
		xMin = Min[points2D[[All, 1]]];
	    xMax = maximumXValue;
	    xOffset = (xMax - xMin)/20.0;
	    xMin = xMin - xOffset;
	    xMax = xMax + xOffset;

	    yMin = Min[points2D[[All, 2]]];
	    yMax = Max[points2D[[All, 2]]];
	    yOffset = (yMax - yMin)/20.0;
	    yMin = yMin - yOffset;
	    yMax = yMax + yOffset;
    
	    If[Length[optionalPoints2D] == 0,
	    	
			(* No optionalPoints2D defined *)
			epilog = {},
			
			(* Display optionalPoints2D *)
			epilog = {PointSize[pointSize], pointColor, Point[optionalPoints2D]}
		];
    
    	Return[
			ListLinePlot[
				points2D, 
				Axes -> False, 
				Frame -> True, 
				FrameTicks -> Automatic, 
				FrameLabel -> {labels[[1]], labels[[2]]}, 
				PlotLabel -> labels[[3]], 
				PlotStyle -> plotStyle, 
				PlotRange -> {{xMin, xMax}, {yMin, yMax}}, 
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				Epilog -> epilog, 
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
    	]
	];

PlotPoints2D[

	(* Displays points2D (in blue)

	   Returns : Graphics *)


    (* points2D: {point1, point2, ...}
       point: {x, y} *)
    points2D_/;MatrixQ[points2D, NumberQ],
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
    ] :=
  
	Module[
    
    	{
    		argumentRange,
			displayFunction,
			imageSize,
			functionValueRange,
    		xOffset, 
    		yOffset, 
    		xMin, 
    		xMax, 
    		yMin, 
    		yMax,
    		pointSize,
    		pointColor
    	},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    argumentRange = GraphicsOptionArgumentRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];
	    functionValueRange = GraphicsOptionFunctionValueRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	
		If[Length[argumentRange] == 0,

			xMin = Min[points2D[[All, 1]]];
		    xMax = Max[points2D[[All, 1]]],
		    
			xMin = argumentRange[[1]];
		    xMax = argumentRange[[2]]
		];
	    xOffset = (xMax - xMin)/20.0;
	    xMin = xMin - xOffset;
	    xMax = xMax + xOffset;

		If[Length[functionValueRange] == 0,

		    yMin = Min[points2D[[All, 2]]];
		    yMax = Max[points2D[[All, 2]]],
		    
		    yMin = functionValueRange[[1]];
		    yMax = functionValueRange[[2]]
		];
	    yOffset = (yMax - yMin)/20.0;
	    yMin = yMin - yOffset;
	    yMax = yMax + yOffset;
    
    	Return[
			ListPlot[
				points2D, 
				Axes -> False, 
				Frame -> True, 
				FrameTicks -> Automatic, 
				FrameLabel -> {labels[[1]], labels[[2]]}, 
				PlotLabel -> labels[[3]], 
				PlotRange -> {{xMin, xMax}, {yMin, yMax}}, 
				PlotStyle -> {PointSize[pointSize], pointColor}, 
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
    	]
	];

PlotPoints2DAboveDiagonal[

	(* Displays points2D (in blue) above diagonal

	   Returns : Graphics *)


    (* points2D: {point1, point2, ...}
       point: {x, y} *)
    points2D_/;MatrixQ[points2D, NumberQ],
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
    ] :=
  
	Module[
    
    	{
			displayFunction,
			imageSize,
    		max,
    		min, 
    		offset,
    		pointSize,
    		pointColor
    	},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	
	    min = Min[Min[points2D[[All, 1]]],Min[points2D[[All, 2]]]];
	    max = Max[Max[points2D[[All, 1]]],Max[points2D[[All, 2]]]];
	    offset = (max - min)/20.0;
	    min -= offset;
	    max += offset;
	    
		Return[
			ListPlot[
				points2D, 
				Axes -> False, 
				Frame -> True, 
				FrameTicks -> Automatic, 
				FrameLabel -> {labels[[1]], labels[[2]]}, 
				PlotLabel -> labels[[3]], 
				PlotRange -> {{min, max}, {min, max}}, 
				PlotStyle -> {PointSize[pointSize], pointColor}, 
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				Prolog -> {Thickness[0.005], Black, Line[{{min, min}, {max, max}}]}, 
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
		]
	];

PlotPoints2DAboveFunction[

	(* Displays points2D (in blue) above function (in black)

	   Returns : Graphics *)

    
    (* points2D: {point1, point2, ...}
       point: {x, y} *)
    points2D_/;MatrixQ[points2D, NumberQ],
    
    (* Pure function*)
    pureFunction_,
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
    ] :=
  
	Module[
    
	    {
	   		argumentRange,
			functionValueRange,
	    	argumentOfFunction,
			displayFunction,
			imageSize,
	    	xOffset, 
	    	yOffset, 
	    	xMin, 
	    	xMax, 
	    	yMin, 
	    	yMax,
    		pointSize,
    		pointColor,
    		plotStyle
	    },
	    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	    argumentRange = GraphicsOptionArgumentRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];
	    functionValueRange = GraphicsOptionFunctionValueRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];
	    plotStyle = GraphicsOptionLinePlotStyle/.{opts}/.Options[GraphicsOptionsLinePlotStyle];
	
		If[Length[argumentRange] == 0,

			xMin = Min[points2D[[All, 1]]];
		    xMax = Max[points2D[[All, 1]]],
		    
			xMin = argumentRange[[1]];
		    xMax = argumentRange[[2]]
		];
	    xOffset = (xMax - xMin)/20.0;
	    xMin = xMin - xOffset;
	    xMax = xMax + xOffset;

		If[Length[functionValueRange] == 0,

		    yMin = Min[points2D[[All, 2]]];
		    yMax = Max[points2D[[All, 2]]],
		    
		    yMin = functionValueRange[[1]];
		    yMax = functionValueRange[[2]]
		];
	    yOffset = (yMax - yMin)/20.0;
	    yMin = yMin - yOffset;
	    yMax = yMax + yOffset;
			    
	    Return[
		    Plot[
		    	pureFunction[argumentOfFunction], 
		    	{argumentOfFunction, xMin, xMax}, 
		    	Axes -> False, 
		    	Frame -> True, 
		    	FrameTicks -> Automatic, 
		    	FrameLabel -> {labels[[1]], labels[[2]]}, 
		    	PlotLabel -> labels[[3]], 
		    	PlotRange -> {yMin, yMax},
				PlotStyle -> plotStyle, 
		    	BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
		    	Epilog -> {PointSize[pointSize], pointColor, Point[points2D]}, 
		    	DisplayFunction -> displayFunction,
				ImageSize -> imageSize
		    ]
	    ]
	];

PlotPoints2DAboveFunctionWithMaximumYValue[

	(* Displays points2D (in blue) above function (in black) with maximum y-value

	   Returns : Graphics *)

    
    (* points2D: {point1, point2, ...}
       point: {x, y} *)
    points2D_/;MatrixQ[points2D, NumberQ],
    
    (* Pure function*)
    pureFunction_,
	
    (* maximum y-value *)
	maximumYValue_,
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
    ] :=
  
	Module[
    
	    {
	    	argumentOfFunction,
			displayFunction,
			imageSize,
	    	xOffset, 
	    	yOffset, 
	    	xMin, 
	    	xMax, 
	    	yMin, 
	    	yMax, 
    		pointSize,
    		pointColor,
	    	plotStyle
	    },
	    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	    plotStyle = GraphicsOptionLinePlotStyle/.{opts}/.Options[GraphicsOptionsLinePlotStyle];
	
	    xMin = Min[points2D[[All, 1]]];
	    xMax = Max[points2D[[All, 1]]];
	    xOffset = (xMax - xMin)/20.0;
	    xMin = xMin - xOffset;
	    xMax = xMax + xOffset;
	    
	    yMin = Min[points2D[[All, 2]]];
	    yMax = maximumYValue;
	    yOffset = (yMax - yMin)/20.0;
	    yMin = yMin - yOffset;
	    yMax = yMax + yOffset;
	    
	    Return[
		    Plot[
		    	pureFunction[argumentOfFunction], 
		    	{argumentOfFunction, xMin, xMax}, 
		    	Axes -> False, 
		    	Frame -> True, 
		    	FrameTicks -> Automatic, 
		    	FrameLabel -> {labels[[1]], labels[[2]]}, 
		    	PlotLabel -> labels[[3]], 
		    	PlotRange -> {yMin, yMax}, 
				PlotStyle -> plotStyle, 
		    	BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
		    	Epilog -> {PointSize[pointSize], pointColor, Point[points2D]}, 
		    	DisplayFunction -> displayFunction,
				ImageSize -> imageSize
		    ]
	    ]
	];

PlotPoints2DAboveMultipleFunctions[

	(* Displays points2D (in blue) above multiple functions.

	   Returns : Graphics *)

    
    (* points2D: {point1, point2, ...}
       point: {x, y} *)
    points2D_/;MatrixQ[points2D, NumberQ],
    
	(* Pure functions: {PureFunction1, PureFunction2, ...}*)      
	pureFunctions_,

	(* {xMin, xMax} *)
	argumentRange_/;VectorQ[argumentRange, NumberQ],

	(* {yMin, yMax} *)
	functionValueRange_/;VectorQ[functionValueRange, NumberQ],
      
    (* Plot style *)
	plotStyle_,
      
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
    ] :=
  
	Module[
    
		{
			displayFunction,
			imageSize,
			i,
			xOffset, 
			xMin, 
			xMax,
			x,
			yOffset, 
			yMin, 
			yMax,
    		pointSize,
    		pointColor
		},
      
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	
		xMin = argumentRange[[1]];
		xMax = argumentRange[[2]];
		xOffset = (xMax - xMin)/20.0;
		xMin = xMin - xOffset;
		xMax = xMax + xOffset;

		yMin = functionValueRange[[1]];
		yMax = functionValueRange[[2]];
		yOffset = (yMax - yMin)/20.0;
		yMin = yMin - yOffset;
		yMax = yMax + yOffset;
      
      	Return[
			Plot[
				Evaluate[Table[pureFunctions[[i]][x], {i, Length[pureFunctions]}]], 
				{x, xMin, xMax}, 
				PlotRange -> {yMin, yMax},
				Axes -> False, 
				Frame -> True, 
				FrameTicks -> Automatic, 
				FrameLabel -> {labels[[1]], labels[[2]]}, 
				PlotLabel -> labels[[3]], 
				PlotStyle -> plotStyle, 
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"},
		    	Epilog -> {PointSize[pointSize], pointColor, Point[points2D]}, 
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
      	]
	];

PlotPoints2DWithOptionalPoints[

	(* Displays points2D (in black) and optionally points2D (in blue)

	   Returns : Graphics *)


    (* points2D: {point1, point2, ...}
       point: {x, y} *)
    points2D_/;MatrixQ[points2D, NumberQ],
    
    (* If not {} then coordinate of optionalPoints2D that will be displayed in blue
	   points2D : {point1, point2, ...}
	   point : {x, y} *)
    optionalPoints2D_,
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
    ] :=
  
	Module[
    
    	{
			displayFunction,
			imageSize,
    		xOffset, 
    		yOffset, 
    		xMin, 
    		xMax, 
    		yMin, 
    		yMax, 
    		epilog,
    		pointSize,
    		pointColor    		
    	},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	
	    xMin = Min[points2D[[All, 1]]];
	    xMax = Max[points2D[[All, 1]]];
	    xOffset = (xMax - xMin)/20.0;
	    xMin = xMin - xOffset;
	    xMax = xMax + xOffset;
	    
	    yMin = Min[points2D[[All, 2]]];
	    yMax = Max[points2D[[All, 2]]];
	    yOffset = (yMax - yMin)/20.0;
	    yMin = yMin - yOffset;
	    yMax = yMax + yOffset;
    
		If[Length[optionalPoints2D] == 0,
			
			(* No optionalPoints2D defined *)
			epilog = {},
			
			(* Display optionalPoints2D *)
			epilog = {PointSize[pointSize], pointColor, Point[optionalPoints2D]}
		];
    
		Return[
			ListPlot[
				points2D, 
				Axes -> False, 
				Frame -> True, 
				FrameTicks -> Automatic, 
				FrameLabel -> {labels[[1]], labels[[2]]}, 
				PlotLabel -> labels[[3]], 
				PlotRange -> {{xMin, xMax}, {yMin, yMax}}, 
				PlotStyle -> {RGBColor[0, 0, 0, 0.8], PointSize[pointSize]}, 
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				Epilog -> epilog, 
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
		]
	];

PlotFunction3D[

	(* Plots 3D function y=f(x, y).
	   
	   Returns : Graphics3D *)
	
	
    (* Pure function f(x, y) *)
    pureFunction_,

    (* {xMin, xMax} *)
	xRange_/;VectorQ[xRange, NumberQ],
	
    (* {yMin, yMax} *)
	yRange_/;VectorQ[yRange, NumberQ],
	
    (* {xLabel, yLabel, zLabel} *)
	labels_,

	(* Options *)
	opts___

	] :=
  
	Module[
    
		{
			regionFunction,
			plotStyle,
			colorFunction,
			displayFunction,
			imageSize,
			mesh,
			meshStyle,
			numberOfPlotPoints,
			viewPoint,
			x,
			y
		},
	
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    viewPoint = GraphicsOptionViewPoint3D/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    numberOfPlotPoints = GraphicsOptionNumberOfPlotPoints/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    colorFunction = GraphicsOptionColorFunction/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    mesh = GraphicsOptionIsMesh/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    meshStyle = GraphicsOptionMeshStyle/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    plotStyle = GraphicsOptionPlotStyle3D/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    regionFunction = GraphicsOptionRegionFunction/.{opts}/.Options[GraphicsOptionsGraphics3D];

		Return[
			Plot3D[
				pureFunction[x, y], 
				{x, xRange[[1]], xRange[[2]]}, 
				{y, yRange[[1]], yRange[[2]]}, 
				AspectRatio -> 1, 
				AxesLabel -> {labels[[1]], labels[[2]], labels[[3]]}, 
				BoxRatios -> {1, 1, 1}, 
				ColorFunction -> colorFunction, 
				Mesh -> mesh, 
				MeshStyle -> meshStyle, 
				PlotPoints -> numberOfPlotPoints, 
				PlotRange -> {Full, Full, Full}, 
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				Ticks -> Automatic, 
				ViewPoint -> viewPoint, 
				PlotStyle -> plotStyle,
				RegionFunction -> regionFunction,
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
		]
	];

PlotPoints3D[

	(* Plots specified points3D in 3D box.
	   
	   Returns : Graphics3D *)
	
	
    (* points3D: {point1, point2, ...}
	   point: {x, y, z} *)
	points3D_/;MatrixQ[points3D, NumberQ], 
	
    (* {xLabel, yLabel, zLabel} *)
	labels_,

	(* Options *)
	opts___

	] :=
  
	Module[
    
		{
			argument1Range,
			argument2Range,
			displayFunction,
			imageSize,
			viewPoint,
    		pointSize,
    		pointColor
		},
	
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    viewPoint = GraphicsOptionViewPoint3D/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	    argument1Range = GraphicsOptionArgument1Range3D/.{opts}/.Options[GraphicsOptionsPlotRange3D];
	    argument2Range = GraphicsOptionArgument2Range3D/.{opts}/.Options[GraphicsOptionsPlotRange3D];

		If[Length[argument1Range] == 0,
			argument1Range = Full
		];

		If[Length[argument2Range] == 0,
			argument2Range = Full
		];

		Return[
			ListPointPlot3D[
				points3D, 
				AxesLabel -> {labels[[1]], labels[[2]], labels[[3]]}, 
				AspectRatio -> 1, 
				BoxRatios -> {1, 1, 1}, 
				PlotRange -> {argument1Range, argument2Range, Full}, 
				PlotStyle -> {{PointSize[pointSize], pointColor}}, 
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				Ticks -> Automatic, 
				ViewPoint -> viewPoint, 
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
		]
	];

Plot3dDataSet[

	(* Plots 3D data set in 3D box.
	   
	   Returns : Graphics3D *)
	
	
	(* 3D dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2}
	   output: {outputValue} *)
	dataSet3D_,
	
    (* {xLabel, yLabel, zLabel} *)
	labels_,

	(* Options *)
	opts___

	] :=
  
	Module[
    
		{
			argument1Range,
			argument2Range,
			displayFunction,
			imageSize,
			viewPoint,
    		pointSize,
    		pointColor,
			points3D,
			ioPair,
			i
		},
	
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    viewPoint = GraphicsOptionViewPoint3D/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	    argument1Range = GraphicsOptionArgument1Range3D/.{opts}/.Options[GraphicsOptionsPlotRange3D];
	    argument2Range = GraphicsOptionArgument2Range3D/.{opts}/.Options[GraphicsOptionsPlotRange3D];

		points3D = 
			Table[
				ioPair = dataSet3D[[i]];
				{ioPair[[1, 1]], ioPair[[1, 2]], ioPair[[2, 1]]},
				
				{i, Length[dataSet3D]}
			];
	
		Return[
			PlotPoints3D[
				points3D,
				labels,
	    		GraphicsOptionDisplayFunction -> displayFunction,
	    		GraphicsOptionViewPoint3D -> viewPoint,
	    		GraphicsOptionPointSize -> pointSize,
	    		GraphicsOptionPointColor -> pointColor,
	    		GraphicsOptionArgument1Range3D -> argument1Range,
	    		GraphicsOptionArgument2Range3D -> argument2Range
			]
		]
	];

PlotPoints3DWithFunction[

	(* Plots specified points3D in 3D box together with specified pure function.
	   
	   Returns : Graphics3D *)
	
	
    (* points3D: {point1, point2, ...}
	   point: {x, y, z} *)
	points3D_/;MatrixQ[points3D, NumberQ], 

    (* Pure function f(x, y) *)
    pureFunction_,
	
    (* {xLabel, yLabel, zLabel} *)
	labels_,

	(* Options *)
	opts___

	] :=
  
	Module[
    
		{
			regionFunction,
			plotStyle,
			colorFunction,
			displayFunction,
			imageSize,
			max,
			mesh,
			meshStyle,
			min,
			numberOfPlotPoints,
			viewPoint,
			offset,
			pointGraphics,
			functionGraphics,
			x,
			xRange,
			y,
			yRange,
    		pointSize,
    		pointColor,
			argument1Range,
			argument2Range
		},
	
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    viewPoint = GraphicsOptionViewPoint3D/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    numberOfPlotPoints = GraphicsOptionNumberOfPlotPoints/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    colorFunction = GraphicsOptionColorFunction/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    mesh = GraphicsOptionIsMesh/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    meshStyle = GraphicsOptionMeshStyle/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    plotStyle = GraphicsOptionPlotStyle3D/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    regionFunction = GraphicsOptionRegionFunction/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	    argument1Range = GraphicsOptionArgument1Range3D/.{opts}/.Options[GraphicsOptionsPlotRange3D];
	    argument2Range = GraphicsOptionArgument2Range3D/.{opts}/.Options[GraphicsOptionsPlotRange3D];

		If[Length[argument1Range] == 0,

			min = Min[points3D[[All, 1]]];
			max = Max[points3D[[All, 1]]];
	    	offset = (max - min)/40.0;
		    min = min - offset;
		    max = max + offset;
			xRange = {min, max},
			(* xRange = {Min[points3D[[All, 1]]], Max[points3D[[All, 1]]]}, *)

			xRange = argument1Range
		];

		If[Length[argument2Range] == 0,

			min = Min[points3D[[All, 2]]];
			max = Max[points3D[[All, 2]]];
	    	offset = (max - min)/40.0;
		    min = min - offset;
		    max = max + offset;
			yRange = {min, max},
			(* yRange = {Min[points3D[[All, 2]]], Max[points3D[[All, 2]]]}, *)
			
			yRange = argument2Range
		];

		pointGraphics = 
			ListPointPlot3D[
				points3D, 
				AxesLabel -> {labels[[1]], labels[[2]], labels[[3]]}, 
				AspectRatio -> 1, 
				BoxRatios -> {1, 1, 1}, 
				PlotRange -> {xRange, yRange, Full}, 
				PlotStyle -> {{PointSize[pointSize], pointColor}}, 
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				Ticks -> Automatic, 
				ViewPoint -> viewPoint
			];

		functionGraphics = 
			Plot3D[
				pureFunction[x, y], 
				{x, xRange[[1]], xRange[[2]]}, 
				{y, yRange[[1]], yRange[[2]]}, 
				AspectRatio -> 1, 
				AxesLabel -> {labels[[1]], labels[[2]], labels[[3]]}, 
				BoxRatios -> {1, 1, 1}, 
				ColorFunction -> colorFunction, 
				Mesh -> mesh, 
				MeshStyle -> meshStyle, 
				PlotPoints -> numberOfPlotPoints, 
				PlotRange -> {Full, Full, Full}, 
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				Ticks -> Automatic, 
				ViewPoint -> viewPoint, 
				PlotStyle -> plotStyle,
				RegionFunction -> regionFunction
			];
	
		Return[
			Show[
				{pointGraphics, functionGraphics}, 
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
		]
	];

Plot3dDataSetWithFunction[

	(* Plots 3D data set in 3D box together with specified pure function.
	   
	   Returns : Graphics3D *)
	
	
	(* 3D dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2}
	   output: {outputValue} *)
	dataSet3D_,

    (* Pure function f(x, y) *)
    pureFunction_,
	
    (* {xLabel, yLabel, zLabel} *)
	labels_,

	(* Options *)
	opts___

	] :=
  
	Module[
    
		{
			regionFunction,
			plotStyle,
			colorFunction,
			displayFunction,
			imageSize,
			mesh,
			meshStyle,
			numberOfPlotPoints,
			viewPoint,
			points3D,
			i,
			ioPair,
			argument1Range,
			argument2Range,
			pointSize,
			pointColor
		},
	
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    viewPoint = GraphicsOptionViewPoint3D/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    numberOfPlotPoints = GraphicsOptionNumberOfPlotPoints/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    colorFunction = GraphicsOptionColorFunction/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    mesh = GraphicsOptionIsMesh/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    meshStyle = GraphicsOptionMeshStyle/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    plotStyle = GraphicsOptionPlotStyle3D/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    regionFunction = GraphicsOptionRegionFunction/.{opts}/.Options[GraphicsOptionsGraphics3D];
	    argument1Range = GraphicsOptionArgument1Range3D/.{opts}/.Options[GraphicsOptionsPlotRange3D];
	    argument2Range = GraphicsOptionArgument2Range3D/.{opts}/.Options[GraphicsOptionsPlotRange3D];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];

		points3D = 
			Table[
				ioPair = dataSet3D[[i]];
				{ioPair[[1, 1]], ioPair[[1, 2]], ioPair[[2, 1]]},
				
				{i, Length[dataSet3D]}
			];
	
		Return[
			PlotPoints3DWithFunction[
				points3D,
				pureFunction,
				labels,
	    		GraphicsOptionDisplayFunction -> displayFunction,
	    		GraphicsOptionViewPoint3D -> viewPoint,
	    		GraphicsOptionNumberOfPlotPoints -> numberOfPlotPoints,
	    		GraphicsOptionColorFunction -> colorFunction,
	    		GraphicsOptionIsMesh -> mesh,
	    		GraphicsOptionMeshStyle -> meshStyle,
	    		GraphicsOptionPlotStyle3D -> plotStyle,
	    		GraphicsOptionArgument1Range3D -> argument1Range,
	    		GraphicsOptionArgument2Range3D -> argument2Range,
				GraphicsOptionRegionFunction -> regionFunction,
				GraphicsOptionImageSize -> imageSize,
				GraphicsOptionPointSize -> pointSize,
				GraphicsOptionPointColor -> pointColor
			]
		]
	];

PlotBarChart[

	(* Displays bar chart.

	   Returns : Graphics *)

    
    (* {Value1OfBar1, Value1OfBar2, ...} *)
    barValues_/;VectorQ[barValues, NumberQ],
    
    (* {chartLabels, yLabel, plotLabel} 
       chartLabels: {barLabel1, barLabel2, ...} *)
	labels_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
    	{
    		chartLabels,
    		yLabel,
    		plotLabel,
    		imageSize
    	},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
  
		chartLabels = labels[[1]];
  		yLabel = labels[[2]];
  		plotLabel = labels[[3]];
  		
    	Return[
			BarChart[
				barValues,
				ChartElementFunction -> "GlassRectangle", 
				ChartStyle -> "Pastel",
				ChartBaseStyle -> EdgeForm[None],
				PlotLabel -> plotLabel,
				AxesLabel -> yLabel,
		    	BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"},
		    	ChartLabels -> chartLabels,
		    	LabelingFunction -> (Placed[#, Above] &),
		    	ImageSize -> imageSize
		    ]
    	]
	];

PlotGroupedBarChart[

	(* Displays bar chart with groups of bars.

	   Returns : Graphics *)

    
    (* {{Value1OfBar1, Value2OfBar1, ...}, {Value1OfBar2, Value2OfBar2, ...}, ...} *)
    barValues_/;MatrixQ[barValues, NumberQ],
    
    (* {chartLabels, yLabel, plotLabel} 
       chartLabels: {singleBarLabels, barGroupLabels} *)
	labels_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
    	{
    		chartLabels,
    		singleBarLabels,
    		barGroupLabels,
    		yLabel,
    		plotLabel,
    		imageSize
    	},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
  
		chartLabels = labels[[1]];
		singleBarLabels = chartLabels[[1]];
		barGroupLabels = chartLabels[[2]];
  		yLabel = labels[[2]];
  		plotLabel = labels[[3]];
  		
    	Return[
			BarChart[
				barValues,
				ChartElementFunction -> "GlassRectangle", 
				ChartStyle -> "Pastel",
				ChartBaseStyle -> EdgeForm[None],
				PlotLabel -> plotLabel,
				BarSpacing -> {0, 0.2},
				AxesLabel -> yLabel,
		    	BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"},
		    	ChartLabels -> {Placed[barGroupLabels, Axis],Placed[singleBarLabels, Axis]},
		    	LabelingFunction -> (Placed[#, Above] &),
		    	ImageSize -> imageSize
		    ]
    	]
	];

PlotIndexedPoints2D[

	(* Displays list plot of list yList (in blue) in interval [1, x]

	   Returns : Graphics *)
    

	(* {yValue1, yValue2, ...} *)
    yList_/;VectorQ[yList, NumberQ],
    
    x_?IntegerQ,
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
    ] :=
  
	Module[
    
    	{
			displayFunction,
			imageSize,
    		yOffset, 
    		yMin, 
    		yMax, 
    		xOffset, 
    		xMin, 
    		xMax,
    		pointSize,
    		pointColor
    	},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	
	    xOffset = (x - 1.0)/20.0;
	    xMin = 1 - xOffset;
	    xMax = x + xOffset;
	    
	    yMin = Min[yList];
	    yMax = Max[yList];
	    yOffset = (yMax - yMin)/20.0;
	    yMin = yMin - yOffset;
	    yMax = yMax + yOffset;
    
    	Return[
		    ListPlot[
		    	yList, 
		    	Axes -> False, 
		    	Frame -> True, 
		    	FrameTicks -> Automatic, 
		    	FrameLabel -> {labels[[1]], labels[[2]]}, 
		    	PlotLabel -> labels[[3]], 
		    	PlotStyle -> {PointSize[pointSize], pointColor}, 
		    	PlotRange -> {{xMin, xMax}, {yMin, yMax}}, 
		    	BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
		    	DisplayFunction -> displayFunction,
				ImageSize -> imageSize
		    ]
    	]
	];

PlotIndexedLine2D[

	(* Displays joined list plot of list yList (in black) in interval [1, x]

	   Returns : Graphics *)

    
	(* {yValue1, yValue2, ...} *)
    yList_/;VectorQ[yList, NumberQ],
    
    x_?IntegerQ,
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
    ] :=
  
	Module[
    
		{
			displayFunction,
			imageSize,
			yOffset, 
			yMin, 
			yMax, 
			xOffset, 
			xMin, 
			xMax,
			plotStyle
		},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    plotStyle = GraphicsOptionLinePlotStyle/.{opts}/.Options[GraphicsOptionsLinePlotStyle];
	
	    xOffset = (x - 1.0)/20.0;
	    xMin = 1 - xOffset;
	    xMax = x + xOffset;
	    
	    yMin = Min[yList];
	    yMax = Max[yList];
	    yOffset = (yMax - yMin)/20.0;
	    yMin = yMin - yOffset;
	    yMax = yMax + yOffset;
    
    	Return[
			ListLinePlot[
				yList, 
				Axes -> False, 
				Frame -> True, 
				FrameTicks -> Automatic, 
				FrameLabel -> {labels[[1]], labels[[2]]}, 
				PlotLabel -> labels[[3]], 
				PlotStyle -> plotStyle, 
				PlotRange -> {{xMin, xMax}, {yMin, yMax}}, 
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
    	]
	];

PlotMultipleLines2D[
    
	(* Displays multiple joined points2D in their individual specified plot.

	   Returns : Graphics *)


    (* points2DWithPlotStyleList: {{points2D_1, plotStyle1}, {points2D_2, plotStyle2}, ...}
       points2D: {point1, point2, ...} or {}
       point: {x, y}
       plotStyle: {Line style, color} *)
    points2DWithPlotStyleList_,
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
    	{
	   		argumentRange,
    		currentPlotStyle,
			currentPoints2D,
    		currentXMin,
    		currentXMax,
    		currentYMin,
    		currentYMax,
			displayFunction,
			imageSize,
			i,
			points2DList,
			plotStyle,
			functionValueRange,
    		yOffset, 
    		yMin, 
    		yMax, 
    		xOffset, 
    		xMin, 
    		xMax
    	},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    argumentRange = GraphicsOptionArgumentRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];
	    functionValueRange = GraphicsOptionFunctionValueRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];

		If[Length[argumentRange] == 0,

			xMin = Infinity;
			xMax = -Infinity;
		    Do[
		    	currentPoints2D = points2DWithPlotStyleList[[i, 1]];
		    	currentXMin = Min[currentPoints2D[[All, 1]]];
		    	If[currentXMin < xMin,
		    		xMin = currentXMin
		    	];
		    	currentXMax = Max[currentPoints2D[[All, 1]]];
		    	If[currentXMax > xMax,
		    		xMax = currentXMax
		    	],
		    	
		    	{i, Length[points2DWithPlotStyleList]}
		    ],
		    
			xMin = argumentRange[[1]];
		    xMax = argumentRange[[2]]
		];
	    xOffset = (xMax - xMin)/20.0;
	    xMin = xMin - xOffset;
	    xMax = xMax + xOffset;
		
		If[Length[functionValueRange] == 0,
			yMin = Infinity;
			yMax = -Infinity;
		    Do[
		    	currentPoints2D = points2DWithPlotStyleList[[i, 1]];
		    	currentYMin = Min[currentPoints2D[[All, 2]]];
		    	If[currentYMin < yMin,
		    		yMin = currentYMin
		    	];
		    	currentYMax = Max[currentPoints2D[[All, 2]]];
		    	If[currentYMax > yMax,
		    		yMax = currentYMax
		    	],
		    	
		    	{i, Length[points2DWithPlotStyleList]}
		    ],
		    
		    yMin = functionValueRange[[1]];
		    yMax = functionValueRange[[2]]
		];
	    yOffset = (yMax - yMin)/20.0;
	    yMin = yMin - yOffset;
	    yMax = yMax + yOffset;

    	points2DList = {};
    	plotStyle = {};
	    Do[
	    	currentPoints2D = points2DWithPlotStyleList[[i, 1]];
	    	currentPlotStyle = points2DWithPlotStyleList[[i, 2]];
    		AppendTo[points2DList, currentPoints2D];
    		AppendTo[plotStyle, currentPlotStyle],
	    	
	    	{i, Length[points2DWithPlotStyleList]}
	    ];
    
    	Return[
			ListLinePlot[
				points2DList, 
				Axes -> False, 
				Frame -> True, 
				FrameTicks -> Automatic, 
				FrameLabel -> {labels[[1]], labels[[2]]}, 
				PlotLabel -> labels[[3]], 
				PlotRange -> {{xMin, xMax}, {yMin, yMax}}, 
 				PlotStyle -> plotStyle,
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
    	]
	];

PlotMultiplePoints2D[
    
	(* Displays multiple points2D in individual specified plot style.

	   Returns : Graphics *)


    (* points2DWithPlotStyleList: {{points2D_1, plotStyle1}, {points2D_2, plotStyle2}, ...}
       points2D: {point1, point2, ...} or {}
       point: {x, y}
       plotStyle: {point size, color} *)
    points2DWithPlotStyleList_,
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
    	{
	   		argumentRange,
    		currentPlotStyle,
			currentPoints2D,
    		currentXMin,
    		currentXMax,
    		currentYMin,
    		currentYMax,
			displayFunction,
			imageSize,
			i,
			points2DList,
			plotStyle,
			functionValueRange,
    		yOffset, 
    		yMin, 
    		yMax, 
    		xOffset, 
    		xMin, 
    		xMax
    	},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    argumentRange = GraphicsOptionArgumentRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];
	    functionValueRange = GraphicsOptionFunctionValueRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];

		If[Length[argumentRange] == 0,

			xMin = Infinity;
			xMax = -Infinity;
		    Do[
		    	currentPoints2D = points2DWithPlotStyleList[[i, 1]];
		    	currentXMin = Min[currentPoints2D[[All, 1]]];
		    	If[currentXMin < xMin,
		    		xMin = currentXMin
		    	];
		    	currentXMax = Max[currentPoints2D[[All, 1]]];
		    	If[currentXMax > xMax,
		    		xMax = currentXMax
		    	],
		    	
		    	{i, Length[points2DWithPlotStyleList]}
		    ],
		    
			xMin = argumentRange[[1]];
		    xMax = argumentRange[[2]]
		];
	    xOffset = (xMax - xMin)/20.0;
	    xMin = xMin - xOffset;
	    xMax = xMax + xOffset;
		
		If[Length[functionValueRange] == 0,
			yMin = Infinity;
			yMax = -Infinity;
		    Do[
		    	currentPoints2D = points2DWithPlotStyleList[[i, 1]];
		    	currentYMin = Min[currentPoints2D[[All, 2]]];
		    	If[currentYMin < yMin,
		    		yMin = currentYMin
		    	];
		    	currentYMax = Max[currentPoints2D[[All, 2]]];
		    	If[currentYMax > yMax,
		    		yMax = currentYMax
		    	],
		    	
		    	{i, Length[points2DWithPlotStyleList]}
		    ],
		    
		    yMin = functionValueRange[[1]];
		    yMax = functionValueRange[[2]]
		];
	    yOffset = (yMax - yMin)/20.0;
	    yMin = yMin - yOffset;
	    yMax = yMax + yOffset;

    	points2DList = {};
    	plotStyle = {};
	    Do[
	    	currentPoints2D = points2DWithPlotStyleList[[i, 1]];
	    	currentPlotStyle = points2DWithPlotStyleList[[i, 2]];
    		AppendTo[points2DList, currentPoints2D];
    		AppendTo[plotStyle, currentPlotStyle],
	    	
	    	{i, Length[points2DWithPlotStyleList]}
	    ];
    
    	Return[
			ListPlot[
				points2DList, 
				Axes -> False, 
				Frame -> True, 
				FrameTicks -> Automatic, 
				FrameLabel -> {labels[[1]], labels[[2]]}, 
				PlotLabel -> labels[[3]], 
				PlotRange -> {{xMin, xMax}, {yMin, yMax}}, 
 				PlotStyle -> plotStyle,
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
    	]
	];

PlotResiduals[

	(* Displays residuals.

	   Returns : Graphics *)
    

	(* {yValue1, yValue2, ...} *)
    residuals_/;VectorQ[residuals, NumberQ],
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
    ] :=
  
	Module[
    
    	{
			displayFunction,
			imageSize,
    		yOffset, 
    		yMin, 
    		yMax, 
    		x,
    		xOffset, 
    		xMin, 
    		xMax,
    		pointSize,
    		pointColor
    	},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	
		x = Length[residuals];
	    xOffset = (x - 1.0)/20.0;
	    xMin = 1 - xOffset;
	    xMax = x + xOffset;
	    
	    yMin = Min[residuals];
	    yMax = Max[residuals];
	    yOffset = (yMax - yMin)/20.0;
	    yMin = yMin - yOffset;
	    yMax = yMax + yOffset;
    
    	Return[
		    ListPlot[
		    	residuals, 
		    	Axes -> False, 
		    	Frame -> True, 
		    	FrameTicks -> Automatic, 
		    	FrameLabel -> {labels[[1]], labels[[2]]}, 
		    	PlotLabel -> labels[[3]], 
		    	PlotStyle -> {PointSize[pointSize], pointColor}, 
		    	PlotRange -> {{xMin, xMax}, {yMin, yMax}}, 
		    	BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				Prolog -> {Thickness[0.005], Black, Line[{{1.0, 0.0}, {x, 0.0}}]}, 
				Filling -> Axis,
		    	DisplayFunction -> displayFunction,
				ImageSize -> imageSize
		    ]
    	]
	];

PlotSilhouetteWidths[

	(* Displays silhouette widths.

	   Returns : Graphics *)
    

	(* {silhouette width 1, silhouette width 2, ...} *)
    silhouetteWidths_/;VectorQ[silhouetteWidths, NumberQ],
    
	(* Mean value of silhouette widths *)
    meanSilhouetteWidth_?NumberQ,
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
    ] :=
  
	Module[
    
    	{
    		i,
			displayFunction,
			imageSize,
    		yMin, 
    		yMax, 
    		yOffset,
    		x,
    		xOffset, 
    		xMin, 
    		xMax,
    		pointSize,
    		pointColor,
    		rectangleColor,
    		sortedSilhouetteWidths,
    		silhouetteWidthsForDisplay,
    		numberOfPointsForDisplay
    	},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	    numberOfPointsForDisplay = GraphicsOptionNumberOfPointsForDisplay/.{opts}/.Options[GraphicsOptionsPoint];
	
		x = N[Length[silhouetteWidths]];
	    xOffset = (x - 1.0)/20.0;
	    xMin = 1.0 - xOffset;
	    xMax = x + xOffset;
	    
	    yMin = Min[silhouetteWidths];
	    If[yMin > 0.0, yMin = 0.0];
	    yMax = Max[silhouetteWidths];
	    yOffset = (yMax - yMin)/20.0;
	    yMin = yMin - yOffset;
	    yMax = yMax + yOffset;

		If[meanSilhouetteWidth >= 0.7,

			(* "Good" = green *)
    		rectangleColor = RGBColor[0, 1, 0, 0.2],
			
			If[meanSilhouetteWidth >= 0.5,

				(* "Medium" = yellow *)
	    		rectangleColor = RGBColor[0, 1, 1, 0.2],

				(* "Bad" = red *)
	    		rectangleColor = RGBColor[1, 0, 0, 0.2]
			]
		];    
    	
    	If[Length[silhouetteWidths] > numberOfPointsForDisplay,

			sortedSilhouetteWidths = Sort[silhouetteWidths];
    		silhouetteWidthsForDisplay = 
    			Table[
    				{Round[i],sortedSilhouetteWidths[[Round[i]]]},
    				
    				{i, 1.0, x, (x - 1.0)/(numberOfPointsForDisplay - 1.0)}
    			],
    		
    		silhouetteWidthsForDisplay = Sort[silhouetteWidths]
    	];
    	
		Return[
		    ListPlot[
		    	silhouetteWidthsForDisplay, 
		    	Axes -> False, 
		    	Frame -> True, 
		    	FrameTicks -> Automatic, 
		    	FrameLabel -> {labels[[1]], labels[[2]]}, 
		    	PlotLabel -> labels[[3]], 
		    	PlotStyle -> {PointSize[pointSize], pointColor}, 
		    	PlotRange -> {{xMin, xMax}, {yMin, yMax}}, 
		    	BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				Epilog ->
					{
						rectangleColor, 
						Rectangle[{1.0, 0.0}, {x, meanSilhouetteWidth}]
					},
				Filling -> Axis,
		    	DisplayFunction -> displayFunction,
				ImageSize -> imageSize
		    ]
		]
	];

PlotTwoIndexedLines2D[

	(* Displays joined list plot of lists yList1 (in blue) and yList2 (in red) in interval [1, x]

	   Returns : Graphics *)

    
	(* {yValue1, yValue2, ...} *)
    yList1_/;VectorQ[yList1, NumberQ],
    
    (* {yValue1, yValue2, ...} *)
    yList2_/;VectorQ[yList2, NumberQ],
    
    x_?NumberQ,
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
    	{
			displayFunction,
			imageSize,
    		yOffset, 
    		yMin, 
    		yMax, 
    		xOffset, 
    		xMin, 
    		xMax
    	},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	
		xOffset = (x - 1.0)/20.0;
	    xMin = 1 - xOffset;
	    xMax = x + xOffset;
    
	    yMin = Min[Min[yList1], Min[yList2]];
	    yMax = Max[Max[yList1], Max[yList2]];
	    yOffset = (yMax - yMin)/20.0;
	    yMin = yMin - yOffset;
	    yMax = yMax + yOffset;
    
    	Return[
			ListLinePlot[
				{yList1, yList2}, 
				Axes -> False, 
				Frame -> True, 
				FrameTicks -> Automatic, 
				FrameLabel -> {labels[[1]], labels[[2]]}, 
				PlotLabel -> labels[[3]], 
				PlotJoined -> True, 
				PlotStyle -> {{Thickness[0.005], RGBColor[0, 0, 1]}, {Thickness[0.001], RGBColor[1, 0, 0]}}, 
				PlotRange -> {{xMin, xMax}, {yMin, yMax}}, 
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
    	]
	];

PlotUpToFourLines2D[
    
	(* Displays up to 4 joined points2D sets as joined lines in different colours.

	   Returns : Graphics *)


    (* points2D: {point1, point2, ...} or {}
       point: {x, y} *)
    points2DRed_,
    
    (* points2D: {point1, point2, ...} or {}
       point: {x, y} *)
    points2DGreen_,
    
    (* points2D: {point1, point2, ...} or {}
       point: {x, y} *)
    points2DBlue_,
    
    (* points2D: {point1, point2, ...} or {}
       point: {x, y} *)
    points2DBlack_,
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
    	{
			displayFunction,
			imageSize,
			plotStyle,
			pointsList,
    		yOffset, 
    		yMin, 
    		yMax, 
    		xOffset, 
    		xMin, 
    		xMax,
    		joinedPoints2D,
    		xValues,
    		yValues
    	},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	
		joinedPoints2D = Join[points2DRed, points2DGreen, points2DBlue, points2DBlack];
		xValues = joinedPoints2D[[All, 1]];
		yValues = joinedPoints2D[[All, 2]];
		
		xMin = Min[xValues];
		xMax = Max[xValues];
	    xOffset = (xMax - xMin)/20.0;
	    xMin = xMin - xOffset;
	    xMax = xMax + xOffset;
    
		yMin = Min[yValues];
		yMax = Max[yValues];
	    yOffset = (yMax - yMin)/20.0;
	    yMin = yMin - yOffset;
	    yMax = yMax + yOffset;
    	
    	pointsList = {};
    	plotStyle = {};
    	If[Length[points2DRed] > 0, 
    		AppendTo[pointsList, points2DRed];
    		AppendTo[plotStyle, {Thickness[0.005], RGBColor[1, 0, 0]}]
    	];
    	If[Length[points2DGreen] > 0, 
    		AppendTo[pointsList, points2DGreen];
    		AppendTo[plotStyle, {Thickness[0.005], RGBColor[0, 1, 0]}]
    	];
    	If[Length[points2DBlue] > 0, 
    		AppendTo[pointsList, points2DBlue];
    		AppendTo[plotStyle, {Thickness[0.005], RGBColor[0, 0, 1]}]
    	];
    	If[Length[points2DBlack] > 0, 
    		AppendTo[pointsList, points2DBlack];
    		AppendTo[plotStyle, {Thickness[0.005], RGBColor[0, 0, 0]}]
    	];
    	
    	Return[
			ListLinePlot[
				pointsList, 
				Axes -> False, 
				Frame -> True, 
				FrameTicks -> Automatic, 
				FrameLabel -> {labels[[1]], labels[[2]]}, 
				PlotLabel -> labels[[3]], 
				PlotJoined -> True, 
				PlotStyle -> plotStyle, 
				PlotRange -> {{xMin, xMax}, {yMin, yMax}}, 
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
    	]
	];

PlotUpToFourPoint2DSets[
    
	(* Displays up to 4 points2D sets in different colours.

	   Returns : Graphics *)


    (* points2D: {point1, point2, ...} or {}
       point: {x, y} *)
    points2DRed_,
    
    (* points2D: {point1, point2, ...} or {}
       point: {x, y} *)
    points2DGreen_,
    
    (* points2D: {point1, point2, ...} or {}
       point: {x, y} *)
    points2DBlue_,
    
    (* points2D: {point1, point2, ...} or {}
       point: {x, y} *)
    points2DBlack_,
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
    	{
			displayFunction,
			imageSize,
			pointsList,
			plotStyle,
    		yOffset, 
    		yMin, 
    		yMax, 
    		xOffset, 
    		xMin, 
    		xMax,
    		pointSize,
    		joinedPoints2D,
    		xValues,
    		yValues
    	},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	
		joinedPoints2D = Join[points2DRed, points2DGreen, points2DBlue, points2DBlack];
		xValues = joinedPoints2D[[All, 1]];
		yValues = joinedPoints2D[[All, 2]];
		
		xMin = Min[xValues];
		xMax = Max[xValues];
	    xOffset = (xMax - xMin)/20.0;
	    xMin = xMin - xOffset;
	    xMax = xMax + xOffset;
    
		yMin = Min[yValues];
		yMax = Max[yValues];
	    yOffset = (yMax - yMin)/20.0;
	    yMin = yMin - yOffset;
	    yMax = yMax + yOffset;

    	pointsList = {};
    	plotStyle = {};
    	If[Length[points2DRed] > 0, 
    		AppendTo[pointsList, points2DRed];
    		AppendTo[plotStyle, {PointSize[pointSize], RGBColor[1, 0, 0, 0.8]}]
    	];
    	If[Length[points2DGreen] > 0, 
    		AppendTo[pointsList, points2DGreen];
    		AppendTo[plotStyle, {PointSize[pointSize], RGBColor[0, 1, 0, 0.8]}]
    	];
    	If[Length[points2DBlue] > 0, 
    		AppendTo[pointsList, points2DBlue];
    		AppendTo[plotStyle, {PointSize[pointSize], RGBColor[0, 0, 1, 0.8]}]
    	];
    	If[Length[points2DBlack] > 0, 
    		AppendTo[pointsList, points2DBlack];
    		AppendTo[plotStyle, {PointSize[pointSize], RGBColor[0, 0, 0, 0.8]}]
    	];
    
    	Return[
			ListPlot[
				pointsList, 
				Axes -> False, 
				Frame -> True, 
				FrameTicks -> Automatic, 
				FrameLabel -> {labels[[1]], labels[[2]]}, 
				PlotLabel -> labels[[3]], 
				PlotRange -> {{xMin, xMax}, {yMin, yMax}}, 
				PlotStyle -> plotStyle,
				BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 10, FontWeight -> "Plain"}, 
				DisplayFunction -> displayFunction,
				ImageSize -> imageSize
			]
    	]
	];

PlotXyErrorData[

	(* Displays points2D in xyErrorData (in blue).

	   Returns : Graphics *)


    (* xyErrorData: {{x1, y1, error1}, ..., {x<numberOfData>, y<numberOfData>, error<numberOfData>}} *)
    xyErrorData_/;MatrixQ[xyErrorData, NumberQ],
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
    ] :=
  
	Module[
    
    	{
	   		argumentRange,
			displayFunction,
			imageSize,
			functionValueRange,
    		pointSize,
    		pointColor
    	},
    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    argumentRange = GraphicsOptionArgumentRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];
	    functionValueRange = GraphicsOptionFunctionValueRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	
    	Return[
    		PlotPoints2D[
				xyErrorData[[All, {1, 2}]],
				labels,
				GraphicsOptionDisplayFunction -> displayFunction,
				GraphicsOptionArgumentRange2D -> argumentRange,
				GraphicsOptionFunctionValueRange2D -> functionValueRange,
				GraphicsOptionPointSize -> pointSize,
				GraphicsOptionPointColor -> pointColor,
				GraphicsOptionImageSize -> imageSize
    		]
    	]
	];

PlotXyErrorDataAboveFunction[

	(* Displays points2D in xyErrorData (in blue) above function (in black)

	   Returns : Graphics *)

    
    (* xyErrorData: {{x1, y1, error1}, ..., {x<numberOfData>, y<numberOfData>, error<numberOfData>}} *)
    xyErrorData_/;MatrixQ[xyErrorData, NumberQ],
    
    (* Pure function*)
    pureFunction_,
    
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
    ] :=
  
	Module[
    
	    {
			displayFunction,
			imageSize,
    		pointSize,
    		pointColor,
    		argumentRange,
    		functionValueRange,
    		plotStyle
	    },
	    
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	    argumentRange = GraphicsOptionArgumentRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];
	    functionValueRange = GraphicsOptionFunctionValueRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];
	    plotStyle = GraphicsOptionLinePlotStyle/.{opts}/.Options[GraphicsOptionsLinePlotStyle];
	
	    Return[
	    	PlotPoints2DAboveFunction[
				xyErrorData[[All, {1, 2}]],
	    		pureFunction, 
	    		labels, 
		    	GraphicsOptionDisplayFunction -> displayFunction,
				GraphicsOptionPointSize -> pointSize,
				GraphicsOptionPointColor -> pointColor,
				GraphicsOptionImageSize -> imageSize,
				GraphicsOptionArgumentRange2D -> argumentRange, 
				GraphicsOptionFunctionValueRange2D -> functionValueRange, 
				GraphicsOptionLinePlotStyle -> plotStyle,
				GraphicsOptionImageSize -> imageSize  
	    	]
	    ]
	];

PlotXyErrorDataAboveFunctions[

	(* Displays points2D in xyErrorData (in blue) above multiple functions.

	   Returns : Graphics *)

    
    (* xyErrorData: {{x1, y1, error1}, ..., {x<numberOfData>, y<numberOfData>, error<numberOfData>}} *)
    xyErrorData_/;MatrixQ[xyErrorData, NumberQ],
    
	(* Pure functions: {PureFunction1, PureFunction2, ...}*)      
	pureFunctions_,

	(* {xMin, xMax} *)
	argumentRange_/;VectorQ[argumentRange, NumberQ],

	(* {yMin, yMax} *)
	functionValueRange_/;VectorQ[functionValueRange, NumberQ],
      
    (* Plot style *)
	plotStyle_,
      
    (* {xLabel, yLabel, diagramLabel} *)
	labels_,
	
	(* Options *)
	opts___
    
    ] :=
  
	Module[
    
		{
			displayFunction,
			imageSize,
    		pointSize,
    		pointColor
		},
      
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	
      	Return[
			PlotPoints2DAboveMultipleFunctions[
				xyErrorData[[All, {1, 2}]],
				pureFunctions,
				argumentRange,
				functionValueRange,
				plotStyle,
				labels,
				GraphicsOptionDisplayFunction -> displayFunction,
				GraphicsOptionPointSize -> pointSize,
				GraphicsOptionPointColor -> pointColor,
				GraphicsOptionImageSize -> imageSize
			]
      	]
	];

ShowClassificationResult[

	(* Shows (general) classification results according to named property list.

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
	   outputValue: 0.0/1.0
	   Data set must be a classification data set, i.e. the output components must 0/1 code a class,
	   i.e. class 4 of 5 must be coded {0.0, 0.0, 0.0, 1.0, 0.0} 
	   NOTE: Data set MUST be in original units *)
    classificationDataSet_,
    
	(* Pure function of inputs
	   inputs = {input1, input2, ...} 
	   input: {inputComponent1, inputComponent2, ...} *)
	pureFunction_,
	
	(* Options *)
	opts___
    
	] :=

	Module[
    
    	{
			minMaxIndex,
			imageSize,
    		i,
    		namedProperty
    	},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    minMaxIndex = GraphicsOptionMinMaxIndex/.{opts}/.Options[GraphicsOptionsIndex];
    	
    	Do[
    		namedProperty = namedPropertyList[[i]];
    		ShowSingleClassificationResult[
    			namedProperty, 
    			classificationDataSet, 
    			pureFunction,
				GraphicsOptionImageSize -> imageSize,
				GraphicsOptionMinMaxIndex -> minMaxIndex
    		],
    		
    		{i, Length[namedPropertyList]}
    	]
	];

ShowClassificationScan[

	(* Shows result of classification scan of clustered training sets.

	   Returns: Nothing *)


	(* classificationTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, methodInfo1}, {trainingAndTestSet2, methodInfo2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, classification result in percent for training set}, {trainingFraction, classification result in percent for training set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)
	classificationTrainingScanResult_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			i,
			labels,
			scanReport,
			trainingPoints2D,
			testPoints2D,
			trainingPoints2DWithPlotStyle,
			testPoints2DWithPlotStyle,
			points2DWithPlotStyleList,
			bestIndexList,
			maximum,
			imageSize,
			displayFunction
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];

		scanReport = classificationTrainingScanResult[[2]];
		trainingPoints2D = scanReport[[All, 1]];
		testPoints2D = scanReport[[All, 2]];

		labels = {"Training fraction", "Correct classifications [%]", "Training (green), Test (red)"};
		trainingPoints2DWithPlotStyle = {trainingPoints2D, {Thickness[0.005], Green}};
		testPoints2DWithPlotStyle = {testPoints2D, {Thickness[0.005], Red}};
		points2DWithPlotStyleList = {trainingPoints2DWithPlotStyle, testPoints2DWithPlotStyle};
		Print[
			PlotMultipleLines2D[
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
		Print["Best test set classification with index = ", bestIndexList]
	];

ShowDataSetInfo[

	(* Shows data set information according to named property list.

	   Returns: Nothing *)


	(* Properties to be analyzed: 
	   Full list: 
	   {
	       "IoPairs",
	       "InputComponents",
	       "OutputComponents",
	       "ClassCount"
	    } *)
 	namedPropertyList_,
    
    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...} *)
    dataSet_
    
	] :=

	Module[
    
    	{
    		i,
    		namedProperty
    	},
    	
    	Do[
    		namedProperty = namedPropertyList[[i]];
    		ShowSingleDataSetInfo[namedProperty, dataSet],
    		
    		{i, Length[namedPropertyList]}
    	]
	];

ShowInputRelevanceClass[

	(* Shows inputComponentRelevanceListForClassification.

	   Returns: Nothing *)


	(* inputComponentRelevanceListForClassification: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, info data structure}
	   trainingSetResult: {numberOfRemovedInputs, (best) correct classification in percent of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best correct classification in percent of test set} *)
	inputComponentRelevanceListForClassification_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			trainingPoints2D,
			testPoints2D,
			labels,
			displayFunction,
			imageSize,
			trainingPoints2DWithPlotStyle,
			testPoints2DWithPlotStyle,
			points2DWithPlotStyleList,
			longestInputComponentList 
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];

		trainingPoints2D = inputComponentRelevanceListForClassification[[All, 1]];
		testPoints2D = inputComponentRelevanceListForClassification[[All, 2]];
		If[Length[Flatten[testPoints2D]] == 0,
			testPoints2D = {}
		];

		If[Length[testPoints2D] > 0,
			
			(* Training and test set *)
			labels = {"Number of input components", "Correct classification [%]", "Training (green), Test (red)"};
			trainingPoints2DWithPlotStyle = {trainingPoints2D, {Thickness[0.005], Green}};
			testPoints2DWithPlotStyle = {testPoints2D, {Thickness[0.005], Red}};
			points2DWithPlotStyleList = {trainingPoints2DWithPlotStyle, testPoints2DWithPlotStyle};
			Print[
				PlotMultipleLines2D[
					points2DWithPlotStyleList, 
					labels,
					GraphicsOptionImageSize -> imageSize,
					GraphicsOptionDisplayFunction -> displayFunction
				]
			],

			(* Training set only *)
			labels = {"Number of input components", "Correct classification [%]", "Training (green)"};
			trainingPoints2DWithPlotStyle = {trainingPoints2D, {Thickness[0.005], Green}};
			points2DWithPlotStyleList = {trainingPoints2DWithPlotStyle};
			Print[
				PlotMultipleLines2D[
					points2DWithPlotStyleList, 
					labels,
					GraphicsOptionImageSize -> imageSize,
					GraphicsOptionDisplayFunction -> displayFunction
				]
			]
		];
		longestInputComponentList = inputComponentRelevanceListForClassification[[Length[inputComponentRelevanceListForClassification], 3]];
		Print["Input component list = ", longestInputComponentList]
	];

ShowInputRelevanceRegress[

	(* Shows inputComponentRelevanceListForRegression.

	   Returns: Nothing *)


	(* inputComponentRelevanceListForRegression: {relevance1, relevance2, ...}
	   relevance: {trainingSetResult, testSetResult, removedInputComponentList, info data structure}
	   trainingSetResult: {numberOfRemovedInputs, (best) RMSE of training set (if there is no test set)}
	   testSetResult: {numberOfRemovedInputs, best RMSE of test set} *)
	inputComponentRelevanceListForRegression_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			trainingPoints2D,
			testPoints2D,
			labels,
			displayFunction,
			imageSize,
			trainingPoints2DWithPlotStyle,
			testPoints2DWithPlotStyle,
			points2DWithPlotStyleList,
			longestInputComponentList
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];

		trainingPoints2D = inputComponentRelevanceListForRegression[[All, 1]];
		testPoints2D = inputComponentRelevanceListForRegression[[All, 2]];
		If[Length[Flatten[testPoints2D]] == 0,
			testPoints2D = {}
		];

		If[Length[testPoints2D] > 0,
			
			(* Training and test set *)
			labels = {"Number of input components", "RMSE", "Training (green), Test (red)"};
			trainingPoints2DWithPlotStyle = {trainingPoints2D, {Thickness[0.005], Green}};
			testPoints2DWithPlotStyle = {testPoints2D, {Thickness[0.005], Red}};
			points2DWithPlotStyleList = {trainingPoints2DWithPlotStyle, testPoints2DWithPlotStyle};
			Print[
				PlotMultipleLines2D[
					points2DWithPlotStyleList, 
					labels,
					GraphicsOptionImageSize -> imageSize,
					GraphicsOptionDisplayFunction -> displayFunction
				]
			],

			(* Training set only *)
			labels = {"Number of input components", "RMSE", "Training (green)"};
			trainingPoints2DWithPlotStyle = {trainingPoints2D, {Thickness[0.005], Green}};
			points2DWithPlotStyleList = {trainingPoints2DWithPlotStyle};
			Print[
				PlotMultipleLines2D[
					points2DWithPlotStyleList, 
					labels,
					GraphicsOptionImageSize -> imageSize,
					GraphicsOptionDisplayFunction -> displayFunction
				]
			]
		];
		longestInputComponentList = inputComponentRelevanceListForRegression[[Length[inputComponentRelevanceListForRegression], 3]];
		Print["Input component list = ", longestInputComponentList]
	];

ShowInputsInfo[

	(* Shows inputs information according to named property list.

	   Returns: Nothing *)


	(* Properties to be analyzed: 
	   Full list: 
	   {
	       "InputVectors",
	       "InputComponents"
	    } *)
 	namedPropertyList_,
    
	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>} *)
    inputs_/;MatrixQ[inputs, NumberQ]
    
	] :=

	Module[
    
    	{
    		i,
    		namedProperty
    	},
    	
    	Do[
    		namedProperty = namedPropertyList[[i]];
    		ShowSingleInputsInfo[namedProperty, inputs],
    		
    		{i, Length[namedPropertyList]}
    	]
	];

ShowRegressionResult[
	
	(* Shows regression results according to named property list.

	   Returns : Nothing *)

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
	   output : {outputComponent1, outputComponent2, ...} *)
    dataSet_,

	(* Pure function of inputs
	   inputs = {input1, input2, ...} 
	   input: {inputComponent1, inputComponent2, ...} *)
	pureFunction_,
	
	(* Options *)
	opts___
    
	] :=

	Module[
    
    	{
    		i,
    		namedProperty,
    		pointColor,
    		pointSize,
    		imageSize
    	},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];

    	Do[
    		namedProperty = namedPropertyList[[i]];
    		ShowSingleRegressionResult[
    			namedProperty, 
    			dataSet, 
    			pureFunction,
    			GraphicsOptionPointSize -> pointSize,
				GraphicsOptionPointColor -> pointColor,
				GraphicsOptionImageSize -> imageSize  			
    		],
    		
    		{i, Length[namedPropertyList]}
    	]
	];

ShowRegressionScan[

	(* Shows result of regression scan of clustered training sets.

	   Returns: Nothing *)


	(* regressionTrainingScanResult: {trainingAndTestSetsInfo, scanReport}
	   trainingAndTestSetsInfo: {{trainingAndTestSet1, methodInfo1}, {trainingAndTestSet2, methodInfo2}, ...}
	   trainingAndTestSetsInfo[[i]] corresponds to trainingFractionList[[i]]
	   scanReport: {scanResult1, scanResult2, ...}
	   scanResult: {{trainingFraction, RMSE for training set}, {trainingFraction, RMSE for test set}}
	   scanResult[[i]] corresponds to trainingFractionList[[i]] and trainingAndTestSetsInfo[[i]] *)
	regressionTrainingScanResult_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			i,
			labels,
			scanReport,
			trainingPoints2D,
			testPoints2D,
			trainingPoints2DWithPlotStyle,
			testPoints2DWithPlotStyle,
			points2DWithPlotStyleList,
			bestIndexList,
			minimum,
			imageSize,
			displayFunction
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];

		scanReport = regressionTrainingScanResult[[2]];
		trainingPoints2D = scanReport[[All, 1]];
		testPoints2D = scanReport[[All, 2]];

		labels = {"Training fraction", "RMSE", "Training (green), Test (red)"};
		trainingPoints2DWithPlotStyle = {trainingPoints2D, {Thickness[0.005], Green}};
		testPoints2DWithPlotStyle = {testPoints2D, {Thickness[0.005], Red}};
		points2DWithPlotStyleList = {trainingPoints2DWithPlotStyle, testPoints2DWithPlotStyle};
		Print[
			PlotMultipleLines2D[
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
		Print["Best test set regression with index = ", bestIndexList]
	];

ShowSingleClassificationResult[

	(* Shows single (general) classification result according to named property.

	   Returns: Nothing *)


	(* Properties to be analyzed: 
	   Full list: 
	   {
	       "CorrectClassification",
	       "CorrectClassificationPerClass",
	       "WrongClassificationDistribution",
	       "WrongClassificationPairs"
	   } *)
 	namedProperty_,
    
	(* classificationDataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputValue1, inputValue2, ...}
	   output: {outputValue1, outputValue2, ...} 
	   outputValue: 0.0/1.0
	   Data set must be a classification data set, i.e. the output components must 0/1 code a class,
	   i.e. class 4 of 5 must be coded {0.0, 0.0, 0.0, 1.0, 0.0} 
	   NOTE: Data set MUST be in original units *)
    classificationDataSet_,
    
	(* Pure function of inputs
	   inputs = {input1, input2, ...} 
	   input: {inputComponent1, inputComponent2, ...} *)
	pureFunction_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			minMaxIndex,
			imageSize,
			classNumber,
			classNumberDesired,
			classNumbers,
			correctPredictions,
			i,
			inputs,
			outputs,
			wrongIOPairList,
			numberOfClasses,
			classificationDataSubSet,
			correctPredictionsInPercentList,
			chartLabels,
			yLabel,
			plotLabel,
			labels,
			wrongPredictions,
			wrongPredictionList,
			wrongPredictionsInPercentList
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    minMaxIndex = GraphicsOptionMinMaxIndex/.{opts}/.Options[GraphicsOptionsIndex];

		Switch[namedProperty,
			
			(* -------------------------------------------------------------------------------- *)
			"CorrectClassification",

	    	inputs = CIP`Utility`GetInputsOfDataSet[classificationDataSet];
	    	outputs = CIP`Utility`GetOutputsOfDataSet[classificationDataSet];
	    	classNumbers = pureFunction[inputs];
		    correctPredictions = 0;
		    Do[
		    	classNumber = classNumbers[[i]];
				classNumberDesired = CIP`Utility`GetPositionOfMaximumValue[outputs[[i]]];
				If[classNumber == classNumberDesired,
					(* Correct classification *)
					correctPredictions++
				],
	      
				{i, Length[inputs]}
			];
		    Print[NumberForm[correctPredictions/Length[classificationDataSet]*100.0, 3], "% correct classifications"],

			(* -------------------------------------------------------------------------------- *)
			"CorrectClassificationPerClass",
			
			numberOfClasses = Length[classificationDataSet[[1, 2]]];
			correctPredictionsInPercentList = {};
		    Do[
		    	classificationDataSubSet = CIP`DataTransformation`GetSpecificClassDataSubSet[classificationDataSet, i];
		    	inputs = CIP`Utility`GetInputsOfDataSet[classificationDataSubSet];
		    	outputs = CIP`Utility`GetOutputsOfDataSet[classificationDataSubSet];
		    	classNumbers = pureFunction[inputs];
			    correctPredictions = 0;
			    Do[
			    	classNumber = classNumbers[[i]];
					classNumberDesired = CIP`Utility`GetPositionOfMaximumValue[outputs[[i]]];
					If[classNumber == classNumberDesired,
						(* Correct classification *)
						correctPredictions++
					],
		      
					{i, Length[inputs]}
				];
				AppendTo[correctPredictionsInPercentList, CIP`Utility`RoundTo[correctPredictions/Length[classificationDataSubSet]*100.0, 1]],
	      
				{i, numberOfClasses}
			];
			chartLabels = Table[StringJoin["Class ", ToString[i]], {i, numberOfClasses}];
			If[Length[minMaxIndex] > 0,
				correctPredictionsInPercentList = Take[correctPredictionsInPercentList, minMaxIndex];
				chartLabels = Take[chartLabels, minMaxIndex]
			];
			yLabel = "%";
			plotLabel = "Correct Classifications";
			labels = {chartLabels, yLabel, plotLabel};
			Print[
				PlotBarChart[
					correctPredictionsInPercentList,
					labels,
					GraphicsOptionImageSize -> imageSize
				]
			],

			(* -------------------------------------------------------------------------------- *)
			"WrongClassificationDistribution",

	    	inputs = CIP`Utility`GetInputsOfDataSet[classificationDataSet];
	    	outputs = CIP`Utility`GetOutputsOfDataSet[classificationDataSet];
	    	classNumbers = pureFunction[inputs];
			numberOfClasses = Length[classificationDataSet[[1, 2]]];
		    wrongPredictionList = Table[0, {i, numberOfClasses}];
		    wrongPredictions = 0;
		    Do[
		    	classNumber = classNumbers[[i]];
				classNumberDesired = CIP`Utility`GetPositionOfMaximumValue[outputs[[i]]];
				If[classNumber != classNumberDesired,
					(* Wrong classification *)
					wrongPredictionList[[classNumber]] = wrongPredictionList[[classNumber]] + 1;
					wrongPredictions++
				],
	      
				{i, Length[inputs]}
			];
			wrongPredictionsInPercentList = {};
		    Do[
		    	If[wrongPredictions > 0,
		    		(* wrongPredictions > 0 *)
					AppendTo[wrongPredictionsInPercentList, CIP`Utility`RoundTo[wrongPredictionList[[i]]/wrongPredictions*100.0, 1]],
					(* wrongPredictions = 0 *)
					AppendTo[wrongPredictionsInPercentList, 0.0]
		    	],
	      
				{i, numberOfClasses}
			];
			chartLabels = Table[StringJoin["Class ", ToString[i]], {i, numberOfClasses}];
			If[Length[minMaxIndex] > 0,
				wrongPredictionsInPercentList = Take[wrongPredictionsInPercentList, minMaxIndex];
				chartLabels = Take[chartLabels, minMaxIndex]
			];
			yLabel = "%";
			plotLabel = "Wrong Classification Distribution";
			labels = {chartLabels, yLabel, plotLabel};
			Print[
				PlotBarChart[
					wrongPredictionsInPercentList,
					labels,
					GraphicsOptionImageSize -> imageSize
				]
			],
			
			(* -------------------------------------------------------------------------------- *)
			"WrongClassificationPairs",

	    	inputs = CIP`Utility`GetInputsOfDataSet[classificationDataSet];
	    	outputs = CIP`Utility`GetOutputsOfDataSet[classificationDataSet];
	    	classNumbers = pureFunction[inputs];
		    wrongIOPairList = {};
		    Do[
		    	classNumber = classNumbers[[i]];
				classNumberDesired = CIP`Utility`GetPositionOfMaximumValue[outputs[[i]]];
				If[classNumber != classNumberDesired,
					(* Wrong classification *)
					AppendTo[wrongIOPairList, 
						StringJoin[
							"Wrong I/O pair index = ", 
							ToString[i], 
							"; input = ", 
							ToString[classificationDataSet[[i, 1]]], 
							"; class desired/machine = ", 
							ToString[classNumberDesired], 
							" / ", 
							ToString[classNumber]
						]
					]
				],
	      
				{i, Length[inputs]}
			];
			If[Length[wrongIOPairList] > 0,
				
				Do[
					Print[wrongIOPairList[[i]]], 
					
					{i, Length[wrongIOPairList]}
				],
				
				Print["All I/O pairs are correctly classified"]
			]
		]
	];

ShowSingleDataSetInfo[

	(* Shows single data set information according to named property.

	   Returns: Nothing *)


	(* Properties to be analyzed: 
	   Full list: 
	   {
	       "IoPairs",
	       "InputComponents",
	       "OutputComponents",
	       "ClassCount"
	    } *)
 	namedProperty_,
    
    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...} *)
    dataSet_
    
	] :=
  
	Module[
    
		{
			classIndexMinMaxList,
			i,
			min,
			max,
			sortResult
		},

		Switch[namedProperty,
			
			(* -------------------------------------------------------------------------------- *)
			"IoPairs",
			Print["Number of IO pairs = ", Length[dataSet]],
			
			(* -------------------------------------------------------------------------------- *)
			"InputComponents",
		    Print["Number of input components = ", Length[dataSet[[1, 1]]]],
		    
			(* -------------------------------------------------------------------------------- *)
		    "OutputComponents",
		    Print["Number of output components = ", Length[dataSet[[1, 2]]]],
		    
		    (* -------------------------------------------------------------------------------- *)
		    "ClassCount",
		    sortResult = CIP`DataTransformation`SortClassificationDataSet[dataSet];
		    classIndexMinMaxList = sortResult[[2]];
		    Do[
		    	min = classIndexMinMaxList[[i, 1]];
		    	max = classIndexMinMaxList[[i, 2]];
		    	Print["Class ", i, " with ", max - min + 1, " members"],
		    	
		    	{i, Length[classIndexMinMaxList]}
		    ]
		]
	];

ShowSingleInputsInfo[

	(* Shows single inputs information according to named property.

	   Returns: Nothing *)


	(* Properties to be analyzed: 
	   Full list: 
	   {
	       "InputVectors",
	       "InputComponents"
	    } *)
 	namedProperty_,
    
	(* inputs : {vector1, vector2, ...}
	   vector : {component1, ..., component<NumberOfComponentsOfVector>} *)
    inputs_/;MatrixQ[inputs, NumberQ]
    
	] :=
  
	Module[
    
		{},

		Switch[namedProperty,
			
			(* -------------------------------------------------------------------------------- *)
			"InputVectors",
			Print["Number of input vectors = ", Length[inputs]],
			
			(* -------------------------------------------------------------------------------- *)
			"InputComponents",
			Print["Number of input components = ", Length[inputs[[1]]]]
		]
	];

ShowSingleRegressionResult[
	
	(* Shows single regression result according to named property.

	   Returns : Nothing *)

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
		   "RelativeSortedResidualsPlot",
		   "AbsoluteResidualsDistribution",
		   "RelativeResidualsDistribution"
	    } *)
 	namedProperty_,
    
    (* dataSet: {IOPair1, IOPair2, ...}
	   IOPair: {input, output}
	   input: {inputComponent1, inputComponent2, ...}
	   output : {outputComponent1, outputComponent2, ...} *)
    dataSet_,

	(* Pure function of inputs
	   inputs = {input1, input2, ...} 
	   input: {inputComponent1, inputComponent2, ...} *)
	pureFunction_,
	
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
    	{
    		k,
    		numberOfOutputComponents,
    		pointSize,
    		pointColor,
    		regressionResult,
    		numberOfIntervals,
    		labels,
    		displayFunction,
    		imageSize,
    		argumentRange2D,
    		functionValueRange2D,
    		correlationCoefficient
    	},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    numberOfIntervals = GraphicsOptionNumberOfIntervals/.{opts}/.Options[GraphicsOptionsResidualsDistribution];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    argumentRange2D = GraphicsOptionArgumentRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];
	    functionValueRange2D = GraphicsOptionFunctionValueRange2D/.{opts}/.Options[GraphicsOptionsPlotRange2D];

    	numberOfOutputComponents = Length[dataSet[[1, 2]]];

		Switch[namedProperty,
			
			(* -------------------------------------------------------------------------------- *)
			"RMSE",
			regressionResult = GetSingleRegressionResult["RMSE", dataSet, pureFunction];
	    	Print["Root mean squared error (RMSE) = ", ScientificForm[regressionResult, 4]],

			(* -------------------------------------------------------------------------------- *)
	        "SingleOutputRMSE",
			regressionResult = GetSingleRegressionResult["SingleOutputRMSE", dataSet, pureFunction];
		    Do[
				Print[
					"Out ", k, " : Root mean squared error (RMSE) = ", 
					ScientificForm[regressionResult[[k]], 4]
				],
				
				{k, numberOfOutputComponents}
			],
			
			(* -------------------------------------------------------------------------------- *)
			"AbsoluteResidualsStatistics",
			regressionResult = GetSingleRegressionResult["AbsoluteResidualsStatistics", dataSet, pureFunction];
		    Do[
				Print[
					"Out ", k, " : Absolute residuals (Data - Model): Mean/Median/Maximum Value = ", 
					ScientificForm[regressionResult[[k, 1]], 3], 
					" / ", 
					ScientificForm[regressionResult[[k, 2]], 3], 
					" / ", 
					ScientificForm[regressionResult[[k, 3]], 3]
				],
				
				{k, numberOfOutputComponents}
			],
			
			(* -------------------------------------------------------------------------------- *)
			"RelativeResidualsStatistics",
			regressionResult = GetSingleRegressionResult["RelativeResidualsStatistics", dataSet, pureFunction];
		    Do[
				Print[
					"Out ", k, " : Relative residuals (100*(Data - Model)/Data): Mean/Median/Maximum Value in % = ", 
					ScientificForm[regressionResult[[k, 1]], 3], 
					" / ", 
					ScientificForm[regressionResult[[k, 2]], 3], 
					" / ", 
					ScientificForm[regressionResult[[k, 3]], 3]
				],
				
				{k, numberOfOutputComponents}
			],
			
			(* -------------------------------------------------------------------------------- *)
			"ModelVsDataPlot",
			regressionResult = GetSingleRegressionResult["ModelVsData", dataSet, pureFunction];
		    Do[
				Print[
					PlotPoints2DAboveDiagonal[
						regressionResult[[k]],
						{
							StringJoin["Model Out ", ToString[k]],
							StringJoin["Data Out ", ToString[k]],
							StringJoin["Out ", ToString[k], " - Model versus data"]
						},
						GraphicsOptionPointSize -> pointSize,
						GraphicsOptionPointColor -> pointColor,
						GraphicsOptionImageSize -> imageSize
					]
				],
				
				{k, numberOfOutputComponents}
			],
			
			(* -------------------------------------------------------------------------------- *)
			"CorrelationCoefficient",
			regressionResult = GetSingleRegressionResult["CorrelationCoefficient", dataSet, pureFunction];
		    Do[
		    	correlationCoefficient = regressionResult[[k]];
				Print[
					"Out ", k, " : Correlation coefficient = ", correlationCoefficient
				],
				
				{k, numberOfOutputComponents}
			],
			
			(* -------------------------------------------------------------------------------- *)
			"SortedModelVsDataPlot",
			regressionResult = GetSingleRegressionResult["SortedModelVsData", dataSet, pureFunction];
		    Do[
				Print[
					PlotTwoIndexedLines2D[
						regressionResult[[k, 1]], 
						regressionResult[[k, 2]], 
						Length[regressionResult[[k, 1]]], 
						{
							"Index of sorted Value", 
							"Value", 
							StringJoin["Out ", ToString[k], " - Data above sorted model"]
						}
					]
				],
				
				{k, numberOfOutputComponents}
			],
			
			(* -------------------------------------------------------------------------------- *)
			"AbsoluteSortedResidualsPlot",
			regressionResult = GetSingleRegressionResult["AbsoluteSortedResiduals", dataSet, pureFunction];
		    Do[
				Print[
					PlotResiduals[
						regressionResult[[k]], 
						{
							"Index of sorted value", 
							"Data - Model", 
							StringJoin["Out ", ToString[k], " - Absolute residuals"]
						},
						GraphicsOptionPointSize -> pointSize,
						GraphicsOptionPointColor -> pointColor,
						GraphicsOptionImageSize -> imageSize
					]
				],
				
				{k, numberOfOutputComponents}
			],
			
			(* -------------------------------------------------------------------------------- *)
			"RelativeSortedResidualsPlot",
			regressionResult = GetSingleRegressionResult["RelativeSortedResiduals", dataSet, pureFunction];
		    Do[
				Print[
					PlotResiduals[
						regressionResult[[k]], 
						{
							"Index of sorted value", 
							"(Data - Model)/Data * 100", 
							StringJoin["Out ", ToString[k], " - Relative residuals"]
						},
						GraphicsOptionPointSize -> pointSize,
						GraphicsOptionPointColor -> pointColor,
						GraphicsOptionImageSize -> imageSize
					]
				],
				
				{k, numberOfOutputComponents}
			],
			
			(* -------------------------------------------------------------------------------- *)
		   "AbsoluteResidualsDistribution",
			regressionResult = 
				GetSingleRegressionResult[
					"AbsoluteResidualsDistribution", 
					dataSet, 
					pureFunction, 
					GraphicsOptionNumberOfIntervals -> numberOfIntervals
				];
		    Do[
				labels = 
					{
						"Absolute residual value",
						"Percentage in Interval",
						StringJoin["Out ", ToString[k], " - Distribution with ", ToString[Length[regressionResult[[k]]]], " intervals"]
					};
				Print[
					PlotLine2DWithOptionalPoints[
						regressionResult[[k]], 
						regressionResult[[k]],
						labels,
						GraphicsOptionDisplayFunction -> displayFunction,
						GraphicsOptionPointSize -> pointSize,
						GraphicsOptionPointColor -> pointColor,
						GraphicsOptionImageSize -> imageSize,
						GraphicsOptionArgumentRange2D -> argumentRange2D,
						GraphicsOptionFunctionValueRange2D -> functionValueRange2D
					]
				],
				
				{k, numberOfOutputComponents}
			],
		   
			(* -------------------------------------------------------------------------------- *)
		   "RelativeResidualsDistribution",
			regressionResult = 
				GetSingleRegressionResult[
					"RelativeResidualsDistribution", 
					dataSet, 
					pureFunction, 
					GraphicsOptionNumberOfIntervals -> numberOfIntervals
				];
		    Do[
				labels = 
					{
						"Relative residual value in percent",
						"Percentage in Interval",
						StringJoin["Out ", ToString[k], " - Distribution with ", ToString[Length[regressionResult[[k]]]], " intervals"]
					};
				Print[
					PlotLine2DWithOptionalPoints[
						regressionResult[[k]], 
						regressionResult[[k]],
						labels,
						GraphicsOptionDisplayFunction -> displayFunction,
						GraphicsOptionPointSize -> pointSize,
						GraphicsOptionPointColor -> pointColor,
						GraphicsOptionImageSize -> imageSize,
						GraphicsOptionArgumentRange2D -> argumentRange2D,
						GraphicsOptionFunctionValueRange2D -> functionValueRange2D
					]
				],
				
				{k, numberOfOutputComponents}
			]
		]
	];

ShowTrainOptimization[

	(* Shows training set optimization result.

	   Returns: Nothing *)


	(* trainingSetOptimizationResult = {trainingSetRmseList, testSetRmseList, not interesting, not interesting}
	   trainingSetRmseList: List with {number of optimization step, RMSE of training set}
	   testSetRmseList: List with {number of optimization step, RMSE of test set} *)
	trainingSetOptimizationResult_,
    
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			trainingPoints2D,
			testPoints2D,
			labels,
			displayFunction,
			imageSize,
			trainingPoints2DWithPlotStyle,
			testPoints2DWithPlotStyle,
			points2DWithPlotStyleList
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    imageSize = GraphicsOptionImageSize/.{opts}/.Options[GraphicsOptionsImageSize];
	    displayFunction = GraphicsOptionDisplayFunction/.{opts}/.Options[GraphicsOptionsDisplayFunction];

		trainingPoints2D = trainingSetOptimizationResult[[1]];
		testPoints2D = trainingSetOptimizationResult[[2]];
		
		labels = {"Training set optimization step", "RMSE", "Training (green), Test (red)"};
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
		]
	];

(* ::Section:: *)
(* End of Package *)

End[]

EndPackage[]
