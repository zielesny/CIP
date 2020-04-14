(*
-----------------------------------------------------------------------
Computational Intelligence Packages (CIP): Package Custom
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

BeginPackage["CIP`Custom`", {"CIP`CurveFit`", "CIP`DataTransformation`"}]

(* ::Section:: *)
(* Off settings *)

Off[General::"spell1"]
Off[General::shdw]
Off[General::compat]
Off[FittedModel::precw]
Off[NonlinearModelFit::sszero]
Off[NMinimize::cvmit]
Off[General::munfl]

(* ::Section:: *)
(* Options *)

Options[CustomOptionsHeuristics01] = 
{
    (* Scale factor for y data *)
    CustomOptionYScaleFactor -> 1.0,

    (* Fraction of the xy data to detect y minimum from lower x to upper x *)
    CustomOptionYMinDetectionFraction -> 0.0,
        
    (* Errors for y values that corresponds to minimum and maximum x value (see code) *)
    CustomOptionYMinMaxErrors -> {1.0, 1.0},

	(* Residuals type: "AbsoluteResiduals", "RelativeResiduals" *)
 	CustomOptionResidualsType -> "RelativeResiduals",
    
    (* Number of removal steps *)
    CustomOptionNumberOfRemovalSteps -> 1,

    (* parameterIntervals: {parameterInterval1, parameterInterval2, ...}
       parameterInterval: {minimum value, maximum value} *)
	CustomOptionParameterIntervals -> {}
    
}

(* ::Section:: *)
(* Declarations *)
FitModelFunctionWithHeuristics01::usage = 
	"FitModelFunctionWithHeuristics01[xyRawData, modelFunction, argumentOfModelFunction, parametersOfModelFunction, options]"	

(* ::Section:: *)
(* Functions *)

Begin["`Private`"]
	
FitModelFunctionWithHeuristics01[

	(* Fits linear or nonlinear model function to {x, y} data with heuristic approach (denoted 01, for details see code).

	   Returns:
	   {curveFitInfo, xyErrorDataWithoutOutliers, xyErrorDataAll} 
	   NOTE: 
	   curveFitInfo corresponds to xyErrorDataWithoutOutliers
	   xyErrorDataWithoutOutliers has weighted errors 
	   xyErrorDataAll has error 1.0 *)

    
    (* {x, y} raw data: {{x1, y1}, {x2, y2}, ...} *)
    xyRawData_/;MatrixQ[xyRawData, NumberQ],
    
    (* Model function to be fitted *)
    modelFunction_,
    
    (* Argument of fit model function *)
    argumentOfModelFunction_,
    
    (* Parameters of fit model function : {parameter1, parameter2, ...} *)
    parametersOfModelFunction_,
        
	(* Options *)
	opts___
    
	] :=
  
	Module[
    
		{
			confidenceLevelOfParameterErrors,
			currentMaxIterations,
			currentWorkingPrecision,
			i,
			method,
			startParameters,
			varianceEstimator,
			minIndex,
			yMin,
			xyErrorDataAll,
			xyErrorData,
			sortedXyData,
			fitResult,
			yScaleFactor,
			yMinDetectionFraction,
			yMinMaxErrors,
			residualsType,
			numberOfRemovalSteps,
			parameterIntervals,
			minimizationPrecision,
			maximumNumberOfIterations,
			searchType,
			numberOfTrialPoints,
			randomValueInitialization
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
		(* Options for CIP`CurveFit`FitModelFunctionWithOutlierRemoval *)
	    confidenceLevelOfParameterErrors = CurveFitOptionConfidenceLevel/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		currentWorkingPrecision = CurveFitOptionCurrentWorkingPrecision/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		currentMaxIterations = CurveFitOptionCurrentMaxIterations/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		startParameters = CurveFitOptionStartParameters/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		method = CurveFitOptionMethod/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		varianceEstimator = CurveFitOptionVarianceEstimator/.{opts}/.Options[CurveFitOptionsFitModelFunction];

		(* Options for CIP`CurveFit`GetStartParameters *)
	    minimizationPrecision = CurveFitOptionMinimizationPrecision/.{opts}/.Options[CurveFitOptionsStartParameters];
	    maximumNumberOfIterations = CurveFitOptionMaximumIterations/.{opts}/.Options[CurveFitOptionsStartParameters];
	    searchType = CurveFitOptionSearchType/.{opts}/.Options[CurveFitOptionsStartParameters];
	    numberOfTrialPoints = CurveFitOptionNumberOfTrialPoints/.{opts}/.Options[CurveFitOptionsStartParameters];
        randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
		
		yScaleFactor = CustomOptionYScaleFactor/.{opts}/.Options[CustomOptionsHeuristics01];
		yMinDetectionFraction = CustomOptionYMinDetectionFraction/.{opts}/.Options[CustomOptionsHeuristics01];
		yMinMaxErrors = CustomOptionYMinMaxErrors/.{opts}/.Options[CustomOptionsHeuristics01];
		residualsType = CustomOptionResidualsType/.{opts}/.Options[CustomOptionsHeuristics01];
		numberOfRemovalSteps = CustomOptionNumberOfRemovalSteps/.{opts}/.Options[CustomOptionsHeuristics01];
		parameterIntervals = CustomOptionParameterIntervals/.{opts}/.Options[CustomOptionsHeuristics01];

		(* Sort data first *)
		sortedXyData = Sort[xyRawData];
		
		(* Scale data and detect possible minimum *)
		minIndex = 1;
		yMin = sortedXyData[[minIndex, 2]] * yScaleFactor;
		Do[
  			sortedXyData[[i, 2]] = sortedXyData[[i, 2]] * yScaleFactor;
  			If[sortedXyData[[i, 2]] < yMin,
  				yMin = sortedXyData[[i, 2]];
  				minIndex = i
  			],
  			
  			{i, 1, Length[sortedXyData]}
  		];

		(* Transform to valid xyError data *)
		xyErrorDataAll = CIP`DataTransformation`AddErrorToXYData[sortedXyData, 1.0];

		(* Exclude initial data points before "reasonable" minimum *)
		If[minIndex > Length[xyErrorDataAll] * yMinDetectionFraction,
  			
  			xyErrorData = xyErrorDataAll,
  
  			xyErrorData = Take[xyErrorDataAll, {minIndex, Length[xyErrorDataAll]}]
  		];

		(* Weigh data points with errors *)
		Do[
  			xyErrorData[[i, 3]] = CIP`DataTransformation`TransformLinear[i, 1, Length[xyErrorData], yMinMaxErrors[[1]], yMinMaxErrors[[2]]],
  			
  			{i, Length[xyErrorData]}
		];

		If[Length[parameterIntervals] > 0,
			startParameters =
				CIP`CurveFit`GetStartParameters[
					xyErrorData, 
					modelFunction, 
					argumentOfModelFunction, 
					parametersOfModelFunction, 
					parameterIntervals, 
				    CurveFitOptionMinimizationPrecision -> minimizationPrecision,
				    CurveFitOptionMaximumIterations -> maximumNumberOfIterations,
				    CurveFitOptionSearchType -> searchType,
				    CurveFitOptionNumberOfTrialPoints -> numberOfTrialPoints,
			        UtilityOptionRandomInitializationMode -> randomValueInitialization
				];
		];

		fitResult = 
  			CIP`CurveFit`FitModelFunctionWithOutlierRemoval[
   				xyErrorData,
   				modelFunction,
   				argumentOfModelFunction,
   				parametersOfModelFunction,
   				residualsType,
   				numberOfRemovalSteps,
				CurveFitOptionConfidenceLevel -> confidenceLevelOfParameterErrors,
				CurveFitOptionCurrentWorkingPrecision -> currentWorkingPrecision,
				CurveFitOptionCurrentMaxIterations -> currentMaxIterations,
				CurveFitOptionStartParameters -> startParameters,
				CurveFitOptionMethod -> method,
				CurveFitOptionVarianceEstimator -> varianceEstimator
   			];

		Return[{fitResult[[1]], fitResult[[2]], xyErrorDataAll}]
	];

(* ::Section:: *)
(* End of Package *)

End[]

EndPackage[]
