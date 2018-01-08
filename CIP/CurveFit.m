(*
-----------------------------------------------------------------------
Computational Intelligence Packages (CIP): Package CurveFit
Version 3.0 for Mathematica 11 or higher
-----------------------------------------------------------------------

Author: Achim Zielesny

GNWI - Gesellschaft fuer naturwissenschaftliche Informatik mbH, 
Oer-Erkenschwick, Germany

Citation:
Achim Zielesny, Computational Intelligence Packages (CIP), Version 3.0, 
GNWI mbH (http://www.gnwi.de), Oer-Erkenschwick, Germany, 2018.

Code partially based on:
G.W. Mueller, Plotprogramme in Basic, Muenchen 1983, pages 68f and 
program code on page 75f.

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
curveFitInfo: {fitType, fitInfo}

	fitType = "ModelFunction"
	-------------------------
	fitInfo: 
		{
			fitted model function, 
			argument of model function, 
			fitted parameters,
			parameter errors,
			parameter confidence intervals
		} 

	fitType = "SmoothingCubicSplines"
	---------------------------------
	fitInfo: 
		{
			xValues, 
			{Coefficients1, Coefficients2, ..., Coefficients<numberOfXValues>}
		}
	xValues = {x1, x2, ..., x<numberOfXValues>} from xyErrorData
	Coefficients[[i]] = {aCoefficients[[i]], bCoefficients[[i]], cCoefficients[[i]], dCoefficients[[i]]}
	Spline data are used for splines function evaluation:
	function value = argument^3*aCoefficients + argument^2*bCoefficients + argument*cCoefficients + dCoefficients
-----------------------------------------------------------------------
*)

(* ::Section:: *)
(* Package and dependencies *)

BeginPackage["CIP`CurveFit`", {"CIP`Utility`", "CIP`Graphics`", "CIP`DataTransformation`", "CIP`CalculatedData`", "Combinatorica`"}]

(* ::Section:: *)
(* Off settings *)

Off[General::"spell1"]
Off[General::shdw]
Off[General::compat]
Off[FittedModel::precw]
Off[NonlinearModelFit::sszero]
Off[NMinimize::cvmit]

(* ::Section:: *)
(* Options *)

Options[CurveFitOptionsStartParameters] = 
{
	(* Precision of minimization result, see Mathematica documentation *)
    CurveFitOptionMinimizationPrecision -> 3,
    
	(* Maximum number of iterations *)
    CurveFitOptionMaximumIterations -> 20,
    
	(* "Random" or "NMinimize" *)
    CurveFitOptionSearchType -> "NMinimize",
    
	(* Number of trial points for CurveFitOptionSearchType = "Random" *)
    CurveFitOptionNumberOfTrialPoints -> 1000
}

Options[CurveFitOptionsFitModelFunction] = 
{
    (* Statistical confidence level of parameter errors *)
    CurveFitOptionConfidenceLevel -> 0.6827,
    
    (* Working precision for calculations *)
    CurveFitOptionCurrentWorkingPrecision -> MachinePrecision,
    
    (* Maximum number of iterations *)
    CurveFitOptionCurrentMaxIterations -> 100000,
    
    (* Start values for parameters of model function of form: 
       {{parameter1, startValue1}, {parameter2, startValue2}, ...} *)
    CurveFitOptionStartParameters -> {},
    
    (* Minimization method *)
    CurveFitOptionMethod -> Automatic,
    
    (* Variance Estimation: "ErrorData" or "ReducedChiSquare" *)
    CurveFitOptionVarianceEstimator -> "ErrorData"
};

Options[CurveFitOptionsFitResult] = 
{
    (* Labels for model function plot below data:
       {xLabel, yLabel, diagramLabel} *)
    CurveFitOptionLabels -> {"x", "y", "Data above model function"}
};

(* ::Section:: *)
(* Declarations *)

CalculateFunctionValue::usage = 
	"CalculateFunctionValue[argumentValue, curveFitInfo]"

CalculateDerivativeValue::usage = 
	"CalculateDerivativeValue[orderOfDerivative, argumentValue, curveFitInfo]"

CalculateOutputs::usage = 
	"CalculateOutputs[inputs, curveFitInfo]"
	
FitCubicSplines::usage = 
	"FitCubicSplines[xyErrorData, reducedChisquare]"
	
FitModelFunction::usage = 
	"FitModelFunction[xyErrorData, modelFunction, argumentOfModelFunction, parametersOfModelFunction, options]"	

GetChiSquare::usage = 
	"GetChiSquare[xyErrorData, modelFunction, argumentOfModelFunction, parametersOfModelFunction, parameterValues]"	

Get3dChiSquareBelowThreshold::usage = 
	"Get3dChiSquareBelowThreshold[xyErrorData, modelFunction, argumentOfModelFunction, parametersOfModelFunction, chiSquareAtMinimum, parameterValuesAtMinimum, threshold, indexOfParameter1, indexOfParameter2, valueOfParameter1, valueOfParameter2]"

GetNumberOfData::usage = 
	"GetNumberOfData[desiredWidthOfConfidenceRegion, indexOfParameter, pureOriginalFunction, argumentRange, standardDeviationRange, modelFunction, argumentOfModelFunction, parametersOfModelFunction, options]"

GetParameterConfidenceRegion::usage = 
	"GetParameterConfidenceRegion[indexOfParameter, curveFitInfo]"

GetParameterValuesAtMinimum::usage = 
	"GetParameterValuesAtMinimum[parametersOfModelFunction, curveFitInfo]"

GetSingleFitResult::usage =
	"GetSingleFitResult[namedProperty, xyErrorData, curveFitInfo]"

GetStartParameters::usage = 
	"GetStartParameters[xyErrorData, modelFunction, argumentOfModelFunction, parametersOfModelFunction, parameterIntervals, options]"	

ShowBestExponent::usage = 
	"ShowBestExponent[xyErrorData, minExponent, maxExponent, exponentStepSize, options]"	

ShowFitResult::usage =
	"ShowFitResult[namedPropertyList, xyErrorData, curveFitInfo, options]"

(* ::Section:: *)
(* Functions *)

Begin["`Private`"]

CalculateFunctionValue[

	(* Returns function value for curveFitInfo.

	   Returns: 
	   Function value *)


  	(* Argument of function *)
	argumentValue_?NumberQ,

  	(* See "Frequently used data structures" *)
	curveFitInfo_
   
	] :=
  
	Module[
    
		{
			argumentOfModelFunction,
			fitType,
			fitInfo,
			modelFunction
		},

		fitType = curveFitInfo[[1]];
		fitInfo = curveFitInfo[[2]];
		Switch[fitType,
			
			"ModelFunction",
			modelFunction = fitInfo[[1]];
			argumentOfModelFunction = fitInfo[[2]];
			Return[N[modelFunction/.argumentOfModelFunction -> argumentValue]],	
			
			"SmoothingCubicSplines",
	     	Return[GetSmoothingCubicSplinesFunctionValue[argumentValue, fitInfo]]   
		]
	];

CalculateDerivativeValue[

	(* Returns derivative value for curveFitInfo.

	   Returns: 
	   Function value *)

	(* Order of derivative *)
	orderOfDerivative_?IntegerQ,

  	(* Argument of function *)
	argumentValue_?NumberQ,

  	(* See "Frequently used data structures" *)
	curveFitInfo_
   
	] :=
  
	Module[
    
		{
			argumentOfModelFunction,
			derivativeOfOrder,
			fitType,
			fitInfo,
			modelFunction
		},

		fitType = curveFitInfo[[1]];
		fitInfo = curveFitInfo[[2]];
		Switch[fitType,
			
			"ModelFunction",
			modelFunction = fitInfo[[1]];
			argumentOfModelFunction = fitInfo[[2]];
			derivativeOfOrder = D[modelFunction, {argumentOfModelFunction, orderOfDerivative}];
			Return[N[derivativeOfOrder/.argumentOfModelFunction -> argumentValue]],	
			
			"SmoothingCubicSplines",
	        Switch[orderOfDerivative, 
	        	
	        	1,
	     		Return[GetSmoothingCubicSplinesFirstDerivativeValue[argumentValue, fitInfo]],  
	        	
	        	2,
	     		Return[GetSmoothingCubicSplinesSecondDerivativeValue[argumentValue, fitInfo]],  
	        	
	        	3,
	     		Return[GetSmoothingCubicSplinesThirdDerivativeValue[argumentValue, fitInfo]]  
	        ];
	        Return[0.0]
		]
	];

CalculateOutputs[

	(* Returns function values for curveFitInfo.

	   Returns: 
	   Outputs: {{functionValue1}, {functionValue2}, ..., {functionValue<Length[argumentValues]>}} *)


  	(* {{argument1}, {argument2}, ...} *)
	inputs_/;MatrixQ[inputs, NumberQ],

  	(* See "Frequently used data structures" *)
	curveFitInfo_
   
	] :=
  
	Module[
    
		{
			argumentOfModelFunction,
			fitType,
			fitInfo,
			i,
			modelFunction
		},

		fitType = curveFitInfo[[1]];
		fitInfo = curveFitInfo[[2]];
		Switch[fitType,
			
			"ModelFunction",
			modelFunction = fitInfo[[1]];
			argumentOfModelFunction = fitInfo[[2]];
			Return[
				Table[
					{N[modelFunction/.argumentOfModelFunction -> inputs[[i, 1]]]},
					
					{i, Length[inputs]}
				]
			],	
			
			"SmoothingCubicSplines",
	     	Return[
	     		Table[
		     		{GetSmoothingCubicSplinesFunctionValue[inputs[[i, 1]], fitInfo]},
		     		
					{i, Length[inputs]}
	     		]
	     	]   
		]
	];

FitCubicSplines[

    (* Fits data by smoothing cubic splines. Code is based on: G.W. Mueller, Plotprogramme in Basic, Muenchen 1983, pages 68f and program code on page 75f.
    
       Returns:
	   curveFitInfo *)

	
    (* {x, y, error} data: {{x1, y1, error1}, {x2, y2, error2}, ...} *)
    xyErrorData_/;MatrixQ[xyErrorData, NumberQ],
	
    (* Reduced chisquare: Optimum to be 1 *)
	reducedChiSquare_?NumberQ
   
	] :=
  
	Module[
    
		{
			xValues,
			yValues,
			errorValues,
			epsilon,
			aCoefficients,
			bCoefficients,
			cCoefficients,
			dCoefficients,
			rArray,
			sArray,
			eArray,
			uArray,
			tArray,
			gArray,
			vArray,
			n1,
			n2,
			m1,
			m2,
			p,
			h,
			f,
			g,
			h1,
			e,
			f2,
			j,
			i,
			numberOfValues,
			sortedData,
			smoothingCubicSplinesInfo,
			chiSquare
		},

		(* Set precision for termination *)
		epsilon = 10^(-(Round[MachinePrecision] - 4));
		
		(* Sort data and eliminate possible dublettes *)
		sortedData = Sort[xyErrorData];
		xValues = {};
		yValues = {};
		errorValues = {};
		Do[
			If[sortedData[[i, 1]] != sortedData[[i + 1, 1]],
				xValues = {xValues, sortedData[[i, 1]]};
				yValues = {yValues, sortedData[[i, 2]]};
				errorValues = {errorValues, sortedData[[i, 3]]};
			],
			
			{i, Length[sortedData] - 1}
		];
		xValues = Flatten[{xValues, sortedData[[Length[sortedData], 1]]}];
		yValues = Flatten[{yValues, sortedData[[Length[sortedData], 2]]}];
		errorValues = Flatten[{errorValues, sortedData[[Length[sortedData], 3]]}];
		numberOfValues = Length[xValues];
		chiSquare = reducedChiSquare*numberOfValues;

		(* Initialize arrays *)
		aCoefficients = Table[0.0,{numberOfValues + 2}];
		bCoefficients = Table[0.0,{numberOfValues + 2}];
		cCoefficients = Table[0.0,{numberOfValues + 2}];
		dCoefficients = Table[0.0,{numberOfValues + 2}];
		rArray = Table[0.0,{numberOfValues + 2}];
		sArray = Table[0.0,{numberOfValues + 2}];
		eArray = Table[0.0,{numberOfValues + 2}];
		uArray = Table[0.0,{numberOfValues + 2}];
		tArray = Table[0.0,{numberOfValues + 2}];
		gArray = Table[0.0,{numberOfValues + 2}];
		vArray = Table[0.0,{numberOfValues + 2}];

		(* Determine coefficients *)
		n1 = 2;
		n2 = numberOfValues + 1;
		m1 = n1 + 1;
		m2 = n2 - 1;
		p = 0.0;
		h = xValues[[m1 - 1]] - xValues[[n1 - 1]];
		f = (yValues[[m1 - 1]] - yValues[[n1 - 1]])/h;

		Do[
			g = h;
			h1 = 0.0;
			h = xValues[[i]] - xValues[[i - 1]];
			e = f;
			f = (yValues[[i]] - yValues[[i - 1]])/h;
			dCoefficients[[i - 1]] = f - e;
			tArray[[i]] = (2.0/3.0)*(g + h);
			gArray[[i]] = h/3.0;
			eArray[[i]] = errorValues[[i - 2]]/g;
			rArray[[i]] = errorValues[[i]]/h;
			sArray[[i]] = -errorValues[[i - 1]]/g - errorValues[[i - 1]]/h,
           
			{i, m1, m2}
		];

		Do[
			cCoefficients[[i - 1]] = rArray[[i]]*rArray[[i]] + sArray[[i]]*sArray[[i]] + eArray[[i]]*eArray[[i]];
			bCoefficients[[i - 1]] = rArray[[i]]*sArray[[i + 1]] + sArray[[i]]*eArray[[i + 1]];
			aCoefficients[[i - 1]] = rArray[[i]]*eArray[[i + 2]],
           
			{i, m1, m2}
		];
		f2 = -chiSquare;

		While[True,

			Do[
				sArray[[i - 1]] = f*rArray[[i - 1]];
				eArray[[i - 2]] = g*rArray[[i - 2]];
                rArray[[i]] = 1.0/(p*cCoefficients[[i - 1]] + tArray[[i]] - f*sArray[[i - 1]] - g*eArray[[i - 2]]);
                uArray[[i]] = dCoefficients[[i - 1]] - sArray[[i - 1]]*uArray[[i - 1]] - eArray[[i - 2]]*uArray[[i - 2]];
                f = p*bCoefficients[[i - 1]] + gArray[[i]] - h*sArray[[i - 1]];
                g = h;
                h = aCoefficients[[i - 1]]*p,
                
                {i, m1, m2}
			];

			Do[
                uArray[[i]] = rArray[[i]]*uArray[[i]] - sArray[[i]]*uArray[[i + 1]] - eArray[[i]]*uArray[[i + 2]],
                
                {i, m2, m1 - 1, -1}
			];

			e = 0.0;
			h = 0.0;

			Do[
				g = h;
                h = (uArray[[i + 1]] - uArray[[i]])/(xValues[[i]] - xValues[[i - 1]]);
                vArray[[i]] = (h - g)*errorValues[[i - 1]]*errorValues[[i - 1]];
                e = e + vArray[[i]]*(h - g),
                
                {i, n1, m2}
			];

			g = -h*errorValues[[n2 - 1]]*errorValues[[n2 - 1]];
			vArray[[n2]] = -h*errorValues[[n2 - 1]]*errorValues[[n2 - 1]];
			e = e - g*h;
			g = f2;
			f2 = e*p*p;

			If[f2 > chiSquare, Break[]];
			If[f2 < g, Break[]];

			f = 0.0;
			h = (vArray[[m1]] - vArray[[n1]])/(xValues[[m1 - 1]] - xValues[[n1 - 1]]);

			Do[
				g = h;
                h = (vArray[[i + 1]] - vArray[[i]])/(xValues[[i]] - xValues[[i - 1]]);
                g = h - g - sArray[[i - 1]]*rArray[[i - 1]] - eArray[[i - 2]]*rArray[[i - 2]];
                f = f + g*rArray[[i]]*g;
                rArray[[i]] = g,
                
                {i, m1, m2}
			];

			h = e - p*f;
			(* If[h == h1, Break[]]; is replaced by *)
			If[Abs[h - h1] < Abs[h]*epsilon, Break[]];

			h1 = h;
			If[h <= 0.0, Break[]];

			j = Sqrt[chiSquare/e];
			p = p + (chiSquare - f2)/((j + p)*h);
		];

		Do[
			dCoefficients[[i - 1]] = yValues[[i - 1]] - p*vArray[[i]];
			bCoefficients[[i - 1]] = uArray[[i]],
           
			{i, n1, n2}
		];

		Do[
			h = xValues[[i]] - xValues[[i - 1]];
			aCoefficients[[i - 1]] = (bCoefficients[[i]] - bCoefficients[[i - 1]])/(3.0*h);
			cCoefficients[[i - 1]] = (dCoefficients[[i]] - dCoefficients[[i - 1]])/h - (h*aCoefficients[[i - 1]] + bCoefficients[[i - 1]])*h,
           
			{i, n1, m2}
		];

		smoothingCubicSplinesInfo = 
			{
				xValues, 
				Table[{aCoefficients[[i]], bCoefficients[[i]], cCoefficients[[i]], dCoefficients[[i]]},{i, Length[xValues]}]
			};
		Return[{"SmoothingCubicSplines", smoothingCubicSplinesInfo}]
	];
	
FitModelFunction[

	(* Fits linear or nonlinear model function to {x, y, error} data.

	   Returns:
	   curveFitInfo *)

    
    (* {x, y, error} data: {{x1, y1, error1}, {x2, y2, error2}, ...} *)
    xyErrorData_/;MatrixQ[xyErrorData, NumberQ],
    
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
			fittedModel,
			i,
			method,
			startParameters,
			varianceEstimatorFunction,
			varianceEstimator,
			xyData,
			yWeights
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    confidenceLevelOfParameterErrors = CurveFitOptionConfidenceLevel/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		currentWorkingPrecision = CurveFitOptionCurrentWorkingPrecision/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		currentMaxIterations = CurveFitOptionCurrentMaxIterations/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		startParameters = CurveFitOptionStartParameters/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		method = CurveFitOptionMethod/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		varianceEstimator = CurveFitOptionVarianceEstimator/.{opts}/.Options[CurveFitOptionsFitModelFunction];

		Switch[varianceEstimator,
			
			"ErrorData",
			varianceEstimatorFunction = (1 &),
			
			"ReducedChiSquare",
			varianceEstimatorFunction = Automatic
		];
    
    	If[Length[startParameters] == 0,
    		startParameters = 
	    		Table[
	    			{parametersOfModelFunction[[i]], RandomReal[1.0]},
	    			
	    			{i, Length[parametersOfModelFunction]}
	    		];
    	];
    	
	    xyData = xyErrorData[[All, {1, 2}]];
	    yWeights = 1./(xyErrorData[[All, 3]])^2 ;
	    
	    (* With "VarianceEstimatorFunction -> (1 &)" the parameter errors are computed ONLY from the weights, see Mathematica documentation *)
		fittedModel = 
			NonlinearModelFit[
				xyData,
				modelFunction,
				startParameters,
				argumentOfModelFunction,
				Weights -> yWeights,
				ConfidenceLevel -> confidenceLevelOfParameterErrors,
				MaxIterations -> currentMaxIterations,
				WorkingPrecision -> currentWorkingPrecision,
				Method -> method,
				VarianceEstimatorFunction -> varianceEstimatorFunction
			];

		Return[
			{
				"ModelFunction",
				{
					fittedModel["BestFit"], 
					argumentOfModelFunction, 
					fittedModel["BestFitParameters"],
					fittedModel["ParameterErrors"], 
					fittedModel["ParameterConfidenceIntervals"]
				}
			}
		]
	];

GetChiSquare[

	(* Returns chi-square.

	   Returns: 
	   Chi-square *)
	
    (* {x, y, error} data: {{x1, y1, error1}, {x2, y2, error2}, ...} *)
    xyErrorData_/;MatrixQ[xyErrorData, NumberQ],
	
    (* Model function to be fitted *)
    modelFunction_,
    
    (* Argument of fit model function *)
    argumentOfModelFunction_,
    
    (* Parameters of fit model function : {parameter1, parameter2, ...} *)
    parametersOfModelFunction_,

    (* Values of parameters: {value of parameter1, value of parameter2, ...} *)
	parameterValues_/;VectorQ[parameterValues, NumberQ]
	
	] :=
    
	Module[
    
    	{
    		functionValue,
    		i,
    		rules,
    		substitutedModelFunction
    	},

		rules = 
			Table[
				parametersOfModelFunction[[i]] -> parameterValues[[i]], 
					
				{i, Length[parametersOfModelFunction]}
			];
		substitutedModelFunction = modelFunction/.rules;
		Return[
			Apply[Plus,
				Table[
					functionValue = substitutedModelFunction/.argumentOfModelFunction -> xyErrorData[[i, 1]];
					((xyErrorData[[i, 2]] - functionValue)/xyErrorData[[i, 3]])^2,
				
					{i, Length[xyErrorData]}
				]
			]
		]
	];

Get3dChiSquareBelowThreshold[

	(* Returns 3D chi-square surface below threshold.

	   Returns: 
	   3D Chi-square surface below threshold*)
	
    (* {x, y, error} data: {{x1, y1, error1}, {x2, y2, error2}, ...} *)
    xyErrorData_/;MatrixQ[xyErrorData, NumberQ],
	
    (* Model function to be fitted *)
    modelFunction_,
    
    (* Argument of fit model function *)
    argumentOfModelFunction_,
    
    (* Parameters of fit model function : {parameter1, parameter2, ...} *)
    parametersOfModelFunction_,

    (* Chi-Square at minimum *)
	chiSquareAtMinimum_?NumberQ, 
	
    (* Values of parameters at minimum: {value of parameter1, value of parameter2, ...} *)
	parameterValuesAtMinimum_/;VectorQ[parameterValuesAtMinimum, NumberQ],
	
	(* Threshold *)
	threshold_?NumberQ,
	
	(* Index of parameter 1 *)
	indexOfParameter1_?IntegerQ,
	
	(* Index of parameter 2 *)
	indexOfParameter2_?IntegerQ,

	(* Value of parameter 1 *)
	valueOfParameter1_?NumberQ,
	
	(* Value of parameter 2 *)
	valueOfParameter2_?NumberQ
	
	] :=
    
	Module[
    
    	{
    		currentParameterValues,
    		chiSquare
    	},

		currentParameterValues = ReplacePart[parameterValuesAtMinimum, {indexOfParameter1 -> valueOfParameter1, indexOfParameter2 -> valueOfParameter2}];
		chiSquare = GetChiSquare[xyErrorData, modelFunction, argumentOfModelFunction, parametersOfModelFunction, currentParameterValues];
		
		If[chiSquare - chiSquareAtMinimum <= threshold,
			
			Return[chiSquare - chiSquareAtMinimum],
			
			Return[threshold]
		]
	];

GetChiSquareInternal[

	(* Returns chi-square.

	   Returns: 
	   Chi-square *)
	
    (* {x, y, error} data: {{x1, y1, error1}, {x2, y2, error2}, ...} *)
    xyErrorData_/;MatrixQ[xyErrorData, NumberQ],
	
	pureModelFunction_
	
	] :=
    
	Module[
    
    	{
    		i
    	},

		Return[
			Apply[Plus,
				Table[
					((xyErrorData[[i, 2]] - pureModelFunction[xyErrorData[[i, 1]]])/xyErrorData[[i, 3]])^2,
				
					{i, Length[xyErrorData]}
				]
			]
		]
	];

GetNumberOfData[

	(* Returns number of necessary data to achieve confidence interval constraints of specified parameter.

	   Returns: 
	   Number of necessary data *)

	(* Desired width of confidence region *)
	desiredWidthOfConfidenceRegion_?NumberQ,
	
	(* Index of parameter for confidence region to check *)
	indexOfParameter_?IntegerQ,

	(* Pure original function *)      
    pureOriginalFunction_,
	
    (* {argumentStart, argumentEnd} *)
    argumentRange_/;VectorQ[argumentRange, NumberQ],

    (* {standardDeviationMin, standardDeviationMax} *)
    standardDeviationRange_/;VectorQ[standardDeviationRange, NumberQ],
	
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
			argumentDistance,
			dataStructure,
			errorType,
			
			confidenceLevelOfParameterErrors,
			currentMaxIterations,
			currentWorkingPrecision,
			method,
			maximumNumberOfData,
			minimumNumberOfData,
			maximumWidthOfConfidenceRegion,
			minimumWidthOfConfidenceRegion,
			numberOfParameters,
			randomValueInitialization,
			startParameters,
			testNumberOfData,
			testWidthOfConfidenceRegion,
			varianceEstimator
    	},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    confidenceLevelOfParameterErrors = CurveFitOptionConfidenceLevel/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		currentWorkingPrecision = CurveFitOptionCurrentWorkingPrecision/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		currentMaxIterations = CurveFitOptionCurrentMaxIterations/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		startParameters = CurveFitOptionStartParameters/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		method = CurveFitOptionMethod/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		varianceEstimator = CurveFitOptionVarianceEstimator/.{opts}/.Options[CurveFitOptionsFitModelFunction];
        randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
        
        argumentDistance = CalculatedDataOptionDistance/.{opts}/.Options[CalculatedDataOptionsDataGeneration];
        dataStructure = CalculatedDataOptionDataStructure/.{opts}/.Options[CalculatedDataOptionsDataGeneration];
        errorType = CalculatedDataOptionErrorType/.{opts}/.Options[CalculatedDataOptionsDataGeneration];

		numberOfParameters = Length[parametersOfModelFunction];
		minimumNumberOfData = numberOfParameters + 1;
		maximumNumberOfData = minimumNumberOfData*2;

		minimumWidthOfConfidenceRegion = 
			GetWidthOfConfidenceRegion[
				minimumNumberOfData, 
				indexOfParameter, 
				pureOriginalFunction, 
				argumentRange, 
				standardDeviationRange, 
				modelFunction, 
				argumentOfModelFunction, 
				parametersOfModelFunction,
				UtilityOptionRandomInitializationMode -> randomValueInitialization,
				CalculatedDataOptionDistance -> argumentDistance,
				CalculatedDataOptionDataStructure -> dataStructure,
				CalculatedDataOptionErrorType -> errorType,
				CurveFitOptionConfidenceLevel -> confidenceLevelOfParameterErrors,
				CurveFitOptionCurrentWorkingPrecision -> currentWorkingPrecision,
				CurveFitOptionCurrentMaxIterations -> currentMaxIterations,
				CurveFitOptionStartParameters -> startParameters,
				CurveFitOptionMethod -> method,
				CurveFitOptionVarianceEstimator -> varianceEstimator
			];
		
		(*
		Print["MinNumber = ", minimumNumberOfData];
		Print["MinWidth  = ", minimumWidthOfConfidenceRegion];
		Print[""];
		*)
		
		If[minimumWidthOfConfidenceRegion <= desiredWidthOfConfidenceRegion, 
			Return[minimumNumberOfData]
		];

		maximumWidthOfConfidenceRegion = 
			GetWidthOfConfidenceRegion[
				maximumNumberOfData, 
				indexOfParameter, 
				pureOriginalFunction, 
				argumentRange, 
				standardDeviationRange, 
				modelFunction, 
				argumentOfModelFunction, 
				parametersOfModelFunction,
				UtilityOptionRandomInitializationMode -> randomValueInitialization,
				CalculatedDataOptionDistance -> argumentDistance,
				CalculatedDataOptionDataStructure -> dataStructure,
				CalculatedDataOptionErrorType -> errorType,
				CurveFitOptionConfidenceLevel -> confidenceLevelOfParameterErrors,
				CurveFitOptionCurrentWorkingPrecision -> currentWorkingPrecision,
				CurveFitOptionCurrentMaxIterations -> currentMaxIterations,
				CurveFitOptionStartParameters -> startParameters,
				CurveFitOptionMethod -> method,
				CurveFitOptionVarianceEstimator -> varianceEstimator
			];

		(*
		Print["MinNumber = ", minimumNumberOfData];
		Print["MinWidth  = ", minimumWidthOfConfidenceRegion];
		Print["MaxNumber = ", maximumNumberOfData];
		Print["MaxWidth  = ", maximumWidthOfConfidenceRegion];
		Print[""];
		*)
		
		While[maximumWidthOfConfidenceRegion > desiredWidthOfConfidenceRegion,
			maximumNumberOfData *= 2;
			maximumWidthOfConfidenceRegion = 
				GetWidthOfConfidenceRegion[
					maximumNumberOfData, 
					indexOfParameter, 
					pureOriginalFunction, 
					argumentRange, 
					standardDeviationRange, 
					modelFunction, 
					argumentOfModelFunction, 
					parametersOfModelFunction,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					CalculatedDataOptionDistance -> argumentDistance,
					CalculatedDataOptionDataStructure -> dataStructure,
					CalculatedDataOptionErrorType -> errorType,
					CurveFitOptionConfidenceLevel -> confidenceLevelOfParameterErrors,
					CurveFitOptionCurrentWorkingPrecision -> currentWorkingPrecision,
					CurveFitOptionCurrentMaxIterations -> currentMaxIterations,
					CurveFitOptionStartParameters -> startParameters,
					CurveFitOptionMethod -> method,
					CurveFitOptionVarianceEstimator -> varianceEstimator
				]
			
			(*
			Print["MaxNumber = ", maximumNumberOfData];
			Print["MaxWidth  = ", maximumWidthOfConfidenceRegion];
			Print[""];
			*)
		];
		
		While[maximumNumberOfData - minimumNumberOfData > 1,
			testNumberOfData = Floor[(maximumNumberOfData + minimumNumberOfData)/2];
			testWidthOfConfidenceRegion = 
				GetWidthOfConfidenceRegion[
					testNumberOfData, 
					indexOfParameter, 
					pureOriginalFunction, 
					argumentRange, 
					standardDeviationRange, 
					modelFunction, 
					argumentOfModelFunction, 
					parametersOfModelFunction,
					UtilityOptionRandomInitializationMode -> randomValueInitialization,
					CalculatedDataOptionDistance -> argumentDistance,
					CalculatedDataOptionDataStructure -> dataStructure,
					CalculatedDataOptionErrorType -> errorType,
					CurveFitOptionConfidenceLevel -> confidenceLevelOfParameterErrors,
					CurveFitOptionCurrentWorkingPrecision -> currentWorkingPrecision,
					CurveFitOptionCurrentMaxIterations -> currentMaxIterations,
					CurveFitOptionStartParameters -> startParameters,
					CurveFitOptionMethod -> method,
					CurveFitOptionVarianceEstimator -> varianceEstimator
				];
			If[testWidthOfConfidenceRegion <= desiredWidthOfConfidenceRegion,
				
				maximumNumberOfData = testNumberOfData,
				
				minimumNumberOfData = testNumberOfData
			]
			
			(*
			Print["MinNumber = ", minimumNumberOfData];
			Print["MinWidth  = ", minimumWidthOfConfidenceRegion];
			Print["MaxNumber = ", maximumNumberOfData];
			Print["MaxWidth  = ", maximumWidthOfConfidenceRegion];
			Print[""];
			*)
		];
		
		Return[maximumNumberOfData]
	];

GetParameterConfidenceRegion[

	(* Returns confidence region of specified parameter.

	   Returns: 
	   {lower bound, upper bound} *)
	
    (* Indes of parameter *)
    indexOfParameter_?IntegerQ,

  	(* See "Frequently used data structures" *)
	curveFitInfo_
	
	] :=
    
	Module[
    
    	{
    		parameterConfidenceRegions,
    		fitInfo
    	},

		fitInfo = curveFitInfo[[2]];
		parameterConfidenceRegions = fitInfo[[5]];
		Return[parameterConfidenceRegions[[indexOfParameter]]]
	];

GetParameterValuesAtMinimum[

	(* Returns parameter's values at minimum.

	   Returns: 
	   {value of parameter 1, value of parameter 1, ...} *)
	
    (* Parameters of fit model function : {parameter1, parameter2, ...} *)
    parametersOfModelFunction_,

  	(* See "Frequently used data structures" *)
	curveFitInfo_
	
	] :=
    
	Module[
    
    	{
    		bestFitParameters,
    		fitInfo
    	},

		fitInfo = curveFitInfo[[2]];
		bestFitParameters = fitInfo[[3]];
		Return[parametersOfModelFunction/.bestFitParameters]
	];

GetSingleFitResult[

	(* Returns single fit result according to named property.

	   Returns:
	   Single fit result according to named property *)


	(* Properties to be shown: 
	   Full list: 
	   {
		   "PureFunction",
		   "AbsoluteResiduals",
		   "RelativeResiduals",
		   "SDFit",
		   "ReducedChiSquare",
	       "ModelFunction",
	       "ParameterErrors",
	       "RMSE",
	       "SingleOutputRMSE",
	       "AbsoluteResidualsStatistics",
		   "RelativeResidualsStatistics",
		   "ModelVsData",
		   "CorrelationCoefficient",
		   "SortedModelVsData",
		   "AbsoluteSortedResiduals",
		   "RelativeSortedResiduals"
	    } *)
 	namedProperty_,

    (* {x, y, error} data: {{x1, y1, error1}, {x2, y2, error2}, ...} *)
    xyErrorData_/;MatrixQ[xyErrorData, NumberQ],

  	(* See "Frequently used data structures" *)
	curveFitInfo_
   
	] :=
  
	Module[
    
		{
			argumentOfModelFunction,
			bestFitParameters,
			chiSquare,
			degreesOfFreedom,
			fitInfo,
			fitType,
			i,
			meanSquaredWeights,
			modelFunction,
			numberOfData,
			numberOfParameters,
			parameterConfidenceRegions,
			parameterErrors,
			pureFunction,
			absoluteResiduals,
			relativeResiduals,
			sortedXyErrorData
		},

		numberOfData = Length[xyErrorData];

		fitType = curveFitInfo[[1]];
		fitInfo = curveFitInfo[[2]];
		Switch[fitType,
			
			"ModelFunction",
			modelFunction = fitInfo[[1]];
			argumentOfModelFunction = fitInfo[[2]];
			numberOfParameters = Length[fitInfo[[3]]];
			pureFunction = Function[x, modelFunction/.argumentOfModelFunction -> x],
			
			"SmoothingCubicSplines",
	        pureFunction = Function[x, GetSmoothingCubicSplinesFunctionValue[x, fitInfo]]
		];

		Switch[namedProperty,
			
			(* -------------------------------------------------------------------------------- *)
			"PureFunction",
			Return[pureFunction],
			
			(* -------------------------------------------------------------------------------- *)
			"AbsoluteResiduals",
			sortedXyErrorData = Sort[xyErrorData];
			absoluteResiduals = 
				Table[
					sortedXyErrorData[[i, 2]] - pureFunction[sortedXyErrorData[[i, 1]]],
					
					{i, Length[sortedXyErrorData]}
				];
			Return[absoluteResiduals],
			
			(* -------------------------------------------------------------------------------- *)
			"RelativeResiduals",
			sortedXyErrorData = Sort[xyErrorData];
			relativeResiduals = 
				Table[
					(sortedXyErrorData[[i, 2]] - pureFunction[sortedXyErrorData[[i, 1]]])/sortedXyErrorData[[i, 2]]*100.0,
					
					{i, Length[sortedXyErrorData]}
				];
			Return[relativeResiduals],

			(* -------------------------------------------------------------------------------- *)
			"SDFit",
			Switch[fitType,
				
				"ModelFunction",
				degreesOfFreedom = numberOfData - numberOfParameters,
				
				"SmoothingCubicSplines",
				degreesOfFreedom = numberOfData
			];
			chiSquare = GetChiSquareInternal[xyErrorData, pureFunction];
			meanSquaredWeights = Sum[1.0/xyErrorData[[i, 3]]^2, {i, numberOfData}];
			Return[Sqrt[(chiSquare/degreesOfFreedom)/(meanSquaredWeights/numberOfData)]],
			
			(* -------------------------------------------------------------------------------- *)
			"ReducedChiSquare",
			Switch[fitType,
				
				"ModelFunction",
				degreesOfFreedom = numberOfData - numberOfParameters,
				
				"SmoothingCubicSplines",
				degreesOfFreedom = numberOfData
			];
			Return[GetChiSquareInternal[xyErrorData, pureFunction]/degreesOfFreedom],
			
			(* -------------------------------------------------------------------------------- *)
			"ModelFunction",
			Switch[fitType,
				
				"ModelFunction",
				Return[modelFunction],
				
				"SmoothingCubicSplines",
				Return["There is no ModelFunction output defined for smoothing cubic splines."]
			],
			
			(* -------------------------------------------------------------------------------- *)
			"ParameterErrors",
			Switch[fitType,
				
				"ModelFunction",
				bestFitParameters = fitInfo[[3]];
				parameterErrors = fitInfo[[4]];
				parameterConfidenceRegions = fitInfo[[5]];
				Return[{bestFitParameters, parameterErrors, parameterConfidenceRegions}],
				
				"SmoothingCubicSplines",
				Return["There is no ParameterErrors output defined for smoothing cubic splines."]
			]
		];

		Return[
			CIP`Graphics`GetSingleRegressionResult[
				namedProperty,
				CIP`DataTransformation`TransformXyErrorDataToDataSet[xyErrorData], 
				Function[inputs, CalculateOutputs[inputs, curveFitInfo]]
			]
		]
	];

GetSmoothingCubicSplinesFirstDerivativeValue[

    (* Calculates first derivative value of smoothing cubic splines for argument

	   Returns: 
	   firstDerivativeValue *)

	
    (* Argument: Must be a numeric value *)
	argumentValue_?NumberQ,

    (* Smoothing cubic spline info *)
    smoothingCubicSplinesInfo_ 
   
	] :=
  
	Module[
    
		{
			index, 
			internalArgumentValue
		},
		
		index = Combinatorica`BinarySearch[smoothingCubicSplinesInfo[[1]], argumentValue];
		If[FractionalPart[index] > 0, index = index - 1/2];
		If[index < 1, index = 1];
		If[index >= Length[smoothingCubicSplinesInfo[[1]]], index = Length[smoothingCubicSplinesInfo[[1]]] - 1];
		
		internalArgumentValue = argumentValue - smoothingCubicSplinesInfo[[1, index]];

		Return[(3.0*smoothingCubicSplinesInfo[[2, index, 1]]*internalArgumentValue + 2.0*smoothingCubicSplinesInfo[[2, index, 2]])*internalArgumentValue + smoothingCubicSplinesInfo[[2, index, 3]]]
	];

GetSmoothingCubicSplinesFullFunctionAndDerivativesValues[

    (* Calculates function value and first, second and third derivative values of smoothing cubic splines for argument

	   Returns: 
	   {functionValue, firstDerivativeValue, secondDerivativeValue, thirdDerivativeValue} *)

	
    (* Argument: Must be a numeric value *)
	argumentValue_?NumberQ,

    (* Smoothing cubic spline info *)
    smoothingCubicSplinesInfo_ 
   
	] :=
  
	Module[
    
		{
			index, 
			internalArgumentValue,
			functionValue,
			firstDerivativeValue,
			secondDerivativeValue,
			thirdDerivativeValue
		},
		
		(* Original code:
		
			smoothingCubicSplinesResult:
			Return[{xValues, aCoefficients, bCoefficients, cCoefficients, dCoefficients}]
			
			...
			
			xValues = smoothingCubicSplinesResult[[1]];
			aCoefficients = smoothingCubicSplinesResult[[2]];
			bCoefficients = smoothingCubicSplinesResult[[3]];
			cCoefficients = smoothingCubicSplinesResult[[4]];
			dCoefficients = smoothingCubicSplinesResult[[5]];
	
			Do[
				k = i;
				If[argumentValue <= xValues[[i]], Break[]],
	          
				{i, 2, Length[xValues]}
			];
			internalArgumentValue = argumentValue - xValues[[k - 1]];
			
			(* Compare Horner scheme: function value = argument^3*aCoefficients + argument^2*bCoefficients + argument*cCoefficients + dCoefficients *)
			functionValue = ((aCoefficients[[k - 1]]*internalArgumentValue + bCoefficients[[k - 1]])*internalArgumentValue + cCoefficients[[k - 1]])*internalArgumentValue + dCoefficients[[k - 1]];
			firstDerivativeValue = (3.0*aCoefficients[[k - 1]]*internalArgumentValue + 2.0*bCoefficients[[k - 1]])*internalArgumentValue + cCoefficients[[k - 1]];
			secondDerivativeValue = 6.0*aCoefficients[[k - 1]]*internalArgumentValue + 2.0*bCoefficients[[k - 1]];
			thirdDerivativeValue = 6.0*aCoefficients[[k - 1]];
			
			Return[{functionValue, firstDerivativeValue, secondDerivativeValue, thirdDerivativeValue}]
		
		*)
		
		index = Combinatorica`BinarySearch[smoothingCubicSplinesInfo[[1]], argumentValue];
		If[FractionalPart[index] > 0, index = index - 1/2];
		If[index < 1, index = 1];
		If[index >= Length[smoothingCubicSplinesInfo[[1]]], index = Length[smoothingCubicSplinesInfo[[1]]] - 1];
		
		internalArgumentValue = argumentValue - smoothingCubicSplinesInfo[[1, index]];
		
		(* Compare Horner scheme: function value = argument^3*aCoefficients + argument^2*bCoefficients + argument*cCoefficients + dCoefficients *)
		functionValue = ((smoothingCubicSplinesInfo[[2, index, 1]]*internalArgumentValue + smoothingCubicSplinesInfo[[2, index, 2]])*internalArgumentValue + smoothingCubicSplinesInfo[[2, index, 3]])*internalArgumentValue + smoothingCubicSplinesInfo[[2, index, 4]];
		firstDerivativeValue = (3.0*smoothingCubicSplinesInfo[[2, index, 1]]*internalArgumentValue + 2.0*smoothingCubicSplinesInfo[[2, index, 2]])*internalArgumentValue + smoothingCubicSplinesInfo[[2, index, 3]];
		secondDerivativeValue = 6.0*smoothingCubicSplinesInfo[[2, index, 1]]*internalArgumentValue + 2.0*smoothingCubicSplinesInfo[[2, index, 2]];
		thirdDerivativeValue = 6.0*smoothingCubicSplinesInfo[[2, index, 1]];
		
		Return[{functionValue, firstDerivativeValue, secondDerivativeValue, thirdDerivativeValue}]
	];

GetSmoothingCubicSplinesFunctionAndDerivativesValues[

    (* Calculates function value and first and second derivative values of smoothing cubic splines for argument

	   Returns: 
	   {functionValue, firstDerivativeValue, secondDerivativeValue} *)

	
    (* Argument: Must be a numeric value *)
	argumentValue_?NumberQ,

    (* Smoothing cubic spline info *)
    smoothingCubicSplinesInfo_ 
   
	] :=
  
	Module[
    
		{
			index, 
			internalArgumentValue,
			functionValue,
			firstDerivativeValue,
			secondDerivativeValue
		},

		index = Combinatorica`BinarySearch[smoothingCubicSplinesInfo[[1]], argumentValue];
		If[FractionalPart[index] > 0, index = index - 1/2];
		If[index < 1, index = 1];
		If[index >= Length[smoothingCubicSplinesInfo[[1]]], index = Length[smoothingCubicSplinesInfo[[1]]] - 1];
		
		internalArgumentValue = argumentValue - smoothingCubicSplinesInfo[[1, index]];
		
		(* Compare Horner scheme: function value = argument^3*aCoefficients + argument^2*bCoefficients + argument*cCoefficients + dCoefficients *)
		functionValue = ((smoothingCubicSplinesInfo[[2, index, 1]]*internalArgumentValue + smoothingCubicSplinesInfo[[2, index, 2]])*internalArgumentValue + smoothingCubicSplinesInfo[[2, index, 3]])*internalArgumentValue + smoothingCubicSplinesInfo[[2, index, 4]];
		firstDerivativeValue = (3.0*smoothingCubicSplinesInfo[[2, index, 1]]*internalArgumentValue + 2.0*smoothingCubicSplinesInfo[[2, index, 2]])*internalArgumentValue + smoothingCubicSplinesInfo[[2, index, 3]];
		secondDerivativeValue = 6.0*smoothingCubicSplinesInfo[[2, index, 1]]*internalArgumentValue + 2.0*smoothingCubicSplinesInfo[[2, index, 2]];
		
		Return[{functionValue, firstDerivativeValue, secondDerivativeValue}]
	];
	
GetSmoothingCubicSplinesFunctionValue[

    (* Calculates function value of smoothing cubic splines for argument.

	   Returns: 
	   functionValue *)

	
    (* Argument: Must be a numeric value *)
	argumentValue_?NumberQ,

    (* Smoothing cubic spline info *)
    smoothingCubicSplinesInfo_ 
   
	] :=
  
	Module[
    
		{
			index, 
			internalArgumentValue
		},
		
		index = Combinatorica`BinarySearch[smoothingCubicSplinesInfo[[1]], argumentValue];
		If[FractionalPart[index] > 0, index = index - 1/2];
		If[index < 1, index = 1];
		If[index >= Length[smoothingCubicSplinesInfo[[1]]], index = Length[smoothingCubicSplinesInfo[[1]]] - 1];
		
		internalArgumentValue = argumentValue - smoothingCubicSplinesInfo[[1, index]];
		
		(* Compare Horner scheme: function value = argument^3*aCoefficients + argument^2*bCoefficients + argument*cCoefficients + dCoefficients *)
		Return[
			((smoothingCubicSplinesInfo[[2, index, 1]]*internalArgumentValue + smoothingCubicSplinesInfo[[2, index, 2]])*internalArgumentValue + smoothingCubicSplinesInfo[[2, index, 3]])*internalArgumentValue + smoothingCubicSplinesInfo[[2, index, 4]]
		]
	];

GetSmoothingCubicSplinesSecondDerivativeValue[

    (* Calculates second derivative value of smoothing cubic splines for argument

	   Returns: 
       secondDerivativeValue *)

	
    (* Argument: Must be a numeric value *)
	argumentValue_?NumberQ,

    (* Smoothing cubic spline info *)
    smoothingCubicSplinesInfo_ 
   
	] :=
  
	Module[
    
		{
			index, 
			internalArgumentValue
		},
		
		index = Combinatorica`BinarySearch[smoothingCubicSplinesInfo[[1]], argumentValue];
		If[FractionalPart[index] > 0, index = index - 1/2];
		If[index < 1, index = 1];
		If[index >= Length[smoothingCubicSplinesInfo[[1]]], index = Length[smoothingCubicSplinesInfo[[1]]] - 1];
		
		internalArgumentValue = argumentValue - smoothingCubicSplinesInfo[[1, index]];

		Return[6.0*smoothingCubicSplinesInfo[[2, index, 1]]*internalArgumentValue + 2.0*smoothingCubicSplinesInfo[[2, index, 2]]]
	];

GetSmoothingCubicSplinesThirdDerivativeValue[

    (* Calculates third derivative value of smoothing cubic splines for argument

	   Returns: 
       thirdDerivativeValue *)

	
    (* Argument: Must be a numeric value *)
	argumentValue_?NumberQ,

    (* Smoothing cubic spline info *)
    smoothingCubicSplinesInfo_ 
   
	] :=
  
	Module[
    
		{
			index, 
			internalArgumentValue
		},
		
		index = Combinatorica`BinarySearch[smoothingCubicSplinesInfo[[1]], argumentValue];
		If[FractionalPart[index] > 0, index = index - 1/2];
		If[index < 1, index = 1];
		If[index >= Length[smoothingCubicSplinesInfo[[1]]], index = Length[smoothingCubicSplinesInfo[[1]]] - 1];
		
		internalArgumentValue = argumentValue - smoothingCubicSplinesInfo[[1, index]];

		Return[6.0*smoothingCubicSplinesInfo[[2, index, 1]]]
	];

GetStartParameters[

	(* Returns "optimum" start parameters in defined intervals.

	   Returns: 
	   Chi-square *)
	
    (* {x, y, error} data: {{x1, y1, error1}, {x2, y2, error2}, ...} *)
    xyErrorData_/;MatrixQ[xyErrorData, NumberQ],
	
    (* Model function to be fitted *)
    modelFunction_,
    
    (* Argument of fit model function *)
    argumentOfModelFunction_,
    
    (* Parameters of fit model function : {parameter1, parameter2, ...} *)
    parametersOfModelFunction_,

    (* parameterIntervals: {parameterInterval1, parameterInterval2, ...}
       parameterInterval: {minimum value, maximum value} *)
	parameterIntervals_/;MatrixQ[parameterIntervals, NumberQ],
	
	(* Options *)
	opts___
	
	] :=
    
	Module[
    
    	{
    		bestParameterValues,
    		constraints,
    		i,
    		k,
    		minimizationPrecision,
    		minimumChiSquare,
    		maximumNumberOfIterations,
    		numberOfTrialPoints,
    		parameters,
    		randomValueInitialization,
    		rules,
    		searchType,
    		startParameters,
    		testChiSquare,
    		testParameterValues
    	},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    minimizationPrecision = CurveFitOptionMinimizationPrecision/.{opts}/.Options[CurveFitOptionsStartParameters];
	    maximumNumberOfIterations = CurveFitOptionMaximumIterations/.{opts}/.Options[CurveFitOptionsStartParameters];
	    searchType = CurveFitOptionSearchType/.{opts}/.Options[CurveFitOptionsStartParameters];
	    numberOfTrialPoints = CurveFitOptionNumberOfTrialPoints/.{opts}/.Options[CurveFitOptionsStartParameters];
        randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
        

        If[ randomValueInitialization == "Seed", SeedRandom[1], SeedRandom[]];

		Switch[searchType,
			
			(* -------------------------------------------------------------------------------- *)
			"Random",
			(* -------------------------------------------------------------------------------- *)
			minimumChiSquare = Infinity;
			bestParameterValues = {};
			Do[
				testParameterValues = 
					Table[
						RandomReal[parameterIntervals[[k]]],
						
						{k, Length[parametersOfModelFunction]}
					];
				testChiSquare = GetChiSquare[xyErrorData, modelFunction, argumentOfModelFunction, parametersOfModelFunction, testParameterValues];
				If[testChiSquare < minimumChiSquare,
					minimumChiSquare = testChiSquare;
					bestParameterValues = testParameterValues
				],
					
				{i, numberOfTrialPoints}
			];
			startParameters = 
				Table[
					{parametersOfModelFunction[[i]], bestParameterValues[[i]]}, 
						
					{i, Length[parametersOfModelFunction]}
				],
			
			(* -------------------------------------------------------------------------------- *)
			"NMinimize",
			(* -------------------------------------------------------------------------------- *)
			parameters = 
				Table[
					Subscript[curveFitStartParameter, i], 
					
					{i, Length[parametersOfModelFunction]}
				];
			constraints = 
				Table[
					parameterIntervals[[i, 1]] <= parameters[[i]] <= parameterIntervals[[i, 2]], 
						
					{i, Length[parametersOfModelFunction]}
				];
			rules = Last[
				NMinimize[
					{
						GetChiSquare[xyErrorData, modelFunction, argumentOfModelFunction, parametersOfModelFunction, parameters],
						constraints
					},
					parameters,
					Method -> {"DifferentialEvolution", "PostProcess" -> False},
					AccuracyGoal -> minimizationPrecision, 
					PrecisionGoal -> minimizationPrecision,
					MaxIterations -> maximumNumberOfIterations
				]
			];
			startParameters = 
				Table[
					{parametersOfModelFunction[[i]], parameters[[i]]/.rules[[i]]}, 
						
					{i, Length[parametersOfModelFunction]}
				]
		];

		Return[startParameters];
	];

GetWidthOfConfidenceRegion[

	(* Returns width of confidence region for settings.

	   Returns: 
	   Width of confidence region *)


	(* Number of data *)
	numberOfData_?IntegerQ,
	
	(* Index of parameter for confidence region to check *)
	indexOfParameter_?IntegerQ,

	(* Pure original function *)      
    pureOriginalFunction_,
	
    (* {argumentStart, argumentEnd} *)
    argumentRange_/;VectorQ[argumentRange, NumberQ],

    (* {standardDeviationMin, standardDeviationMax} *)
    standardDeviationRange_/;VectorQ[standardDeviationRange, NumberQ],
	
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
			argumentDistance,
			dataStructure,
			errorType,

			confidenceLevelOfParameterErrors,
			currentMaxIterations,
			currentWorkingPrecision,
			curveFitInfo,
			fitInfo,
			method,
    		parameterConfidenceRegions,
			randomValueInitialization,
			startParameters,
			varianceEstimator,
			xyErrorData
    	},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    confidenceLevelOfParameterErrors = CurveFitOptionConfidenceLevel/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		currentWorkingPrecision = CurveFitOptionCurrentWorkingPrecision/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		currentMaxIterations = CurveFitOptionCurrentMaxIterations/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		startParameters = CurveFitOptionStartParameters/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		method = CurveFitOptionMethod/.{opts}/.Options[CurveFitOptionsFitModelFunction];
		varianceEstimator = CurveFitOptionVarianceEstimator/.{opts}/.Options[CurveFitOptionsFitModelFunction];
        randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
        
        argumentDistance = CalculatedDataOptionDistance/.{opts}/.Options[CalculatedDataOptionsDataGeneration];
        dataStructure = CalculatedDataOptionDataStructure/.{opts}/.Options[CalculatedDataOptionsDataGeneration];
        errorType = CalculatedDataOptionErrorType/.{opts}/.Options[CalculatedDataOptionsDataGeneration];

		xyErrorData = 
			CIP`CalculatedData`GetXyErrorData[
				pureOriginalFunction,
				argumentRange,
				numberOfData,
				standardDeviationRange,
				UtilityOptionRandomInitializationMode -> randomValueInitialization,
				CalculatedDataOptionDistance -> argumentDistance,
				CalculatedDataOptionDataStructure -> dataStructure,
				CalculatedDataOptionErrorType -> errorType
			];

		curveFitInfo =
			FitModelFunction[
				xyErrorData,
				modelFunction,
				argumentOfModelFunction,
				parametersOfModelFunction,
				CurveFitOptionConfidenceLevel -> confidenceLevelOfParameterErrors,
				CurveFitOptionCurrentWorkingPrecision -> currentWorkingPrecision,
				CurveFitOptionCurrentMaxIterations -> currentMaxIterations,
				CurveFitOptionStartParameters -> startParameters,
				CurveFitOptionMethod -> method,
				CurveFitOptionVarianceEstimator -> varianceEstimator
			];
		
		fitInfo = curveFitInfo[[2]];
		parameterConfidenceRegions = fitInfo[[5]];
		Return[
			parameterConfidenceRegions[[indexOfParameter, 2]] - parameterConfidenceRegions[[indexOfParameter, 1]]
		]
	];

ShowBestExponent[

	(* Shows best exponent for a*x^exponent fits to the data.

	   Returns: Nothing *)

    (* {x, y, error} data: {{x1, y1, error1}, {x2, y2, error2}, ...} *)
    xyErrorData_/;MatrixQ[xyErrorData, NumberQ],

  	(* Minimum exponent *)
	minExponent_?NumberQ,

  	(* Maximum exponent *)
	maxExponent_?NumberQ,

  	(* Exponent step size *)
	exponentStepSize_?NumberQ,
	
	(* Options *)
	opts___
   
	] :=

	Module[
    
    	{
    		exponent,
    		labels,
    		pointColor,
    		pointSize,
    		curveFitInfo,
    		sdFit,
    		curveFitInfoList,
    		sdFitList,
    		minPosition,
    		a1,
    		x,
    		exponentList
    	},


		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    labels = CurveFitOptionLabels/.{opts}/.Options[CurveFitOptionsFitResult];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];

		curveFitInfoList = {};
		sdFitList = {};
		exponentList = {};
		Do[
			curveFitInfo = FitModelFunction[xyErrorData, a1*x^exponent, x, {a1}];
			sdFit = GetSingleFitResult["SDFit", xyErrorData, curveFitInfo];
			AppendTo[curveFitInfoList, curveFitInfo];	
			AppendTo[sdFitList, sdFit];
			AppendTo[exponentList, exponent],
			
			{exponent, minExponent, maxExponent, exponentStepSize}
		];

		minPosition = CIP`Utility`GetPositionOfMinimumValue[sdFitList];
		curveFitInfo = curveFitInfoList[[minPosition]];
	    ShowFitResult[
	    	{"FunctionPlot"}, 
	    	xyErrorData, 
	    	curveFitInfo, 
	    	CurveFitOptionLabels -> labels,
	    	GraphicsOptionPointSize -> pointSize,
	    	GraphicsOptionPointSize -> pointColor
	    ];
	    Print["Best exponent = ", exponentList[[minPosition]]]	    
	];

ShowFitResult[

	(* Shows fit results according to named property list.

	   Returns: Nothing *)

	(* Properties to be shown: 
	   Full list: 
	   {
	       "FunctionPlot",
		   "AbsoluteResidualsPlot",
		   "RelativeResidualsPlot",
		   "SDFit",
		   "ReducedChiSquare",
	       "ModelFunction",
	       "ParameterErrors",
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
 	namedPropertyList_,

    (* {x, y, error} data: {{x1, y1, error1}, {x2, y2, error2}, ...} *)
    xyErrorData_/;MatrixQ[xyErrorData, NumberQ],

  	(* See "Frequently used data structures" *)
	curveFitInfo_,
	
	(* Options *)
	opts___
   
	] :=

	Module[
    
    	{
    		i,
    		labels,
    		namedProperty,
    		pointColor,
    		pointSize,
    		numberOfIntervals
    	},


		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    numberOfIntervals = GraphicsOptionNumberOfIntervals/.{opts}/.Options[GraphicsOptionsResidualsDistribution];
	    labels = CurveFitOptionLabels/.{opts}/.Options[CurveFitOptionsFitResult];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];

    	
    	Do[
    		namedProperty = namedPropertyList[[i]];
    		ShowSingleFitResult[
    			namedProperty, 
    			xyErrorData, 
    			curveFitInfo,
    			CurveFitOptionLabels -> labels,
    			GraphicsOptionPointSize -> pointSize,
    			GraphicsOptionPointColor -> pointColor,
				ClusterOptionNumberOfIntervals -> numberOfIntervals
    		],
    		
    		{i, Length[namedPropertyList]}
    	]
	];

ShowSingleFitResult[

	(* Shows single fit result according to named property.

	   Returns: Nothing *)

	(* Properties to be shown: 
	   Full list: 
	   {
	       "FunctionPlot",
		   "AbsoluteResidualsPlot",
		   "RelativeResidualsPlot",
		   "SDFit",
		   "ReducedChiSquare",
	       "ModelFunction",
	       "ParameterErrors",
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

    (* {x, y, error} data: {{x1, y1, error1}, {x2, y2, error2}, ...} *)
    xyErrorData_/;MatrixQ[xyErrorData, NumberQ],

  	(* See "Frequently used data structures" *)
	curveFitInfo_,
	
	(* Options *)
	opts___
   
	] :=
  
	Module[
    
		{
			bestFitParameters,
			fitResult,
			i,
			labels,
			parameterErrorOutput,
			parameterErrors,
			parameterConfidenceRegions,
    		pointSize,
    		pointColor,
    		numberOfIntervals
		},

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
	    numberOfIntervals = GraphicsOptionNumberOfIntervals/.{opts}/.Options[GraphicsOptionsResidualsDistribution];
	    labels = CurveFitOptionLabels/.{opts}/.Options[CurveFitOptionsFitResult];
	    pointSize = GraphicsOptionPointSize/.{opts}/.Options[GraphicsOptionsPoint];
	    pointColor = GraphicsOptionPointColor/.{opts}/.Options[GraphicsOptionsPoint];

		Switch[namedProperty,
			
			(* -------------------------------------------------------------------------------- *)
			"FunctionPlot",
			fitResult = GetSingleFitResult["PureFunction", xyErrorData, curveFitInfo];
			Print[
				CIP`Graphics`Plot2dPointsAboveFunction[
					xyErrorData[[All, {1, 2}]], 
					fitResult, 
					labels,
					GraphicsOptionPointSize -> pointSize,
					GraphicsOptionPointColor -> pointColor
				]
			],
			
			(* -------------------------------------------------------------------------------- *)
			"AbsoluteResidualsPlot",
			fitResult = GetSingleFitResult["AbsoluteResiduals", xyErrorData, curveFitInfo];
			Print[
				CIP`Graphics`PlotResiduals[
					fitResult, 
					{
						"Index of residual", 
						"y - f(x)", 
						"Absolute residuals"
					},
					GraphicsOptionPointSize -> pointSize,
					GraphicsOptionPointColor -> pointColor
				]
			],
			
			(* -------------------------------------------------------------------------------- *)
			"RelativeResidualsPlot",
			fitResult = GetSingleFitResult["RelativeResiduals", xyErrorData, curveFitInfo];
			Print[
				CIP`Graphics`PlotResiduals[
					fitResult, 
					{
						"Index of residual", 
						"(y - f(x))/y * 100", 
						"Relative residuals"
					},
					GraphicsOptionPointSize -> pointSize,
					GraphicsOptionPointColor -> pointColor
				]
			],
			
			(* -------------------------------------------------------------------------------- *)
			"SDFit",
			fitResult = GetSingleFitResult["SDFit", xyErrorData, curveFitInfo];
			Print["Standard deviation of fit = ", ScientificForm[fitResult, 4]],

			(* -------------------------------------------------------------------------------- *)
			"ReducedChiSquare",
			fitResult = GetSingleFitResult["ReducedChiSquare", xyErrorData, curveFitInfo];
			Print["Reduced chi-square of fit = ", ScientificForm[fitResult, 4]],

			(* -------------------------------------------------------------------------------- *)
			"ModelFunction",
			fitResult = GetSingleFitResult["ModelFunction", xyErrorData, curveFitInfo];
			If[Length[fitResult] > 0,
				Print[fitResult]
			],

			(* -------------------------------------------------------------------------------- *)
			"ParameterErrors",
			fitResult = GetSingleFitResult["ParameterErrors", xyErrorData, curveFitInfo];
			bestFitParameters = fitResult[[1]];
			parameterErrors = fitResult[[2]];
			parameterConfidenceRegions = fitResult[[3]];
			parameterErrorOutput =
				Table[
					{
						ToString[bestFitParameters[[i]]],
						ToString[parameterErrors[[i]]],
						ToString[parameterConfidenceRegions[[i]]]},
					
					{i, Length[bestFitParameters]}
				];
			Print[
				TableForm[
					parameterErrorOutput,
					TableHeadings -> {Table["Parameter", {Length[bestFitParameters]}], {"Value", "Standard error", "Confidence region"}}
				]
			]
		];

		(* General regression deviation plots *)
		CIP`Graphics`ShowSingleRegressionResult[
			namedProperty,
			CIP`DataTransformation`TransformXyErrorDataToDataSet[xyErrorData], 
			Function[inputs, CalculateOutputs[inputs, curveFitInfo]],
			GraphicsOptionPointSize -> pointSize,
			GraphicsOptionPointColor -> pointColor,
			ClusterOptionNumberOfIntervals -> numberOfIntervals
		]
	];

(* ::Section:: *)
(* End of Package *)

End[]

EndPackage[]
