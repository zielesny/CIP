(*
-----------------------------------------------------------------------
Computational Intelligence Packages (CIP): Package CalculatedData
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

BeginPackage["CIP`CalculatedData`", {"CIP`Utility`", "CIP`DataTransformation`"}]

(* ::Section:: *)
(* Off settings *)

Off[General::"spell1"]
Off[General::shdw]
Off[General::compat]

(* ::Section:: *)
(* Options *)

Options[CalculatedDataOptionsDataGeneration] = 
{
	(* "Equal", "LogSmallToLarge", "LogLargeToSmall" *)
	CalculatedDataOptionDistance -> "Equal",
	
	(* "Absolute", "Relative" *)
	CalculatedDataOptionErrorType -> "Absolute",
	
	(* "XyErrorData", "DataSet" *)
	CalculatedDataOptionDataStructure -> "XyErrorData"
}

(* ::Section:: *)
(* Declarations *)

GetFunction3DBasedDataSet::usage = 
    "GetFunction3DBasedDataSet[pureModelFunction, xRange, yRange, numberOfDataPerDimension, standardDeviationRange, options]"

GetFunction3DsBasedDataSet::usage = 
    "GetFunction3DsBasedDataSet[modelFunctions, argumentRange1, argumentRange2, numberOfDataPerDimension, standardDeviationRange, options]"
    
GetDefinedGaussianCloud::usage = 
    "GetDefinedGaussianCloud[cloudDefinition, options]"

GetGaussianCloudsDataSet::usage = 
    "GetGaussianCloudsDataSet[cloudDefinitions, options]"

GetFunctionsBasedDataSet::usage = 
    "GetFunctionsBasedDataSet[modelFunctions, argumentsAndRanges, numberOfData, standardDeviationRange, options]"

GetRandomGaussianCloudsInputs::usage = 
    "GetRandomGaussianCloudsInputs[cloudVectorNumberList, numberOfDimensions, standardDeviation, options]"
    
GetRandomGaussianCloudsDataSet::usage = 
    "GetRandomGaussianCloudsDataSet[cloudVectorNumberList, numberOfDimensions, standardDeviation, options]"
    
GetRandomVectors::usage = 
    "GetRandomVectors[numberOfDimensions, numberOfRandomVectors, options]"
    
GetRandomVectorsInHypercube::usage = 
    "GetRandomVectorsInHypercube[numberOfDimensions, numberOfRandomVectors, options]"

GetXyErrorData::usage = 
    "GetXyErrorData[pureFunction, argumentRange, numberOfData, standardDeviationRange, options]"

(* ::Section:: *)
(* Functions *)
    
Begin["`Private`"]

GetFunction3DBasedDataSet[

    (* Returns a 3D function (with two arguments) based erroneous data set where inputs are grid points in the 2D input space 
       and corresponding 1D output (function value).
       Data set gets numberOfDataPerDimension^2 IOPairs.
       Errors are added as absolute values to the function values.
       standardDeviation defines the standard deviation of the normal distribution.
       For each single standard deviation a random value is chosen between standardDeviationMin and standardDeviationMax.
       Use standardDeviationMin = standardDeviationMax to get the same standard deviation (equal to standardDeviationMin) in each case.

       Returns:
       dataSet = {IOPair1, IOPair2, ..., IOPair<numberOfData>}
       IOPair: {inputs, outputs}
       inputs: {argumentValue1, argumentValue2, ..., argumentValue<Length[argumentsAndRanges]>}
       outputs: {functionValue1, functionValue2, ..., functionValue<Length[modelFunctions]>} *)


    (* Pure model function f(x,y) *)      
    pureModelFunction_,

    (* {xMin, xMax} *)      
    xRange_/;VectorQ[xRange, NumberQ],

    (* {yMin, yMax} *)      
    yRange_/;VectorQ[yRange, NumberQ],

    (* Number of data per each of the 2 input dimensions: 
       Data set gets numberOfDataPerDimension^2 IOPairs *)
    numberOfDataPerDimension_?IntegerQ,

    (* {standardDeviationMin, standardDeviationMax} *)
    standardDeviationRange_/;VectorQ[standardDeviationRange, NumberQ],
    
    (* Options *)
    opts___
      
    ] :=
    Module[ 
    	
    	{
            randomValueInitialization,
    		xMin,
    		xMax,
    		yMin,
    		yMax,
            dataSet,
            x,
            y,
            standardDeviationMin,
            standardDeviationMax,
            input,
            output,
            standardDeviation
        },
        
        randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
       
        If[ randomValueInitialization == "Seed", SeedRandom[1], SeedRandom[]];
        
        xMin = xRange[[1]];
        xMax = xRange[[2]];
        yMin = yRange[[1]];
        yMax = yRange[[2]];
        standardDeviationMin = standardDeviationRange[[1]];
        standardDeviationMax = standardDeviationRange[[2]];
        dataSet = {};
        Do[
            Do[
                (* Set input *)
                input = {x, y};
                (* Get output function values of arguments *)
                output =
					If[ standardDeviationMin <= 0.0 || standardDeviationMax <= 0.0,
                        
                        (* True *)
                        {pureModelFunction[x, y]},

                        (* false *)
                        standardDeviation = RandomReal[standardDeviationRange];
                        {pureModelFunction[x, y] + RandomReal[NormalDistribution[0.0, standardDeviation]]}
                    ];
                AppendTo[dataSet, {input, output}],
                
                {y, yMin, yMax, (yMax - yMin)/(numberOfDataPerDimension - 1)}
            ],
			{x, xMin, xMax, (xMax - xMin)/(numberOfDataPerDimension - 1)}
        ];
        Return[dataSet];
    ];

GetFunction3DsBasedDataSet[

    (* Returns 3D functions (with two arguments) based erroneous data set where inputs are grid points in the 2D input space.
       Data set gets numberOfDataPerDimension^2 IOPairs.
       Errors are added as absolute values to the function values.
       standardDeviation defines the standard deviation of the normal distribution.
       For each single standard deviation a random value is chosen between standardDeviationMin and standardDeviationMax.
       Use standardDeviationMin = standardDeviationMax to get the same standard deviation (equal to standardDeviationMin) in each case.

       Returns:
       dataSet = {IOPair1, IOPair2, ..., IOPair<numberOfData>}
       IOPair: {inputs, outputs}
       inputs: {argumentValue1, argumentValue2, ..., argumentValue<Length[argumentsAndRanges]>}
       outputs: {functionValue1, functionValue2, ..., functionValue<Length[modelFunctions]>} *)


    (* {modelfunction1, modelfunction2, ...} *)      
    modelFunctions_,

    (* argumentRange1: {argument1, start1, end1} *)      
    argumentRange1_,

    (* argumentRange2: {argument2, start2, end2} *)      
    argumentRange2_,

    (* Number of data per each of the 2 input dimensions: 
       Data set gets numberOfDataPerDimension^2 IOPairs *)
    numberOfDataPerDimension_?IntegerQ,

    (* {standardDeviationMin, standardDeviationMax} *)
    standardDeviationRange_/;VectorQ[standardDeviationRange, NumberQ],
    
    (* Options *)
    opts___
      
    ] :=
    Module[ {
            dataSet,
            input,
            k,
            output,
            randomValueInitialization,
            replacement,
            standardDeviation,
            x,
            y
        },
        
        randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
       
        If[ randomValueInitialization == "Seed", SeedRandom[1], SeedRandom[]];
        
        dataSet = {};
        Do[
            Do[
                (* Set input *)
                input = {x, y};
                (* Set arguments with input values *)
                replacement = {argumentRange1[[1]] -> x, argumentRange2[[1]] -> y};
                (* Get output function values of arguments *)
                output = Table[
                    If[ standardDeviationRange[[1]] <= 0. || standardDeviationRange[[2]] <= 0.,
                        
                        (* True *)
                        N[modelFunctions[[k]]/.replacement],

                        (* False *)
                        standardDeviation = RandomReal[standardDeviationRange];
                        N[modelFunctions[[k]]/.replacement] + RandomReal[NormalDistribution[0, standardDeviation]]
                    ],
                    
                    {k, Length[modelFunctions]}
                ];
                AppendTo[dataSet, {input, output}],
                
                {y, argumentRange2[[2]], argumentRange2[[3]], (argumentRange2[[3]] - argumentRange2[[2]])/(numberOfDataPerDimension - 1)}
            ],
            {x, argumentRange1[[2]], argumentRange1[[3]], (argumentRange1[[3]] - argumentRange1[[2]])/(numberOfDataPerDimension - 1)}
        ];
        Return[dataSet];
    ];

GetDefinedGaussianCloud[

    (* Returns random cloud vectors normally distributed with standard deviation around centroid vector.

       Returns:
       Random cloud vectors with same number of dimensions as centroid vector *)


    (* cloudDefinition: {centroidVector, numberOfCloudVectors, standardDeviation} *)
	cloudDefintion_,

    (* Options *)
    opts___
    
    ] :=
    
    Module[ 
    	
    	{
    		centroidVector,
            i,
            numberOfCloudVectors,
            randomVectors,
            randomValueInitialization,
            standardDeviation
        },
        
		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
        randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
	    
        centroidVector = cloudDefintion[[1]];
        numberOfCloudVectors = cloudDefintion[[2]];
        standardDeviation = cloudDefintion[[3]];
        
        If[ randomValueInitialization == "Seed",
        	
            SeedRandom[1],

            SeedRandom[]
        ];

        randomVectors = RandomReal[NormalDistribution[0.0, standardDeviation], {numberOfCloudVectors, Length[centroidVector]}];
        Return[
	        Table[
	            centroidVector + randomVectors[[i]],
	            
	            {i, numberOfCloudVectors}
	        ]
        ]
    ];
    
GetGaussianCloudsDataSet[

    (* Returns data set for classification with gaussian cloud vectors around defined centroid vectors.

       Returns:
       Returns : {IOPair1, IOPair2, ...}
       IOPair : {Input, Output}
       Input : {inputComponent1, inputComponent2, ..., inputComponent <numberOfDimensions>}
       Output : {0/1, ..., 0/1<numberOfCentroidVectors>} *)


    (* cloudDefintions: {cloudDefinition1, cloudDefinition2, ...}
       cloudDefinition: {centroidVector, numberOfCloudVectors, standardDeviation} *)
    cloudDefintions_,
      
    (* Options *)
    opts___
      
    ] :=
    Module[ {
            cloudVectors,
            classificationDataSet,
            i,
            ioPair,
            j,
            outputVector,
            randomValueInitialization
        },

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
        randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];

        classificationDataSet = {};
        Do[
            cloudVectors = 
                GetDefinedGaussianCloud[
                    cloudDefintions[[i]], 
                    UtilityOptionRandomInitializationMode -> randomValueInitialization
                ];
            outputVector = 
                Table[
                    If[ i == j,
                        1.0,
                        0.0
                    ], 
                        
                    {j, Length[cloudDefintions]}
                ];
            Do[
                ioPair = {cloudVectors[[j]], outputVector};
                AppendTo[classificationDataSet, ioPair],
                
                {j, Length[cloudVectors]}
            ],
            
            {i, Length[cloudDefintions]}
        ];
        Return[classificationDataSet];
    ];

GetFunctionsBasedDataSet[

    (* Returns function based erroneous data set.
       Errors are added as absolute values to the function values.
       standardDeviation defines the standard deviation of the normal distribution.
       For each single standard deviation a random value is chosen between standardDeviationMin and standardDeviationMax.
       Use standardDeviationMin = standardDeviationMax to get the same standard deviation (equal to standardDeviationMin) in each case.

       Returns:
       dataSet = {IOPair1, IOPair2, ..., IOPair<numberOfData>}
       IOPair: {inputs, outputs}
       inputs: {argumentValue1, argumentValue2, ..., argumentValue<Length[argumentsAndRanges]>}
       outputs: {functionValue1, functionValue2, ..., functionValue<Length[modelFunctions]>} *)


    (* {modelfunction1, modelfunction2, ...} *)      
    modelFunctions_,

    (* {argumentRange1, argumentRange2, ...} 
       argumentRange: {argument, start, end} *)      
    argumentsAndRanges_,

    (* Number of data *)
    numberOfData_?IntegerQ,

    (* {standardDeviationMin, standardDeviationMax} *)
    standardDeviationRange_/;VectorQ[standardDeviationRange, NumberQ],
    
    (* Option for UtilityOptionRandomInitializationMode:
       "Seed"  : Deterministic random sequence with SeedRandom[1] 
       "NoSeed": Random random sequence with SeedRandom[] *)
    opts___
      
    ] :=
    Module[ {
            dataSet,
            i,
            input,
            k,
            output,
            randomValueInitialization,
            replacement,
            standardDeviation
        },
        randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
        If[ randomValueInitialization == "Seed",
            SeedRandom[1],
            SeedRandom[]
        ];
        dataSet = {};
        Do[
            (* Random input *)
            input = Table[
                RandomReal[{argumentsAndRanges[[k, 2]], argumentsAndRanges[[k, 3]]}],
                
                {k, Length[argumentsAndRanges]}
            ];
            (* Set arguments with random input values *)
            replacement = {};
            Do[
                AppendTo[replacement, argumentsAndRanges[[k, 1]] -> input[[k]]],
                
                {k, Length[argumentsAndRanges]}
            ];
            (* Get output function values of arguments *)
            output = Table[
                If[ standardDeviationRange[[1]] <= 0. || standardDeviationRange[[2]] <= 0.,
                    
                    (* True *)
                    N[modelFunctions[[k]]/.replacement],

                    (* False *)
                    standardDeviation = RandomReal[standardDeviationRange];
                    N[modelFunctions[[k]]/.replacement] + RandomReal[NormalDistribution[0, standardDeviation]]
                ],
                
                {k, Length[modelFunctions]}
            ];
            AppendTo[dataSet, {input, output}],
            
            {i, numberOfData}
        ];
        Return[dataSet];
    ];

GetRandomGaussianCloudsInputs[

    (* Returns vector list with gaussian cloud vectors around random centroid vectors in a unit hypercube (components of centroid vectors are in interval [0, 1]).

       Returns:
       numberOfCentroidVectors = Length[cloudVectorNumberList]
       {Input1, ..., Input<numberOfCentroidVectors*numberOfCloudVectors>}
       Input: {Component1, ...Component<numberOfDimensions>} *)

      
    (* List with number of cloud vectors for each random centroid vector *)
    cloudVectorNumberList_/;VectorQ[cloudVectorNumberList, NumberQ],
      
    (* Defines dimension of vectors *)
    numberOfDimensions_?IntegerQ,
      
    (* Defines cloud size *)
    standardDeviation_?NumberQ,
      
    (* Option for UtilityOptionRandomInitializationMode:
       "Seed"  : Deterministic random sequence with SeedRandom[1] 
       "NoSeed": Random random sequence with SeedRandom[] *)
    opts___
      
    ] :=
    Module[ 
    	
    	{
            i,
            randomCentroidVectorList,
            randomValueInitialization,
            randomVectorsAroundCentroids
        },
        
        randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
        
        randomCentroidVectorList = GetRandomVectorsInHypercube[numberOfDimensions, Length[cloudVectorNumberList], UtilityOptionRandomInitializationMode -> randomValueInitialization];
        randomVectorsAroundCentroids = {};
        Do[
            randomVectorsAroundCentroids = Join[randomVectorsAroundCentroids, 
                GetDefinedGaussianCloud[
                	{randomCentroidVectorList[[i]], cloudVectorNumberList[[i]], standardDeviation},
                	UtilityOptionRandomInitializationMode -> randomValueInitialization]
                ],
                
            {i, Length[cloudVectorNumberList]}
        ];
        Return[randomVectorsAroundCentroids];
    ]
    
GetRandomGaussianCloudsDataSet[

    (* Returns data set for classification with gaussian cloud vectors around random centroid vectors in a unit hypercube.

       Returns:
       numberOfCentroidVectors = Length[cloudVectorNumberList]
       Returns : {IOPair1, ..., IOPair<numberOfCentroidVectors*numberOfCloudVectors>}
       IOPair : {Input, Output}
       Input : {inputComponent1, inputComponent2, ..., inputComponent <numberOfDimensions>}
       Output : {0/1, ..., 0/1<numberOfCentroidVectors>} *)


    (* List with number of cloud vectors for each random centroid vector *)
    cloudVectorNumberList_/;VectorQ[cloudVectorNumberList, NumberQ],
      
    (* Defines dimension of vectors *)
    numberOfDimensions_?IntegerQ,
      
    (* Defines cloud size *)
    standardDeviation_?NumberQ,
      
    (* Option for UtilityOptionRandomInitializationMode:
       "Seed"  : Deterministic random sequence with SeedRandom[1] 
       "NoSeed": Random random sequence with SeedRandom[] *)
    opts___
      
    ] :=
    Module[ 
    	
    	{
            i,
            inputVectorList,
            j,
            outputVector,
            ioPair,
            randomCentroidVectorList,
            randomIOVectorsAroundCentroids,
            randomValueInitialization
        },
        
        randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
        
        randomCentroidVectorList = GetRandomVectorsInHypercube[numberOfDimensions, Length[cloudVectorNumberList], UtilityOptionRandomInitializationMode -> randomValueInitialization];
        randomIOVectorsAroundCentroids = {};
        Do[
            inputVectorList = 
            	GetDefinedGaussianCloud[
            		{randomCentroidVectorList[[i]], cloudVectorNumberList[[i]], standardDeviation}, 
            		UtilityOptionRandomInitializationMode -> randomValueInitialization
            	];
            outputVector = 
            	Table[
            		If[i == j,
						1.0,
						0.0
                    ],
                    
                    {j, Length[cloudVectorNumberList]}
				];
            Do[
                ioPair = {inputVectorList[[j]], outputVector};
                AppendTo[randomIOVectorsAroundCentroids, ioPair],
                
                {j, cloudVectorNumberList[[i]]}
            ],
            
            {i, Length[cloudVectorNumberList]}
        ];
        Return[randomIOVectorsAroundCentroids];
    ];

GetRandomVectors[

    (* Returns random vectors with specified number of dimensions with components in [-1, 1]

       Returns:
       {randomVector1, randomVector2, ...}
       randomVector1 = {randomValue1, ..., randomValue<numberOfDimensions>}
       -1 <= randomValue <= 1 *)


    (* Number of dimensions of each random vector *)
    numberOfDimensions_?IntegerQ,

    (* Number of random vectors *)
    numberOfRandomVectors_?IntegerQ,

    (* Option for UtilityOptionRandomInitializationMode:
       "Seed"  : Deterministic random sequence with SeedRandom[1] 
       "NoSeed": Random random sequence with SeedRandom[] *)
    opts___
    
    ] :=
    Module[ {
            randomValueInitialization
        },

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
        randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];

        If[ randomValueInitialization == "Seed",
            SeedRandom[1],
            SeedRandom[]
        ];

        Return[
        	RandomReal[{-1., 1.}, {numberOfRandomVectors, numberOfDimensions}]
        ]
    ];

GetRandomVectorsInHypercube[

    (* Returns random vectors in unit hypercube with components in [0, 1]

       Returns:
       {randomValue1, ..., randomValue < numberOfDimensions >}
       {0 <= randomValue <= 1 *)
    
    
    numberOfDimensions_?IntegerQ,
    
    numberOfRandomVectors_?IntegerQ,
    
    (* Option for UtilityOptionRandomInitializationMode:
       "Seed"  : Deterministic random sequence with SeedRandom[1] 
       "NoSeed": Random random sequence with SeedRandom[] *)
    opts___
    
    ] :=
    Module[ {
            randomValueInitialization
        },
        randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
        If[ randomValueInitialization == "Seed",
            SeedRandom[1],
            SeedRandom[]
        ];
        Return[
            Table[
                RandomReal[{0, 1}, numberOfDimensions],
                {numberOfRandomVectors}
            ]
        ]
    ];

GetXyErrorData[

    (* Returns {x, y, error of y} list with random normally distributed data around specified pure function. 
       Errors are added as absolute/relative values to the data values according to option.
       StandardDeviation defines the standard deviation of the normal distribution.
       For each single standard deviation a random value is chosen between standardDeviationMin and standardDeviationMax.
       Use standardDeviationMin = standardDeviationMax to get the same standard deviation (equal to standardDeviationMin) in each case.

       Returns:
       {{x1, y1, error1}, ..., {x<numberOfData>, y<numberOfData>, error<numberOfData>}}
       x1 = argumentStart, x<numberOfData> = argumentEnd *)


	(* Pure function *)      
    pureFunction_,

    (* {argumentStart, argumentEnd} *)
    argumentRange_/;VectorQ[argumentRange, NumberQ],

    numberOfData_?IntegerQ,

    (* {standardDeviationMin, standardDeviationMax} *)
    standardDeviationRange_/;VectorQ[standardDeviationRange, NumberQ],

    (* Options *)
    opts___
      
    ] :=
    Module[ 
    	
    	{
    		arguments,
    		argumentDistance,
    		errorType,
    		i,
    		intermediateResult,
    		newArgumentRange,
    		offset,
            randomValueInitialization,
            dataStructure,
            singleIncrement,
            standardDeviation,
            xyErrorData
        },

		(* ----------------------------------------------------------------------------------------------------
		   Options
		   ---------------------------------------------------------------------------------------------------- *)
        randomValueInitialization = UtilityOptionRandomInitializationMode/.{opts}/.Options[UtilityOptionsRandomInitialization];
        argumentDistance = CalculatedDataOptionDistance/.{opts}/.Options[CalculatedDataOptionsDataGeneration];
        dataStructure = CalculatedDataOptionDataStructure/.{opts}/.Options[CalculatedDataOptionsDataGeneration];
        errorType = CalculatedDataOptionErrorType/.{opts}/.Options[CalculatedDataOptionsDataGeneration];

        If[ randomValueInitialization == "Seed",
            SeedRandom[1],
            SeedRandom[]
        ];

		Switch[argumentDistance,
			
			"Equal",	
			singleIncrement = (argumentRange[[2]] - argumentRange[[1]])/(numberOfData - 1);
			arguments = 
				Table[
					argumentRange[[1]] + (i-1)*singleIncrement, 
						
					{i, numberOfData}
				],
		
			"LogSmallToLarge",
			If[argumentRange[[1]] < 1.0,

				offset = 1.0 - argumentRange[[1]];
				newArgumentRange = argumentRange + offset,
				
				offset = 0.0;
				newArgumentRange = argumentRange
			];
			singleIncrement = (Log[newArgumentRange[[2]]] - Log[newArgumentRange[[1]]])/(numberOfData - 1);
			arguments = 
				Table[
					Exp[Log[newArgumentRange[[1]]] + (i-1)*singleIncrement] - offset,
					
					{i, numberOfData}
				],
			
			"LogLargeToSmall",
			If[argumentRange[[1]] < 1.0,

				offset = 1.0 - argumentRange[[1]];
				newArgumentRange = argumentRange + offset,
				
				offset = 0.0;
				newArgumentRange = argumentRange
			];
			singleIncrement = (Log[newArgumentRange[[2]]] - Log[newArgumentRange[[1]]])/(numberOfData - 1);
			intermediateResult = 
				Table[
					Exp[Log[newArgumentRange[[1]]] + (i-1)*singleIncrement] - offset,
					
					{i, numberOfData}
				];
			arguments = 
				Table[
					argumentRange[[1]] + (argumentRange[[2]] - intermediateResult[[numberOfData - i + 1]]),
					
					{i, numberOfData}
				];
		];
		
        If[ standardDeviationRange[[1]] <= 0. || standardDeviationRange[[2]] <= 0.,
            
            (* True *)
            xyErrorData = 
                Table[
                    {
                    	arguments[[i]], 
                    	pureFunction[arguments[[i]]], 
                    	0.0
                    },
                    
                    {i, numberOfData}
                ],

            (* False *)
            Switch[errorType,

				"Absolute",
	            xyErrorData = 
	                Table[
	                    standardDeviation = RandomReal[standardDeviationRange];
	                    {
	                    	arguments[[i]], 
	                    	pureFunction[arguments[[i]]] + RandomReal[NormalDistribution[0.0, standardDeviation]], 
	                    	standardDeviation
	                    },
	                    
	                    {i, numberOfData}
	                ],
	                
	            "Relative",
	            xyErrorData = 
	                Table[
	                    standardDeviation = RandomReal[standardDeviationRange];
	                    {
	                    	arguments[[i]], 
							pureFunction[arguments[[i]]]*(1.0 + RandomReal[NormalDistribution[0.0, standardDeviation]]), 
	                        pureFunction[arguments[[i]]]*standardDeviation
						},
	                    
	                    {i, numberOfData}
	                ]
            ]
        ];

		Switch[dataStructure,
			
			"XyErrorData",	
	        Return[xyErrorData],
	        
	        "DataSet",
	        Return[CIP`DataTransformation`TransformXyErrorDataToDataSet[xyErrorData]]
		]
    ];

(* ::Section:: *)
(* End of Package *)

End[]

EndPackage[]
