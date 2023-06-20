#!/bin/bash

foamCleanTutorials
blockMesh | tee log.blockMesh
pisoFoam | tee log.solver
touch taylor_green.foam
