

#
# A) Package-specific configuration options
#

#
# B) Define the header and source files (and directories)
#

#
# src
#

SET(HEADERS "")
SET(SOURCES "")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

#
# Core Files
#

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

APPEND_SET(HEADERS
  BelosEpetraAdapter.hpp
  BelosEpetraOperator.h
  Belos_Details_Epetra_registerLinearSolverFactory.hpp
  )

APPEND_SET(SOURCES
  BelosEpetraAdapter.cpp
  BelosEpetraOperator.cpp
  Belos_Details_Epetra_registerLinearSolverFactory.cpp
  )

#
# C) Define the targets for package's library(s)
#
TRIBITS_ADD_LIBRARY(
  belosepetra
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  DEPLIBS belos
  )
