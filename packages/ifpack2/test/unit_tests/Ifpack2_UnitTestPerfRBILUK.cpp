/*
HEADER
// ***********************************************************************
//
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//                 Copyright (2009) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
//@HEADER
*/

/// \file Ifpack2_UnitTestSGSMT.cpp
/// \brief Unit test for multithreaded symmetric Gauss-Seidel

#include "Teuchos_UnitTestHarness.hpp"
#include "Ifpack2_UnitTestHelpers.hpp"
#include "Ifpack2_Relaxation.hpp"
#include "MatrixMarket_Tpetra.hpp"

#include "Ifpack2_AdditiveSchwarz.hpp"
#include "Ifpack2_Experimental_RBILUK.hpp"
#include "Ifpack2_Version.hpp"

#include "MatrixMarket_Tpetra.hpp"
#include "TpetraExt_MatrixMatrix.hpp"
#include "Tpetra_Details_gathervPrint.hpp"
#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_Experimental_BlockCrsMatrix.hpp"
#include "Tpetra_Experimental_BlockMultiVector.hpp"
#include "Tpetra_MatrixIO.hpp"
#include "Tpetra_RowMatrix.hpp"




namespace { // (anonymous)

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_const_cast;
typedef typename Tpetra::global_size_t GST;
typedef typename tif_utest::Node Node;
using std::endl;



#define IFPACK2RBILUK_REPORT_GLOBAL_ERR( WHAT_STRING ) do { \
  reduceAll<int, int> (*tcomm, REDUCE_MIN, lclSuccess, outArg (gblSuccess)); \
  TEST_EQUALITY_CONST( gblSuccess, 1 ); \
  if (gblSuccess != 1) { \
    out << WHAT_STRING << " FAILED on one or more processes!" << endl; \
    for (int p = 0; p < numProcs; ++p) { \
      if (myRank == p && lclSuccess != 1) { \
        std::cout << errStrm.str () << std::flush; \
      } \
      tcomm->barrier (); \
      tcomm->barrier (); \
      tcomm->barrier (); \
    } \
    std::cerr << "TEST FAILED; RETURNING EARLY" << endl; \
    return; \
  } \
} while (false)

TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL(BILU, BILUIMPROVE, Scalar, LO, GO)
{
  using Teuchos::outArg;

  using Teuchos::ArrayView;
  using Teuchos::Array;
  using Teuchos::Comm;
  using Teuchos::ParameterList;
  using std::endl;
  using Teuchos::reduceAll;
  using Teuchos::REDUCE_MIN;


  typedef Tpetra::CrsMatrix<Scalar, LO, GO, Node> crs_matrix_type;
  typedef Tpetra::Map<LO, GO, Node> map_type;
  typedef Tpetra::Vector<Scalar, LO, GO, Node> vec_type;
  typedef Tpetra::RowMatrix<Scalar, LO, GO, Node> row_matrix_type;
  typedef Teuchos::ScalarTraits<Scalar> STS;

  std::ostringstream errStrm; // for error collection



  Teuchos::OSTab tab0 (out);
  out << "Create BLOCK CRS MATRIX FROM GIVEN CRS MATRIX" << endl;
  Teuchos::OSTab tab1 (out);


  RCP<const Comm<int> > tcomm =
    Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();

  const int myRank = tcomm->getRank ();
  const int numProcs = tcomm->getSize ();


  Tpetra::MatrixMarket::Reader<crs_matrix_type> crs_reader;
  std::string file_name = "laplace.mtx"; //"sherman1.mtx";
  int blockSize = 5;
  RCP<const crs_matrix_type> A;
#if 1
  {
    std::cout << "READING" << std::endl;
    A = crs_reader.readSparseFile(file_name, tcomm);
    std::cout << "READ COMPLETE " << std::endl;
  }
#else
  {
    if (myRank == 0){
      read_graph_bin<int, double> (
          &nr, &ne, &xadj, &adj, &vals, argv[1]);
    }
    int nr = 0, ne = 0;
    int *xadj = NULL, *adj= NULL;
    double *vals= NULL;

    tcomm->broadcast (0, sizeof(int), (char *)&nr);
    tcomm->broadcast (0, sizeof(int), (char *)&ne);
    std::cout << "nr:" << nr << " ne:" << ne <<std::endl;
    if (rank != 0){
      xadj = new int [nr+1];
      adj = new int [ne];
      vals = new double [ne];
    }
    tcomm->broadcast (0, sizeof(int) * (nr + 1), (char *)xadj);
    tcomm->broadcast (0, sizeof(int) * (ne), (char *)adj);
    tcomm->broadcast (0, sizeof(double) * (ne), (char *)vals);


    int max_num_elements = 0;
    for (zlno_t lclRow = 0; lclRow < nr; ++lclRow) {
      int begin = xadj[lclRow];
      int end = xadj[lclRow + 1];
      if (end - begin > max_num_elements) max_num_elements = end - begin;
    }

    typedef Tpetra::Map<>::node_type znode_t;
    typedef Tpetra::Map<zlno_t, zgno_t, znode_t> map_t;
    size_t numGlobalCoords = nr;
    RCP<const map_t> map = rcp (new map_t (numGlobalCoords, 0, tcomm));
    RCP<const map_t> domainMap = map;
    RCP<const map_t> rangeMap = map;

    A = RCP<const crs_matrix_type> (new crs_matrix_type (map, 0));

  }
#endif

  typedef Tpetra::Experimental::BlockCrsMatrix<Scalar,LO,GO,Node> block_crs_matrix_type;
  typedef Tpetra::CrsGraph<LO,GO,Node> crs_graph_type;

  typedef Tpetra::Experimental::BlockMultiVector<Scalar,LO,GO,Node> BMV;
  typedef Tpetra::MultiVector<Scalar,LO,GO,Node> MV;







  LO numLocal = A->getNodeNumRows();
#if 0
  RCP<block_crs_matrix_type> bcrsmatrix = rcp_const_cast<block_crs_matrix_type>
            (tif_utest::create_triangular_matrix<Scalar, LO, GO, Node, true> (A->getCrsGraph(), blockSize));
#else
  block_crs_matrix_type abcrsmatrix (*(A->getCrsGraph()), blockSize);
  RCP<block_crs_matrix_type> bcrsmatrix(&abcrsmatrix, false);
  size_t MaxNumEntries = A->getGlobalMaxNumRowEntries();

  Teuchos::Array<LO> InI(MaxNumEntries);
  Teuchos::Array<Scalar> InV(MaxNumEntries);

  Teuchos::Array<Scalar> BInV(MaxNumEntries * blockSize *  blockSize);

  size_t NumIn = 0;

  for (LO i = 0; i < numLocal; ++i){
    A->getLocalRowCopy (i, InI(), InV(), NumIn); // Get Values and Indices

    for (size_t j = 0; j < NumIn; ++j){

      for (int k = 0; k < blockSize * blockSize; ++k){
        if (k % blockSize == k / blockSize){
          //std::cout << "k:" << k << " val:" << k + 1 << std::endl;
          if (i == j)
          BInV[j * blockSize * blockSize + k] = 1000;
	  else 
          BInV[j * blockSize * blockSize + k] = -1;

        }
        else if(k % blockSize > k / blockSize){
          BInV[j * blockSize * blockSize + k] = -1;

        }
        else {
          //std::cout << "k:" << k << " val:" << 0 << std::endl;
          BInV[j * blockSize * blockSize + k] = 0;
        }
      }
    }
    bcrsmatrix->replaceLocalValues(i, InI.getRawPtr(),  BInV.getRawPtr(), NumIn);
  }
#endif

  RCP<const map_type> rowMap = A->getRowMap();
  RCP<const map_type> domainMap = rowMap;
  RCP<const map_type> rangeMap = rowMap;

  typedef Tpetra::Experimental::BlockMultiVector<Scalar,LO,GO,Node> BMV;
  typedef Tpetra::MultiVector<Scalar,LO,GO,Node> MV;

  BMV xBlock (* (A->getCrsGraph()->getRowMap ()), blockSize, 1);
  BMV yBlock (* (A->getCrsGraph()->getRowMap ()), blockSize, 1);

  MV x = xBlock.getMultiVectorView ();
  MV y = yBlock.getMultiVectorView ();
  x.putScalar (Teuchos::ScalarTraits<Scalar>::one ());

  TEST_EQUALITY( x.getMap()->getNodeNumElements (), blockSize * numLocal );
  TEST_EQUALITY( y.getMap ()->getNodeNumElements (), blockSize * numLocal );

  int lclSuccess = 1;
  int gblSuccess = 1;
  typedef Ifpack2::Experimental::RBILUK<block_crs_matrix_type> prec_type;
  RCP<prec_type> prec;
  try {
    RCP<const block_crs_matrix_type> const_bcrsmatrix(bcrsmatrix);
    prec = rcp (new prec_type (const_bcrsmatrix));
  } catch (std::exception& e) {
    lclSuccess = 0;
    errStrm << "Process " << myRank << ": Preconditioner constructor threw exception: "
            << e.what () << endl;
  }
  IFPACK2RBILUK_REPORT_GLOBAL_ERR( "Preconditioner constructor" );

  Teuchos::ParameterList params;
  params.set("fact: iluk level-of-fill", (LO) 0);
  params.set("fact: relax value", 0.0);

  try {
    prec->setParameters (params);
  } catch (std::exception& e) {
    lclSuccess = 0;
    errStrm << "Process " << myRank << ": prec->setParameters() threw exception: "
            << e.what () << endl;
  }
  IFPACK2RBILUK_REPORT_GLOBAL_ERR( "prec->setParameters()" );

  std::cout << "Initing" << std::endl;
  try {
    prec->initialize ();
  } catch (std::exception& e) {
    lclSuccess = 0;
    errStrm << "Process " << myRank << ": prec->initialize() threw exception: "
            << e.what () << endl;
  }
  IFPACK2RBILUK_REPORT_GLOBAL_ERR( "prec->initialize()" );
  std::cout << "computing" << std::endl;
  try {
    prec->compute ();
  } catch (std::exception& e) {
    lclSuccess = 0;
    errStrm << "Process " << myRank << ": prec->compute() threw exception: "
            << e.what () << endl;
  }
  IFPACK2RBILUK_REPORT_GLOBAL_ERR( "prec->compute()" );
  std::cout << "applying" << std::endl;

  try {
    prec->apply (x, y);
  } catch (std::exception& e) {
    lclSuccess = 0;
    errStrm << "Process " << myRank << ": prec->apply(x, y) threw exception: "
            << e.what () << endl;
  }
  IFPACK2RBILUK_REPORT_GLOBAL_ERR( "prec->apply(x, y)" );

  std::cout << "Init Time:" << prec->getInitializeTime() << std::endl;
  std::cout << "Compute Time:" << prec->getComputeTime() << std::endl;
  std::cout << "Apply Time:" << prec->getApplyTime() << std::endl;


}


#define UNIT_TEST_GROUP_SC_LO_GO( Scalar, LO, GO ) \
    TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( BILU, BILUIMPROVE, Scalar, LO, GO )
#include "Ifpack2_ETIHelperMacros.h"

IFPACK2_ETI_MANGLING_TYPEDEFS()

// TODO (mfh 24 Aug 2016) Test complex Scalar types

IFPACK2_INSTANTIATE_SLG_REAL( UNIT_TEST_GROUP_SC_LO_GO )

} // namespace (anonymous)


