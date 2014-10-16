#include "util.hpp"

#include "crs_matrix_base.hpp"

#include "crs_row_view.hpp"
#include "crs_matrix_view.hpp"

#include "ichol_left_unblocked.hpp"

using namespace std;

typedef double value_type;
typedef int    ordinal_type;
typedef int    size_type;

typedef Example::CrsMatrixBase<value_type,ordinal_type,size_type> CrsMatrixBase;
typedef Example::CrsMatrixView<CrsMatrixBase> CrsMatrixView;

typedef Example::Uplo Uplo;

int main (int argc, char *argv[]) {
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " filename" << endl;
    return -1;
  }
  
  CrsMatrixBase A;

  ifstream in;
  in.open(argv[1]);
  if (!in.good()) {
    cout << "Error in open the file: " << argv[1] << endl;
    return -1;
  }
  A.importMatrixMarket(in);
  A.showMe(cout);

  CrsMatrixBase L(A, Uplo::Lower);

  int r_val = Example::ichol_left_unblocked_lower(CrsMatrixView(L));
  if (r_val != 0) 
    cout << " Error = " << r_val << endl;

  L.showMe(cout);  

  return 0;
}