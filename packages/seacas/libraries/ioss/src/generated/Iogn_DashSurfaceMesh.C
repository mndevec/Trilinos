// Copyright(C) 2014
// Sandia Corporation. Under the terms of Contract
// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
// certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//     * Neither the name of Sandia Corporation nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "generated/Iogn_GeneratedMesh.h" // for MapVector, IntVector, etc
#include <algorithm>                      // for copy
#include <generated/Iogn_DashSurfaceMesh.h>
#include <vector> // for vector

namespace Iogn {

  int64_t DashSurfaceMesh::node_count() const { return mDashSurfaceData.globalNumberOfNodes; }

  int64_t DashSurfaceMesh::node_count_proc() const
  {
    return mDashSurfaceData.coordinates.size() / SPATIAL_DIMENSION;
  }

  int64_t DashSurfaceMesh::element_count() const { return mDashSurfaceData.globalNumberOfElements; }

  int64_t DashSurfaceMesh::element_count(int64_t surfaceNumber) const
  {
    if (surfaceNumber == 1) {
      return mDashSurfaceData.globalNumberOfElementsSurface1;
    }
    if (surfaceNumber == 2) {
      return mDashSurfaceData.globalNumberOfElementsSurface2;
    }
    throw std::exception();
  }

  int64_t DashSurfaceMesh::block_count() const { return NUMBER_OF_SURFACES; }

  int64_t DashSurfaceMesh::nodeset_count() const { return 0; }

  int64_t DashSurfaceMesh::sideset_count() const { return NUMBER_OF_SURFACES; }

  int64_t DashSurfaceMesh::element_count_proc() const
  {
    return (mDashSurfaceData.surfaceBConnectivity.size() +
            mDashSurfaceData.surfaceAConnectivity.size()) /
           NUM_NODES_PER_QUAD_FACE;
  }

  int64_t DashSurfaceMesh::element_count_proc(int64_t block_number) const
  {
    if (block_number == 1) {
      return mDashSurfaceData.surfaceBConnectivity.size() / NUM_NODES_PER_QUAD_FACE;
    }
    if (block_number == 2) {
      return mDashSurfaceData.surfaceAConnectivity.size() / NUM_NODES_PER_QUAD_FACE;
    }
    throw std::exception();
  }

  int64_t DashSurfaceMesh::nodeset_node_count_proc(int64_t /*id*/) const { return 0; }

  int64_t DashSurfaceMesh::sideset_side_count_proc(int64_t id) const
  {
    return element_count_proc(id);
  }

  int64_t DashSurfaceMesh::communication_node_count_proc() const
  {
    if (mDashSurfaceData.sharedNodes != nullptr) {
      return mDashSurfaceData.sharedNodes->size();
    }

    return 0;
  }

  void DashSurfaceMesh::coordinates(double *coord) const
  {
    std::copy(mDashSurfaceData.coordinates.begin(), mDashSurfaceData.coordinates.end(), coord);
  }

  void DashSurfaceMesh::coordinates(std::vector<double> & /*coord*/) const
  {
    throw std::exception();
  }

  void DashSurfaceMesh::coordinates(int /*component*/, std::vector<double> & /*xyz*/) const
  {
    throw std::exception();
  }

  void DashSurfaceMesh::coordinates(std::vector<double> & /*x*/, std::vector<double> & /*y*/,
                                    std::vector<double> & /*z*/) const
  {
    throw std::exception();
  }

  void DashSurfaceMesh::connectivity(int64_t block_number, int *connect) const
  {
    switch (block_number) {
    case 1:
      std::copy(mDashSurfaceData.surfaceBConnectivity.begin(),
                mDashSurfaceData.surfaceBConnectivity.end(), connect);
      return;
    case 2:
      std::copy(mDashSurfaceData.surfaceAConnectivity.begin(),
                mDashSurfaceData.surfaceAConnectivity.end(), connect);
      return;
    default: throw std::exception();
    }
  }

  std::pair<std::string, int> DashSurfaceMesh::topology_type(int64_t /*block_number*/) const
  {
    const int numNodesPerElement = 4;
    return std::make_pair(std::string("shell4"), numNodesPerElement);
  }

  void DashSurfaceMesh::sideset_elem_sides(int64_t setId, Ioss::Int64Vector &elem_sides) const
  {
    elem_sides.clear();
    size_t numElementsInSurface1 = element_count_proc(1);
    size_t numElementsInSurface2 = element_count_proc(2);
    switch (setId) {
    case 1:
      for (size_t i = 0; i < numElementsInSurface1; ++i) {
        elem_sides.push_back(mDashSurfaceData.globalIdsOfLocalElements[i]);
        elem_sides.push_back(0);
      }
      return;
    case 2:
      for (size_t i = 0; i < numElementsInSurface2; ++i) {
        elem_sides.push_back(mDashSurfaceData.globalIdsOfLocalElements[numElementsInSurface1 + i]);
        elem_sides.push_back(0);
      }
      return;
    default: throw std::exception();
    }
  }

  void DashSurfaceMesh::nodeset_nodes(int64_t /*nset_id*/, Ioss::Int64Vector & /*nodes*/) const {}

  void DashSurfaceMesh::node_communication_map(MapVector &map, std::vector<int> &proc)
  {
    if (mDashSurfaceData.sharedNodes == nullptr) {
      return;
    }

    for (unsigned int i = 0; i < mDashSurfaceData.sharedNodes->size(); i++) {
      map[i]  = (*mDashSurfaceData.sharedNodes)[i].nodeId;
      proc[i] = (*mDashSurfaceData.sharedNodes)[i].procId;
    }
    return;
  }

  void DashSurfaceMesh::node_map(Ioss::IntVector &map) const
  {
    int size = node_count_proc();
    map.resize(size);

    for (int i = 0; i < size; i++) {
      map[i] = mDashSurfaceData.globalIdsOfLocalNodes[i];
    }
  }

  void DashSurfaceMesh::node_map(MapVector &map) const
  {
    int size = node_count_proc();
    map.resize(size);

    for (int i = 0; i < size; i++) {
      map[i] = mDashSurfaceData.globalIdsOfLocalNodes[i];
    }
  }

  void DashSurfaceMesh::element_map(int64_t block_number, Ioss::IntVector &map) const
  {
    size_t numElementsInSurface1 = element_count_proc(1);
    size_t numElementsInSurface2 = element_count_proc(2);
    switch (block_number) {
    case 1:
      for (size_t i = 0; i < numElementsInSurface1; ++i) {
        map[i] = mDashSurfaceData.globalIdsOfLocalElements[i];
      }
      return;
    case 2:
      for (size_t i = 0; i < numElementsInSurface2; ++i) {
        map[numElementsInSurface1 + i] =
            mDashSurfaceData.globalIdsOfLocalElements[numElementsInSurface1 + i];
      }
      return;
    default: throw std::exception();
    }
  }

  void DashSurfaceMesh::element_map(int64_t block_number, MapVector &map) const
  {
    size_t numElementsInSurface1 = element_count_proc(1);
    size_t numElementsInSurface2 = element_count_proc(2);
    switch (block_number) {
    case 1:
      for (size_t i = 0; i < numElementsInSurface1; ++i) {
        map[i] = mDashSurfaceData.globalIdsOfLocalElements[i];
      }
      return;
    case 2:
      for (size_t i = 0; i < numElementsInSurface2; ++i) {
        map[numElementsInSurface1 + i] =
            mDashSurfaceData.globalIdsOfLocalElements[numElementsInSurface1 + i];
      }
      return;
    default: throw std::exception();
    }
  }

  void DashSurfaceMesh::element_map(MapVector &map) const
  {
    size_t count = element_count_proc();
    map.resize(count);

    for (size_t i = 0; i < count; i++) {
      map[i] = mDashSurfaceData.globalIdsOfLocalElements[i];
    }
  }

  void DashSurfaceMesh::element_map(Ioss::IntVector &map) const
  {
    size_t count = element_count_proc();
    map.resize(count);

    for (size_t i = 0; i < count; i++) {
      map[i] = mDashSurfaceData.globalIdsOfLocalElements[i];
    }
  }

  // -----------------------------------------------------------------------------------------

  ExodusMesh::ExodusMesh(const ExodusData &exodusData) : mExodusData(exodusData)
  {
    if (block_count() > 0) {
      mElementOffsetForBlock.resize(block_count());
      mElementOffsetForBlock[0] = 0;
      for (size_t i = 1; i < mExodusData.localNumberOfElementsInBlock.size(); i++) {
        mElementOffsetForBlock[i] =
            mElementOffsetForBlock[i - 1] + mExodusData.localNumberOfElementsInBlock[i - 1];
      }
    }

    mGlobalNumberOfElements = 0;
    for (auto &elem : mExodusData.globalNumberOfElementsInBlock) {
      mGlobalNumberOfElements += elem;
    }

    mLocalNumberOfElements = 0;
    for (auto &elem : mExodusData.localNumberOfElementsInBlock) {
      mLocalNumberOfElements += elem;
    }
  }

  int64_t ExodusMesh::node_count() const { return mExodusData.globalNumberOfNodes; }

  int64_t ExodusMesh::node_count_proc() const
  {
    return mExodusData.coordinates.size() / SPATIAL_DIMENSION;
  }

  int64_t ExodusMesh::element_count() const { return mGlobalNumberOfElements; }

  int64_t ExodusMesh::element_count(int64_t blockNumber) const
  {
    return mExodusData.globalNumberOfElementsInBlock[blockNumber - 1];
  }

  int64_t ExodusMesh::block_count() const
  {
    return mExodusData.globalNumberOfElementsInBlock.size();
  }

  int64_t ExodusMesh::nodeset_count() const { return 0; }

  int64_t ExodusMesh::sideset_count() const { return mExodusData.sidesetConnectivity.size(); }

  int64_t ExodusMesh::element_count_proc() const { return mLocalNumberOfElements; }

  int64_t ExodusMesh::element_count_proc(int64_t blockNumber) const
  {
    return mExodusData.localNumberOfElementsInBlock[blockNumber - 1];
  }

  int64_t ExodusMesh::nodeset_node_count_proc(int64_t /*id*/) const { return 0; }

  int64_t ExodusMesh::sideset_side_count_proc(int64_t /*id*/) const { return 0; }

  int64_t ExodusMesh::communication_node_count_proc() const
  {
    if (mExodusData.sharedNodes != nullptr) {
      return mExodusData.sharedNodes->size();
    }

    return 0;
  }

  void ExodusMesh::coordinates(double *coord) const
  {
    std::copy(mExodusData.coordinates.begin(), mExodusData.coordinates.end(), coord);
  }

  void ExodusMesh::coordinates(std::vector<double> & /*coord*/) const { throw std::exception(); }

  void ExodusMesh::coordinates(int /*component*/, std::vector<double> & /*xyz*/) const
  {
    throw std::exception();
  }

  void ExodusMesh::coordinates(std::vector<double> & /*x*/, std::vector<double> & /*y*/,
                               std::vector<double> & /*z*/) const
  {
    throw std::exception();
  }

  void ExodusMesh::connectivity(int64_t blockNumber, int *connectivityForBlock) const
  {
    if (mExodusData.localNumberOfElementsInBlock[blockNumber - 1] > 0) {
      std::copy(mExodusData.elementBlockConnectivity[blockNumber - 1].begin(),
                mExodusData.elementBlockConnectivity[blockNumber - 1].end(), connectivityForBlock);
    }
  }

  std::pair<std::string, int> ExodusMesh::topology_type(int64_t blockNumber) const
  {
    Topology topology = mExodusData.blockTopologicalData[blockNumber - 1];
    return std::make_pair(getTopologyName(topology), static_cast<int>(topology));
  }

  void ExodusMesh::sideset_elem_sides(int64_t setId, Ioss::Int64Vector &elem_sides) const
  {
    elem_sides.resize(mExodusData.sidesetConnectivity[setId - 1].size());
    elem_sides.insert(elem_sides.begin(), mExodusData.sidesetConnectivity[setId - 1].begin(),
                      mExodusData.sidesetConnectivity[setId - 1].end());
  }

  std::vector<std::string> ExodusMesh::sideset_touching_blocks(int64_t setId) const
  {
    return mExodusData.sidesetTouchingBlocks[setId - 1];
  }

  void ExodusMesh::nodeset_nodes(int64_t /*nset_id*/, Ioss::Int64Vector & /*nodes*/) const {}

  void ExodusMesh::node_communication_map(MapVector &map, std::vector<int> &proc)
  {
    if (mExodusData.sharedNodes == nullptr) {
      return;
    }

    for (unsigned int i = 0; i < mExodusData.sharedNodes->size(); i++) {
      map[i]  = (*mExodusData.sharedNodes)[i].nodeId;
      proc[i] = (*mExodusData.sharedNodes)[i].procId;
    }
  }

  void ExodusMesh::node_map(Ioss::IntVector &map) const
  {
    int size = node_count_proc();
    map.resize(size);

    for (int i = 0; i < size; i++) {
      map[i] = mExodusData.globalIdsOfLocalNodes[i];
    }
  }

  void ExodusMesh::node_map(MapVector &map) const
  {
    int size = node_count_proc();
    map.resize(size);

    for (int i = 0; i < size; i++) {
      map[i] = mExodusData.globalIdsOfLocalNodes[i];
    }
  }

  void ExodusMesh::element_map(int64_t blockNumber, Ioss::IntVector &map) const
  {
    int64_t offset = mElementOffsetForBlock[blockNumber - 1];
    for (int64_t i = 0; i < mExodusData.localNumberOfElementsInBlock[blockNumber - 1]; i++) {
      map[offset + i] = mExodusData.globalIdsOfLocalElements[offset + i];
    }
  }

  void ExodusMesh::element_map(int64_t blockNumber, MapVector &map) const
  {
    int64_t offset = mElementOffsetForBlock[blockNumber - 1];
    for (int64_t i = 0; i < mExodusData.localNumberOfElementsInBlock[blockNumber - 1]; i++) {
      map[offset + i] = mExodusData.globalIdsOfLocalElements[offset + i];
    }
  }

  void ExodusMesh::element_map(MapVector &map) const
  {
    int64_t count = element_count_proc();
    map.resize(count);

    for (int64_t i = 0; i < count; i++) {
      map[i] = mExodusData.globalIdsOfLocalElements[i];
    }
  }

  void ExodusMesh::element_map(Ioss::IntVector &map) const
  {
    int64_t count = element_count_proc();
    map.resize(count);

    for (int64_t i = 0; i < count; i++) {
      map[i] = mExodusData.globalIdsOfLocalElements[i];
    }
  }

} // namespace Iogn
