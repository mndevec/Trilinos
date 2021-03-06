// Copyright(C) 1999-2010
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

#include <Ioss_CodeTypes.h>
#include <Ioss_Utils.h>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <heartbeat/Iohb_DatabaseIO.h>
#include <heartbeat/Iohb_Layout.h>
#include <iostream>
#include <string>
#include <sys/select.h>
#include <time.h>
#include <vector>

#include "Ioss_DBUsage.h"
#include "Ioss_DatabaseIO.h"
#include "Ioss_EntityType.h"
#include "Ioss_Field.h"
#include "Ioss_FileInfo.h"
#include "Ioss_IOFactory.h"
#include "Ioss_ParallelUtils.h"
#include "Ioss_Property.h"
#include "Ioss_Region.h"
#include "Ioss_State.h"
#include "Ioss_Utils.h"
#include "Ioss_VariableType.h"

namespace Ioss {
  class CommSet;
  class EdgeBlock;
  class EdgeSet;
  class ElementBlock;
  class ElementSet;
  class FaceBlock;
  class FaceSet;
  class NodeBlock;
  class NodeSet;
  class SideBlock;
  class SideSet;
} // namespace Ioss

namespace {
  std::string time_stamp(const std::string &format)
  {
    if (format == "") {
      return std::string("");
    }
    const int   length = 256;
    static char time_string[length];

    time_t     calendar_time = time(nullptr);
    struct tm *local_time    = localtime(&calendar_time);

    size_t error = strftime(time_string, length, format.c_str(), local_time);
    if (error != 0) {
      time_string[length - 1] = '\0';
      return std::string(time_string);
    }
    return std::string("[ERROR]");
  }

  std::ostream *open_stream(const std::string &filename, bool *needs_delete, bool append_file)
  {
    // A little wierdness and ambiguity is possible here.  We want to
    // minimize the number of commands, but maximize the
    // functionality. For example, we want to be able to specify output
    // to existing streams (cout/stdout, cerr/stderr, outputP0), files,
    // sockets, XML-RPC?, SOAP?.  However, we want to be able to specify
    // this with a single command.
    // So..., we first check for some 'reserved' stream names.  These
    // are the 'cout, stdout, cerr, stderr, output, outputP0' type.
    // Note that this means the user can't specify a file name of
    // 'cerr', but that shouldn't be too much of a hardship.  [If it is,
    // we can devise a syntax as: 'FILE:cout' or do a './cout'

    std::ostream *log_stream = nullptr;
    *needs_delete            = false;
    if (filename == "cout" || filename == "stdout") {
      log_stream = &std::cout;
    }
    else if (filename == "cerr" || filename == "stderr") {
      log_stream = &std::cerr; // This is also default if nothing specified
    }
    else if (filename == "output" || filename == "outputP0") {
      log_stream = &std::cout; // This should be sierra::Env::outputP0(), but not
                               // available during transition to stk...
    }
    else if (filename == "clog" || filename == "log") {
      log_stream = &std::clog; // Same as cerr, but not flushed automatically.
    }
    else {
      // Open the file (on processor 0 only) Might need to do
      // something better here if we want to share streams among
      // different heartbeats or logging mechanisms.  Need perhaps a
      // 'logger' class which handles sharing and destruction...
      std::ofstream *tmp = nullptr;
      if (append_file) {
        tmp = new std::ofstream(filename.c_str(), std::ios::out | std::ios::app);
      }
      else {
        tmp = new std::ofstream(filename.c_str());
      }
      if (tmp != nullptr && !tmp->is_open()) {
        delete tmp;
      }
      else {
        log_stream    = tmp;
        *needs_delete = true;
      }
    }
    return log_stream;
  }
} // namespace

namespace Iohb {

  // ========================================================================
  const IOFactory *IOFactory::factory()
  {
    static IOFactory registerThis;
    return &registerThis;
  }

  IOFactory::IOFactory() : Ioss::IOFactory("heartbeat") {}

  Ioss::DatabaseIO *IOFactory::make_IO(const std::string &filename, Ioss::DatabaseUsage db_usage,
                                       MPI_Comm                     communicator,
                                       const Ioss::PropertyManager &props) const
  {
    return new DatabaseIO(nullptr, filename, db_usage, communicator, props);
  }

  // ========================================================================
  DatabaseIO::DatabaseIO(Ioss::Region *region, const std::string &filename,
                         Ioss::DatabaseUsage db_usage, MPI_Comm communicator,
                         const Ioss::PropertyManager &props)
      : Ioss::DatabaseIO(region, filename, db_usage, communicator, props), timeLastFlush_(0),
	flushInterval_(10), logStream(nullptr), layout_(nullptr), legend_(nullptr),
	tsFormat("[%H:%M:%S]"), separator_(", "), precision_(5), fieldWidth_(0),
	showLabels(false), showLegend(true), appendOutput(false), addTimeField(false),
	initialized_(false), streamNeedsDelete(false), fileFormat(DEFAULT)
  {
    timeLastFlush_ = time(nullptr);
    dbState       = Ioss::STATE_UNKNOWN;
  }

  DatabaseIO::~DatabaseIO()
  {
    delete layout_;
    delete legend_;
    if (streamNeedsDelete && (logStream != nullptr)) {
      delete logStream;
    }
  }

  void DatabaseIO::initialize(const Ioss::Region * /*region*/) const
  {
    if (!initialized_) {
      assert(layout_ == nullptr);
      assert(legend_ == nullptr);

      DatabaseIO *new_this = const_cast<DatabaseIO *>(this);

      if (properties.exists("FIELD_SEPARATOR")) {
        new_this->separator_ = properties.get("FIELD_SEPARATOR").get_string();
      }

      if (properties.exists("FILE_FORMAT")) {
        std::string format = properties.get("FILE_FORMAT").get_string();
        if (Ioss::Utils::case_strcmp(format, "spyhis") == 0) {
          new_this->fileFormat = SPYHIS;
        }
      }

      bool append = open_create_behavior() == Ioss::DB_APPEND;

      // Try to open file...
      new_this->logStream = nullptr;
      if (util().parallel_rank() == 0) {
        new_this->logStream = open_stream(get_filename(), &(new_this->streamNeedsDelete), append);

        if (new_this->logStream == nullptr) {
          std::ostringstream errmsg;
          errmsg << "ERROR: Could not create heartbeat file '" << get_filename() << "'\n";
          IOSS_ERROR(errmsg);
        }
      }

      // Pull variables from the regions property data...
      if (properties.exists("FLUSH_INTERVAL")) {
	new_this->flushInterval_ = properties.get("FLUSH_INTERVAL").get_int();
      }
      
      if (properties.exists("TIME_STAMP_FORMAT")) {
        new_this->tsFormat = properties.get("TIME_STAMP_FORMAT").get_string();
      }

      if (properties.exists("SHOW_TIME_STAMP")) {
        bool show_time_stamp = properties.get("SHOW_TIME_STAMP").get_int() == 1;
        if (!show_time_stamp) {
          new_this->tsFormat = "";
        }
      }

      if (properties.exists("PRECISION")) {
        new_this->precision_ = properties.get("PRECISION").get_int();
      }

      if (properties.exists("FIELD_WIDTH")) {
        new_this->fieldWidth_ = properties.get("FIELD_WIDTH").get_int();
      }
      else {
        // +1.xxxxxxe+00 The x count is the precision the "+1.e+00" is the 7
        new_this->fieldWidth_ = precision_ + 7;
      }

      if (properties.exists("SHOW_LABELS")) {
        new_this->showLabels = (properties.get("SHOW_LABELS").get_int() == 1);
      }

      if (properties.exists("SHOW_LEGEND")) {
        new_this->showLegend =
            (properties.get("SHOW_LEGEND").get_int() == 1 && !new_this->appendOutput);
      }

      if (properties.exists("SHOW_TIME_FIELD")) {
        new_this->addTimeField = (properties.get("SHOW_TIME_FIELD").get_int() == 1);
      }

      if (fileFormat == SPYHIS) {
        new_this->addTimeField = true;
        new_this->showLegend   = true;
        new_this->showLabels   = false;
        new_this->tsFormat     = "";
      }

      if (showLegend) {
        new_this->legend_ = new Layout(false, precision_, separator_, fieldWidth_);
        if (!tsFormat.empty()) {
          new_this->legend_->add_literal("+");
          new_this->legend_->add_literal(time_stamp(tsFormat));
          new_this->legend_->add_literal(" ");
        }

        if (addTimeField) {
          if (fileFormat == SPYHIS) {
            new_this->legend_->add_legend("TIME");
          }
          else {
            new_this->legend_->add_legend("Time");
          }
        }
      }
      new_this->initialized_ = true;
    }
  }

  bool DatabaseIO::begin(Ioss::State /* state */) { return true; }

  bool DatabaseIO::end(Ioss::State /* state */) { return true; }

  bool DatabaseIO::begin_state(Ioss::Region *region, int /* state */, double time)
  {
    // If this is the first time, open the output stream and see if user wants a legend
    initialize(region);

    layout_ = new Layout(showLabels, precision_, separator_, fieldWidth_);
    if (tsFormat != "") {
      layout_->add_literal("+");
      layout_->add_literal(time_stamp(tsFormat));
      layout_->add_literal(" ");
    }

    if (addTimeField) {
      layout_->add("TIME", time / timeScaleFactor);
    }

    return true;
  }

  void DatabaseIO::flush_database() const
  {
    logStream->flush();
  }

  bool DatabaseIO::end_state(Ioss::Region * /* region */, int /* state */, double /* time */)
  {
    if (legend_ != nullptr) {
      if (fileFormat == SPYHIS) {
        time_t calendar_time = time(nullptr);
        *logStream << "% Sierra SPYHIS Output " << ctime(&calendar_time);
        *logStream << *legend_ << '\n'; // Legend output twice for SPYHIS
      }

      *logStream << *legend_ << '\n';
      delete legend_;
      legend_ = nullptr;
    }

    *logStream << *layout_ << '\n';
    delete layout_;
    layout_ = nullptr;

    // Flush the buffer to disk...
    // flush if there is more than 'flushInterval_' seconds since the last flush to avoid
    // the flush eating up cpu time for small fast jobs...

    // This code is derived from code in finalize_write() in Ioex_DatabaseIO.C
    // See other comments there...

    time_t cur_time = time(nullptr);
    if (cur_time - timeLastFlush_ >= flushInterval_) {
      timeLastFlush_ = cur_time;
      flush_database();
    }

    return true;
  }

  int64_t DatabaseIO::get_field_internal(const Ioss::Region * /* reg */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }

  int64_t DatabaseIO::get_field_internal(const Ioss::NodeBlock * /* nb */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::get_field_internal(const Ioss::EdgeBlock * /* nb */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::get_field_internal(const Ioss::FaceBlock * /* nb */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::get_field_internal(const Ioss::ElementBlock * /* eb */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }

  int64_t DatabaseIO::get_field_internal(const Ioss::NodeSet * /* ns */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::get_field_internal(const Ioss::EdgeSet * /* ns */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::get_field_internal(const Ioss::FaceSet * /* ns */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::get_field_internal(const Ioss::ElementSet * /* ns */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::get_field_internal(const Ioss::SideBlock * /* eb */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::get_field_internal(const Ioss::SideSet * /* fs */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::get_field_internal(const Ioss::CommSet * /* cs */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }

  int64_t DatabaseIO::put_field_internal(const Ioss::Region *region, const Ioss::Field &field,
                                         void *data, size_t data_size) const
  {
    initialize(region);
    Ioss::Field::RoleType role       = field.get_role();
    int64_t               num_to_get = field.verify(data_size);

    if ((role == Ioss::Field::TRANSIENT || role == Ioss::Field::REDUCTION) && num_to_get == 1) {

      int ncomp = field.transformed_storage()->component_count();

      if (legend_ != nullptr && layout_ != nullptr) {
        if (ncomp == 1) {
          legend_->add_legend(field.get_name());
        }
        else {
          const Ioss::VariableType *var_type = field.transformed_storage();
          for (int i = 0; i < ncomp; i++) {
            std::string var_name = var_type->label_name(field.get_name(), i + 1, '_');
            legend_->add_legend(var_name);
          }
        }
      }

      if (field.get_type() == Ioss::Field::STRING) {
        // Assume that if layout_ is nullptr, then we want special one-line output.
        if (layout_ == nullptr) {
          Layout layout(false, 0, separator_, fieldWidth_);
          layout.add_literal("-");
          layout.add_literal(time_stamp(tsFormat));
          layout.add_literal(" ");
          layout.add_literal(*reinterpret_cast<std::string *>(data));
          if (logStream != nullptr) {
            *logStream << layout << '\n';
          }
        }
        else {
          layout_->add(field.get_name(), *reinterpret_cast<std::string *>(data));
        }
      }
      else {
        if (layout_ == nullptr) {
          std::ostringstream errmsg;
          errmsg << "INTERNAL ERROR: Unexpected nullptr layout.\n";
          IOSS_ERROR(errmsg);
        }
        if (field.get_type() == Ioss::Field::INTEGER) {
          assert(field.transformed_count() == 1);

          int *            i_data = reinterpret_cast<int *>(data);
          std::vector<int> idata(ncomp);
          for (int i = 0; i < ncomp; i++) {
            idata[i] = i_data[i];
          }
          layout_->add(field.get_name(), idata);
        }
        else {
          std::vector<double> rdata(ncomp);
          double *            r_data = reinterpret_cast<double *>(data);
          for (int i = 0; i < ncomp; i++) {
            rdata[i] = r_data[i];
          }
          layout_->add(field.get_name(), rdata);
        }
      }
    }
    else {
      std::ostringstream errmsg;
      errmsg << "ERROR: Can not handle non-TRANSIENT or non-REDUCTION fields on regions.\n";
      IOSS_ERROR(errmsg);
    }
    return num_to_get;
  }

  int64_t DatabaseIO::put_field_internal(const Ioss::ElementBlock * /* eb */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::put_field_internal(const Ioss::FaceBlock * /* nb */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::put_field_internal(const Ioss::EdgeBlock * /* nb */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::put_field_internal(const Ioss::NodeBlock * /* nb */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }

  int64_t DatabaseIO::put_field_internal(const Ioss::NodeSet * /* ns */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::put_field_internal(const Ioss::EdgeSet * /* ns */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::put_field_internal(const Ioss::FaceSet * /* ns */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::put_field_internal(const Ioss::ElementSet * /* ns */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::put_field_internal(const Ioss::SideBlock * /* fb */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::put_field_internal(const Ioss::SideSet * /* fs */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }
  int64_t DatabaseIO::put_field_internal(const Ioss::CommSet * /* cs */,
                                         const Ioss::Field & /* field */, void * /* data */,
                                         size_t /* data_size */) const
  {
    return -1;
  }

  unsigned DatabaseIO::entity_field_support() const { return Ioss::REGION; }
} // namespace Iohb
