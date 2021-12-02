/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/examples/label_image_odt/label_image.h"

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <experimental/filesystem>

#include "absl/memory/memory.h"
#include "tensorflow/lite/examples/label_image_odt/bitmap_helpers.h"
#include "tensorflow/lite/examples/label_image_odt/get_top_n.h"
#include "tensorflow/lite/examples/label_image_odt/log.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/signature_runner.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

namespace tflite {
namespace label_image_odt {

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using ProvidedDelegateList = tflite::tools::ProvidedDelegateList;

class DelegateProviders {
 public:
  DelegateProviders() : delegate_list_util_(&params_) {
    delegate_list_util_.AddAllDelegateParams();
  }

  // Initialize delegate-related parameters from parsing command line arguments,
  // and remove the matching arguments from (*argc, argv). Returns true if all
  // recognized arg values are parsed correctly.
  bool InitFromCmdlineArgs(int* argc, const char** argv) {
    std::vector<tflite::Flag> flags;
    delegate_list_util_.AppendCmdlineFlags(flags);

    const bool parse_result = Flags::Parse(argc, argv, flags);
    if (!parse_result) {
      std::string usage = Flags::Usage(argv[0], flags);
      LOG(ERROR) << usage;
    }
    return parse_result;
  }

  // According to passed-in settings `s`, this function sets corresponding
  // parameters that are defined by various delegate execution providers. See
  // lite/tools/delegates/README.md for the full list of parameters defined.
  void MergeSettingsIntoParams(const Settings& s) {
    // Parse settings related to GPU delegate.
    // Note that GPU delegate does support OpenCL. 'gl_backend' was introduced
    // when the GPU delegate only supports OpenGL. Therefore, we consider
    // setting 'gl_backend' to true means using the GPU delegate.
    if (s.gl_backend) {
      if (!params_.HasParam("use_gpu")) {
        LOG(WARN) << "GPU deleate execution provider isn't linked or GPU "
                     "delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_gpu", true);
        // The parameter "gpu_inference_for_sustained_speed" isn't available for
        // iOS devices.
        if (params_.HasParam("gpu_inference_for_sustained_speed")) {
          params_.Set<bool>("gpu_inference_for_sustained_speed", true);
        }
        params_.Set<bool>("gpu_precision_loss_allowed", s.allow_fp16);
      }
    }

    // Parse settings related to NNAPI delegate.
    if (s.accel) {
      if (!params_.HasParam("use_nnapi")) {
        LOG(WARN) << "NNAPI deleate execution provider isn't linked or NNAPI "
                     "delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_nnapi", true);
        params_.Set<bool>("nnapi_allow_fp16", s.allow_fp16);
      }
    }

    // Parse settings related to Hexagon delegate.
    if (s.hexagon_delegate) {
      if (!params_.HasParam("use_hexagon")) {
        LOG(WARN) << "Hexagon deleate execution provider isn't linked or "
                     "Hexagon delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_hexagon", true);
        params_.Set<bool>("hexagon_profiling", s.profiling);
      }
    }

    // Parse settings related to XNNPACK delegate.
    if (s.xnnpack_delegate) {
      if (!params_.HasParam("use_xnnpack")) {
        LOG(WARN) << "XNNPACK deleate execution provider isn't linked or "
                     "XNNPACK delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_xnnpack", true);
        params_.Set<bool>("num_threads", s.number_of_threads);
      }
    }
  }

  // Create a list of TfLite delegates based on what have been initialized (i.e.
  // 'params_').
  std::vector<ProvidedDelegateList::ProvidedDelegate> CreateAllDelegates()
      const {
    return delegate_list_util_.CreateAllRankedDelegates();
  }

 private:
  // Contain delegate-related parameters that are initialized from command-line
  // flags.
  tflite::tools::ToolParams params_;

  // A helper to create TfLite delegates.
  ProvidedDelegateList delegate_list_util_;
};

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
TfLiteStatus ReadLabelsFile(const string& file_name,
                            std::vector<string>* result,
                            size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    LOG(ERROR) << "Labels file " << file_name << " not found";
    return kTfLiteError;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}

typedef struct{
    std::vector<uint8_t>  image;
    int label;
    int width;
    int height;
    int channels;
} DataT;
typedef std::map<std::string, DataT> DatasetT;

DatasetT load_dataset(std::string path, Settings* settings) {
    DatasetT dataset;
#if 1
    for(auto & dir : std::experimental::filesystem::directory_iterator(path) ) {
        if(!std::experimental::filesystem::is_directory(dir)) {
            continue;
        }
        int label = strtol(dir.path().filename().c_str(), nullptr, 10);
        for (auto & image : std::experimental::filesystem::directory_iterator(dir)) {
            if (image.path().extension() != ".bmp" )  continue;
            int width, height, channels;
            std::vector<uint8_t> image_data = read_bmp(image.path().c_str(), &width, &height, &channels, settings);
            dataset.emplace(image.path().filename().c_str(), DataT({image_data, label, width, height, channels}));
        }
    }
#endif
    return dataset;
}

void test_tflite_model(std::unique_ptr<tflite::Interpreter>* interpreter,
                       DatasetT& dataset,
                       Settings* settings) {

    int count = 0;
    int input = (*interpreter)->inputs()[0];

    for(auto it = dataset.begin(); it != dataset.end(); it++) {
        std::vector<float> image= prepare<float>(it->second.image, settings);
        memcpy((*interpreter)->typed_tensor<float>(input), image.data(), sizeof(float)*image.size());

        if ((*interpreter)->Invoke() != kTfLiteOk) {
            LOG(ERROR) << "Failed to invoke tflite!";
            exit(-1);
        }

        const float threshold = 0.0f;
        const int number_of_results = 1;
        std::vector<std::pair<float, int>> top_results;

        // assume output dims to be something like (1, 1, ... ,size)
        int output = (*interpreter)->outputs()[0];
        TfLiteIntArray* output_dims = (*interpreter)->tensor(output)->dims;
        auto output_size = output_dims->data[output_dims->size - 1];
        switch ((*interpreter)->tensor(output)->type) {
            case kTfLiteFloat32:
              get_top_n<float>((*interpreter)->typed_output_tensor<float>(0), output_size,
                               number_of_results, threshold, &top_results,
                               settings->input_type);
              break;
            case kTfLiteInt8:
              get_top_n<int8_t>((*interpreter)->typed_output_tensor<int8_t>(0),
                                output_size, number_of_results, threshold,
                                &top_results, settings->input_type);
              break;
            case kTfLiteUInt8:
              get_top_n<uint8_t>((*interpreter)->typed_output_tensor<uint8_t>(0),
                                 output_size,number_of_results, threshold,
                                 &top_results, settings->input_type);
              break;
            default:
              LOG(ERROR) << "cannot handle output type "
                         << (*interpreter)->tensor(output)->type << " yet";
              exit(-1);
        }

        if(it->second.label == top_results[0].second ) {
            count++;
        }
    }

    LOG(INFO) << "Accuracy is: " << static_cast<double>(count)/static_cast<double>(dataset.size())
                                 << " (" << count << ", " << dataset.size() << ")";
}

void PrintProfilingInfo(const profiling::ProfileEvent* e,
                        uint32_t subgraph_index, uint32_t op_index,
                        TfLiteRegistration registration) {
  // output something like
  // time (ms) , Node xxx, OpCode xxx, symbolic name
  //      5.352, Node   5, OpCode   4, DEPTHWISE_CONV_2D

  LOG(INFO) << std::fixed << std::setw(10) << std::setprecision(3)
            << (e->end_timestamp_us - e->begin_timestamp_us) / 1000.0
            << ", Subgraph " << std::setw(3) << std::setprecision(3)
            << subgraph_index << ", Node " << std::setw(3)
            << std::setprecision(3) << op_index << ", OpCode " << std::setw(3)
            << std::setprecision(3) << registration.builtin_code << ", "
            << EnumNameBuiltinOperator(
                   static_cast<BuiltinOperator>(registration.builtin_code));
}

void PrintTfLiteTensor(const TfLiteTensor* t, std::string name = "") {
    LOG(INFO) << "Tensor Name: " << name ;
}

void RunInference(Settings* settings,
                  const DelegateProviders& delegate_providers) {
  if (!settings->model_name.c_str()) {
    LOG(ERROR) << "no model file name";
    exit(-1);
  }

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  model = tflite::FlatBufferModel::BuildFromFile(settings->model_name.c_str());
  if (!model) {
    LOG(ERROR) << "Failed to mmap model " << settings->model_name;
    exit(-1);
  }
  settings->model = model.get();
  LOG(INFO) << "Loaded model " << settings->model_name;
  model->error_reporter();
  LOG(INFO) << "resolved reporter";

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(ERROR) << "Failed to construct interpreter";
    exit(-1);
  }

  interpreter->SetAllowFp16PrecisionForFp32(settings->allow_fp16);

  if (settings->verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size();
    LOG(INFO) << "nodes size: " << interpreter->nodes_size();
    LOG(INFO) << "inputs: " << interpreter->inputs().size();
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0);

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point;
    }
  }

  if (settings->number_of_threads != -1) {
    interpreter->SetNumThreads(settings->number_of_threads);
  }

//  int input = interpreter->inputs()[0];

  auto delegates = delegate_providers.CreateAllDelegates();
  for (auto& delegate : delegates) {
    const auto delegate_name = delegate.provider->GetName();
    if (interpreter->ModifyGraphWithDelegate(std::move(delegate.delegate)) !=
        kTfLiteOk) {
      LOG(ERROR) << "Failed to apply " << delegate_name << " delegate.";
      exit(-1);
    } else {
      LOG(INFO) << "Applied " << delegate_name << " delegate.";
    }
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "Failed to allocate tensors!";
    exit(-1);
  }

  //Load Testing dataset:
  LOG(INFO) << "Loading test dataset from :" << settings->test_dataset;
  DatasetT datasetTest = load_dataset(settings->test_dataset, settings);
  LOG(INFO) << "Test dataset size: " << datasetTest.size();

  //Test the model initially
  LOG(INFO) << "Testing model accuracy before training";
  test_tflite_model(&interpreter, datasetTest, settings);

  // Get signatures from the interpretter:
  LOG(INFO) << "Model contains following signatures:" ;
  auto signatures = interpreter->signature_keys();
  for(auto& s : signatures) {
      LOG(INFO) << "\t" << *s;
  }

  auto trainSig = interpreter->GetSignatureRunner("train");
  if (trainSig == nullptr) {
      LOG(ERROR) << "Model not contains train signature";
      exit(1);
  }

  // Load train dataset:
  LOG(INFO) << "Loading train dataset from :" << settings->train_dataset;
  DatasetT dataset = load_dataset(settings->train_dataset, settings);
  LOG(INFO) << "Train dataset size: " << dataset.size();

  // Assuming input are named "x" and "y" and the output tensor loss is named "loss"
  const char* INPUT_X = "x";
  const char* INPUT_Y = "y";
  const char* OUTPUT_LOSS = "loss";

  int num_batches = dataset.size() / settings->batch_size;

  LOG(INFO) << "Train signagure inputs and outputs:";
  LOG(INFO) << "\tInputs:";
  auto trIn = interpreter->signature_inputs("train");
  for (auto& s : trIn) {
      LOG(INFO) << "\t\t"<<s.first << " : " << s.second;
  }
  LOG(INFO) << "\tOutputs:";
  auto trOut = interpreter->signature_outputs("train");
  for (auto& s : trOut) {
      LOG(INFO) << "\t\t" << s.first << " : " << s.second;
  }


  if(trainSig->input_tensor(INPUT_X)->dims->size != 3) {
      LOG(ERROR) << "Demo supports only models with input dimemensionality 3 (batch, height, width)";
      exit(-1);
  }
  const int IMG_HEIGHT = trainSig->input_tensor(INPUT_X)->dims->data[1];
  const int IMG_WIDTH  = trainSig->input_tensor(INPUT_X)->dims->data[2];

  if (settings->verbose) {
      LOG(INFO) << "Printing interpretter state before resize OP";
      PrintInterpreterState(interpreter.get());
  }

  // Resize the input tensor to requested batch size:
  LOG(INFO) << "Resize of model to batch size " << settings->batch_size;
  auto status1 = trainSig->ResizeInputTensor(INPUT_X, {settings->batch_size, IMG_WIDTH, IMG_HEIGHT} );
  auto status2 = trainSig->ResizeInputTensor(INPUT_Y, {settings->batch_size, 10});
  auto status3 = trainSig->AllocateTensors();
  if((status1 != kTfLiteOk) || (status2 != kTfLiteOk) || (status3 != kTfLiteOk)) {
      LOG(ERROR) << "Model resize failed.";
      exit(-1);
  }

  if (settings->verbose) {
      LOG(INFO) << "Printing interpretter state after resize OP";
      PrintInterpreterState(interpreter.get());
  }

  std::vector<std::vector<float>> trainImageBatches;
  std::vector<std::vector<float>> trainLabelBatches;

  //Prepare training batches:
  LOG(INFO) << "Preparing training batches of size " << settings->batch_size;
  auto imIt = dataset.begin();
  for (int i = 0; i < num_batches; i++) {
      std::vector<float> trainImages;
      trainImages.reserve(settings->batch_size * IMG_HEIGHT * IMG_WIDTH);
      std::vector<float> trainLabels;
      trainLabels.reserve(settings->batch_size * 10);

      for (int j = 0 ;j< settings->batch_size && imIt != dataset.end(); j++, imIt++) {
          if((imIt->second.height != IMG_HEIGHT) ||
             (imIt->second.width != IMG_WIDTH)   ||
             (imIt->second.channels != 1)) {
              LOG(ERROR) << "Image size not match model's input. Resizing not supported.";
              exit(-1);
          }
          std::vector<float> image= prepare<float>(imIt->second.image, settings);
          trainImages.insert(trainImages.end(),image.begin(),image.end());

          std::vector<float> label = to_categorical<float>(imIt->second.label, 10);
          trainLabels.insert(trainLabels.end(), label.begin(), label.end());
      }
      trainImageBatches.emplace_back(trainImages);
      trainLabelBatches.emplace_back(trainLabels);
      LOG(INFO) << "Batch " << i << " prepared";
  }

  // Get the input tensors:
  TfLiteTensor* inputX = trainSig->input_tensor(INPUT_X);
  TfLiteTensor* inputY = trainSig->input_tensor(INPUT_Y);
  const TfLiteTensor* outputLoss = trainSig->output_tensor(OUTPUT_LOSS);
  LOG(INFO) << "Dimensions of input X: ";
  for( int i = 0 ; i< inputX->dims->size; i++) {
      LOG(INFO) << "\t" << inputX->dims->data[i];
  }
  LOG(INFO) << "Dimensions of input Y:";
  for( int i = 0 ; i< inputY->dims->size; i++) {
      LOG(INFO) << "\t" << inputY->dims->data[i];
  }

  //Run the training for defined number of epochs.
  for( int epoch = 0; epoch < settings->num_epochs; epoch++) {
      for(int batchIdx = 0; batchIdx < num_batches; batchIdx++) {
          memcpy(inputX->data.f, trainImageBatches[batchIdx].data(),
                  trainImageBatches[batchIdx].size()*sizeof(float));
          memcpy(inputY->data.f, trainLabelBatches[batchIdx].data(),
                  trainLabelBatches[batchIdx].size()*sizeof(float));

          LOG(INFO) << "Training on batch " << batchIdx ;
          trainSig->Invoke();
      }
      LOG(INFO) << "Finished " << epoch << " epochs, current loss: "
              << std::fixed <<std::setprecision(40) << *GetTensorData<float>(outputLoss);

      if((epoch % 10) == 0) {
          LOG(INFO) << "Testing trained model after " << epoch << " epochs";
          test_tflite_model(&interpreter, datasetTest, settings);
      }
  }

  // Test trained model:
  LOG(INFO) << "Testing model after training";
  test_tflite_model(&interpreter, datasetTest, settings);

  // Model saving
  LOG(INFO) << "Saving trained model weights to: " << settings->save_path;
  auto saveSig = interpreter->GetSignatureRunner("save");
  if (saveSig == nullptr) {
      LOG(ERROR) << "Model not contains save signature";
      exit(1);
  }
  auto sigInputs = interpreter->signature_outputs("save");
  LOG(INFO) << "Save signature has following inputs:";
  for (auto& s : sigInputs) {
      LOG(INFO) << "\t" << s.first << " : " << s.second;
  }
  LOG(INFO) << "SaveSig output size: " << saveSig->output_size();
  auto names = saveSig->output_names();
  LOG(INFO) << "Save signature has following outputs:";
  for (auto& n: names) {
      LOG(INFO) << "\t" << n;
  }

  TfLiteTensor* saveSigInputTensor = saveSig->input_tensor(saveSig->input_names()[0]);

  if (saveSig->AllocateTensors() != kTfLiteOk) {
      LOG(ERROR) << "Failed to allocate tensors for saveSig Runner";
      exit(-1);
  }

  DynamicBuffer buf;
  buf.AddString(settings->save_path.c_str(), settings->save_path.size());
  buf.WriteToTensor(saveSigInputTensor, nullptr);

  saveSig->Invoke();
  LOG(INFO) << "Weights saved.";
}

void display_usage() {
  LOG(INFO)
      << "label_image_odt\n"
      << "--input_mean, -b: input mean\n"
      << "--input_std, -s: input standard deviation\n"
      << "--image, -i: image_name.bmp\n"
      << "--tflite_model, -m: model_name.tflite\n"
      << "--threads, -t: number of threads\n"
      << "--verbose, -v: [0|1] print more information\n"
      << "--save_path, -o: path to save the trained checkpoint\n"
      << "--xnnpack_delegate, -x [0:1]: xnnpack delegate\n";
}

int Main(int argc, char** argv) {
  DelegateProviders delegate_providers;
  bool parse_result = delegate_providers.InitFromCmdlineArgs(
      &argc, const_cast<const char**>(argv));
  if (!parse_result) {
    return EXIT_FAILURE;
  }

  Settings s;

  int c;
  while (true) {
    static struct option long_options[] = {
        {"verbose", required_argument, nullptr, 'v'},
        {"train_dataset", required_argument, nullptr, 'x'},
        {"test_dataset", required_argument, nullptr, 'y'},
        {"tflite_model", required_argument, nullptr, 'm'},
        {"num_epochs", required_argument, nullptr, 'e'},
        {"batch_size", required_argument, nullptr, 'c'},
        {"threads", required_argument, nullptr, 't'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"save_path", required_argument, nullptr, 'o'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "b:c:e:m:o:s:v:x:y", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'b':
        s.input_mean = strtod(optarg, nullptr);
        break;
      case 'c':
          s.batch_size =
              strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
          break;
      case 'e':
          s.num_epochs =
              strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
          break;
      case 'm':
        s.model_name = optarg;
        break;
      case 'o':
          s.save_path = optarg;
          break;
      case 'r':
        s.number_of_results =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      case 'v':
        s.verbose =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'w':
        s.number_of_warmup_runs =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'x':
          s.train_dataset = optarg;
          break;
      case 'y':
          s.test_dataset = optarg;
          break;
      case 'h':
      case '?':
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
      default:
        exit(-1);
    }
  }

  delegate_providers.MergeSettingsIntoParams(s);
  RunInference(&s, delegate_providers);
  return 0;
}

}  // namespace label_image
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::label_image_odt::Main(argc, argv);
}
