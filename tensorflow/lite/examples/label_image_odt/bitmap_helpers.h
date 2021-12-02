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

#ifndef TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_BITMAP_HELPERS_H_
#define TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_BITMAP_HELPERS_H_

#include "tensorflow/lite/examples/label_image_odt/bitmap_helpers_impl.h"
#include "tensorflow/lite/examples/label_image_odt/label_image.h"
#include "tensorflow/lite/examples/label_image_odt/log.h"

#include <algorithm>

namespace tflite {
namespace label_image_odt {

std::vector<uint8_t> read_bmp(const std::string& input_bmp_name, int* width,
                              int* height, int* channels, Settings* s);

template <class T>
void resize(T* out, uint8_t* in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels, Settings* s);

// explicit instantiation
template void resize<float>(float*, unsigned char*, int, int, int, int, int,
                            int, Settings*);
template void resize<int8_t>(int8_t*, unsigned char*, int, int, int, int, int,
                             int, Settings*);
template void resize<uint8_t>(uint8_t*, unsigned char*, int, int, int, int, int,
                              int, Settings*);
template <class T>
std::vector<T> prepare(std::vector<uint8_t> in, Settings* s) {
    std::vector<T> out;
    out.reserve(in.size());
    for (auto & p : in) {
          out.push_back((static_cast<T>(p) - s->input_mean) / s->input_std);
    }
    return out;
}

template <class T>
std::vector<T> to_categorical(int cat, int num_cat) {
    std::vector<T> out(num_cat, static_cast<T>(0));
    out[cat] = static_cast<T>(1);
    return out;
}

}  // namespace label_image
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_BITMAP_HELPERS_H_
