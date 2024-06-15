/*
Copyright 2024 Emmanouil Krasanakis

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef TENSORLESS_TYPES_H
#define TENSORLESS_TYPES_H

#include "vecutils.h"
#include "raw/int2.h"
#include "raw/int3.h"
#include "raw/float3.h"
#include "raw/float4.h"
#include "raw/float5.h"
#include "raw/float8.h"
#include "signed.h"

namespace tensorless {
    typedef Signed<Int2> SInt2;
    typedef Signed<Int3> SInt3;
    typedef Signed<Float3> SFloat3;
    typedef Signed<Float4> SFloat4;
    typedef Signed<Float5> SFloat5;
    typedef Signed<Float8> SFloat8;
}

#endif  // TENSORLESS_TYPES_H
