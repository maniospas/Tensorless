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
#include "raw/int4.h"
#include "raw/float3.h"
#include "raw/float4.h"
#include "raw/float5.h"
#include "raw/float6.h"
#include "raw/float7.h"
#include "raw/float8.h"
#include "signed.h"
#include "dynamic.h"
#include "floating.h"

namespace tensorless {
    typedef Signed<Int2> int3;
    typedef Signed<Int3> int4;
    typedef Signed<Int4> int5;
    
    typedef Signed<Float3> sfloat4;
    typedef Signed<Float4> sfloat5;
    typedef Signed<Float5> sfloat6;
    typedef Signed<Float6> sfloat7;
    typedef Signed<Float7> sfloat8;
    typedef Signed<Float8> sfloat9;

    typedef Dynamic<sfloat4> dfloat5;
    typedef Dynamic<sfloat5> dfloat6;
    typedef Dynamic<sfloat6> dfloat7;
    typedef Dynamic<sfloat7> dfloat8;
    typedef Dynamic<sfloat8> dfloat9;
    typedef Dynamic<sfloat9> dfloat10;
    
    typedef Floating<sfloat5, int3> float8; 
    typedef Floating<sfloat5, int4> float9; 
    typedef Floating<sfloat6, int4> float10;
    typedef Floating<sfloat7, int4> float11; 
    typedef Floating<sfloat8, int4> float12; 

    // experimental but slower datatypes
    typedef Floating<sfloat8, int5> float13;  
    typedef Floating<sfloat9, int5> float14;  
    typedef Floating<sfloat9, int5> float15; 
}

#endif  // TENSORLESS_TYPES_H
