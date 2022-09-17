#ifndef TWIDDLE_FACT_H
#define TWIDDLE_FACT_H

DATA_LOCATION Complex_type twiddle_factors[FFT_LEN_RADIX2/2] = {
    /*
    {1.000000f, 0.000000f},
    {0.999995f, -0.003068f},
    {0.999981f, -0.006136f},
    {0.999958f, -0.009204f},
    {0.999925f, -0.012272f},
    {0.999882f, -0.015339f},
    {0.999831f, -0.018407f},
    {0.999769f, -0.021474f},
    {0.999699f, -0.024541f},
    {0.999619f, -0.027608f},
    {0.999529f, -0.030675f},
    {0.999431f, -0.033741f},
    {0.999322f, -0.036807f},
    {0.999205f, -0.039873f},
    {0.999078f, -0.042938f},
    {0.998941f, -0.046003f},
    {0.998795f, -0.049068f},
    {0.998640f, -0.052132f},
    {0.998476f, -0.055195f},
    {0.998302f, -0.058258f},
    {0.998118f, -0.061321f},
    {0.997925f, -0.064383f},
    {0.997723f, -0.067444f},
    {0.997511f, -0.070505f},
    {0.997290f, -0.073565f},
    {0.997060f, -0.076624f},
    {0.996820f, -0.079682f},
    {0.996571f, -0.082740f},
    {0.996313f, -0.085797f},
    {0.996045f, -0.088854f},
    {0.995767f, -0.091909f},
    {0.995481f, -0.094963f},
    {0.995185f, -0.098017f},
    {0.994879f, -0.101070f},
    {0.994565f, -0.104122f},
    {0.994240f, -0.107172f},
    {0.993907f, -0.110222f},
    {0.993564f, -0.113271f},
    {0.993212f, -0.116319f},
    {0.992850f, -0.119365f},
    {0.992480f, -0.122411f},
    {0.992099f, -0.125455f},
    {0.991710f, -0.128498f},
    {0.991311f, -0.131540f},
    {0.990903f, -0.134581f},
    {0.990485f, -0.137620f},
    {0.990058f, -0.140658f},
    {0.989622f, -0.143695f},
    {0.989177f, -0.146730f},
    {0.988722f, -0.149765f},
    {0.988258f, -0.152797f},
    {0.987784f, -0.155828f},
    {0.987301f, -0.158858f},
    {0.986809f, -0.161886f},
    {0.986308f, -0.164913f},
    {0.985798f, -0.167938f},
    {0.985278f, -0.170962f},
    {0.984749f, -0.173984f},
    {0.984210f, -0.177004f},
    {0.983662f, -0.180023f},
    {0.983105f, -0.183040f},
    {0.982539f, -0.186055f},
    {0.981964f, -0.189069f},
    {0.981379f, -0.192080f},
    {0.980785f, -0.195090f},
    {0.980182f, -0.198098f},
    {0.979570f, -0.201105f},
    {0.978948f, -0.204109f},
    {0.978317f, -0.207111f},
    {0.977677f, -0.210112f},
    {0.977028f, -0.213110f},
    {0.976370f, -0.216107f},
    {0.975702f, -0.219101f},
    {0.975025f, -0.222094f},
    {0.974339f, -0.225084f},
    {0.973644f, -0.228072f},
    {0.972940f, -0.231058f},
    {0.972226f, -0.234042f},
    {0.971504f, -0.237024f},
    {0.970772f, -0.240003f},
    {0.970031f, -0.242980f},
    {0.969281f, -0.245955f},
    {0.968522f, -0.248928f},
    {0.967754f, -0.251898f},
    {0.966976f, -0.254866f},
    {0.966190f, -0.257831f},
    {0.965394f, -0.260794f},
    {0.964590f, -0.263755f},
    {0.963776f, -0.266713f},
    {0.962953f, -0.269668f},
    {0.962121f, -0.272621f},
    {0.961280f, -0.275572f},
    {0.960431f, -0.278520f},
    {0.959572f, -0.281465f},
    {0.958703f, -0.284408f},
    {0.957826f, -0.287347f},
    {0.956940f, -0.290285f},
    {0.956045f, -0.293219f},
    {0.955141f, -0.296151f},
    {0.954228f, -0.299080f},
    {0.953306f, -0.302006f},
    {0.952375f, -0.304929f},
    {0.951435f, -0.307850f},
    {0.950486f, -0.310767f},
    {0.949528f, -0.313682f},
    {0.948561f, -0.316593f},
    {0.947586f, -0.319502f},
    {0.946601f, -0.322408f},
    {0.945607f, -0.325310f},
    {0.944605f, -0.328210f},
    {0.943593f, -0.331106f},
    {0.942573f, -0.334000f},
    {0.941544f, -0.336890f},
    {0.940506f, -0.339777f},
    {0.939459f, -0.342661f},
    {0.938404f, -0.345541f},
    {0.937339f, -0.348419f},
    {0.936266f, -0.351293f},
    {0.935184f, -0.354164f},
    {0.934093f, -0.357031f},
    {0.932993f, -0.359895f},
    {0.931884f, -0.362756f},
    {0.930767f, -0.365613f},
    {0.929641f, -0.368467f},
    {0.928506f, -0.371317f},
    {0.927363f, -0.374164f},
    {0.926210f, -0.377007f},
    {0.925049f, -0.379847f},
    {0.923880f, -0.382683f},
    {0.922701f, -0.385516f},
    {0.921514f, -0.388345f},
    {0.920318f, -0.391170f},
    {0.919114f, -0.393992f},
    {0.917901f, -0.396810f},
    {0.916679f, -0.399624f},
    {0.915449f, -0.402435f},
    {0.914210f, -0.405241f},
    {0.912962f, -0.408044f},
    {0.911706f, -0.410843f},
    {0.910441f, -0.413638f},
    {0.909168f, -0.416430f},
    {0.907886f, -0.419217f},
    {0.906596f, -0.422000f},
    {0.905297f, -0.424780f},
    {0.903989f, -0.427555f},
    {0.902673f, -0.430326f},
    {0.901349f, -0.433094f},
    {0.900016f, -0.435857f},
    {0.898674f, -0.438616f},
    {0.897325f, -0.441371f},
    {0.895966f, -0.444122f},
    {0.894599f, -0.446869f},
    {0.893224f, -0.449611f},
    {0.891841f, -0.452350f},
    {0.890449f, -0.455084f},
    {0.889048f, -0.457813f},
    {0.887640f, -0.460539f},
    {0.886223f, -0.463260f},
    {0.884797f, -0.465976f},
    {0.883363f, -0.468689f},
    {0.881921f, -0.471397f},
    {0.880471f, -0.474100f},
    {0.879012f, -0.476799f},
    {0.877545f, -0.479494f},
    {0.876070f, -0.482184f},
    {0.874587f, -0.484869f},
    {0.873095f, -0.487550f},
    {0.871595f, -0.490226f},
    {0.870087f, -0.492898f},
    {0.868571f, -0.495565f},
    {0.867046f, -0.498228f},
    {0.865514f, -0.500885f},
    {0.863973f, -0.503538f},
    {0.862424f, -0.506187f},
    {0.860867f, -0.508830f},
    {0.859302f, -0.511469f},
    {0.857729f, -0.514103f},
    {0.856147f, -0.516732f},
    {0.854558f, -0.519356f},
    {0.852961f, -0.521975f},
    {0.851355f, -0.524590f},
    {0.849742f, -0.527199f},
    {0.848120f, -0.529804f},
    {0.846491f, -0.532403f},
    {0.844854f, -0.534998f},
    {0.843208f, -0.537587f},
    {0.841555f, -0.540171f},
    {0.839894f, -0.542751f},
    {0.838225f, -0.545325f},
    {0.836548f, -0.547894f},
    {0.834863f, -0.550458f},
    {0.833170f, -0.553017f},
    {0.831470f, -0.555570f},
    {0.829761f, -0.558119f},
    {0.828045f, -0.560662f},
    {0.826321f, -0.563199f},
    {0.824589f, -0.565732f},
    {0.822850f, -0.568259f},
    {0.821103f, -0.570781f},
    {0.819348f, -0.573297f},
    {0.817585f, -0.575808f},
    {0.815814f, -0.578314f},
    {0.814036f, -0.580814f},
    {0.812251f, -0.583309f},
    {0.810457f, -0.585798f},
    {0.808656f, -0.588282f},
    {0.806848f, -0.590760f},
    {0.805031f, -0.593232f},
    {0.803208f, -0.595699f},
    {0.801376f, -0.598161f},
    {0.799537f, -0.600616f},
    {0.797691f, -0.603067f},
    {0.795837f, -0.605511f},
    {0.793975f, -0.607950f},
    {0.792107f, -0.610383f},
    {0.790230f, -0.612810f},
    {0.788346f, -0.615232f},
    {0.786455f, -0.617647f},
    {0.784557f, -0.620057f},
    {0.782651f, -0.622461f},
    {0.780737f, -0.624859f},
    {0.778817f, -0.627252f},
    {0.776888f, -0.629638f},
    {0.774953f, -0.632019f},
    {0.773010f, -0.634393f},
    {0.771061f, -0.636762f},
    {0.769103f, -0.639124f},
    {0.767139f, -0.641481f},
    {0.765167f, -0.643832f},
    {0.763188f, -0.646176f},
    {0.761202f, -0.648514f},
    {0.759209f, -0.650847f},
    {0.757209f, -0.653173f},
    {0.755201f, -0.655493f},
    {0.753187f, -0.657807f},
    {0.751165f, -0.660114f},
    {0.749136f, -0.662416f},
    {0.747101f, -0.664711f},
    {0.745058f, -0.667000f},
    {0.743008f, -0.669283f},
    {0.740951f, -0.671559f},
    {0.738887f, -0.673829f},
    {0.736817f, -0.676093f},
    {0.734739f, -0.678350f},
    {0.732654f, -0.680601f},
    {0.730563f, -0.682846f},
    {0.728464f, -0.685084f},
    {0.726359f, -0.687315f},
    {0.724247f, -0.689541f},
    {0.722128f, -0.691759f},
    {0.720003f, -0.693971f},
    {0.717870f, -0.696177f},
    {0.715731f, -0.698376f},
    {0.713585f, -0.700569f},
    {0.711432f, -0.702755f},
    {0.709273f, -0.704934f},
    {0.707107f, -0.707107f},
    {0.704934f, -0.709273f},
    {0.702755f, -0.711432f},
    {0.700569f, -0.713585f},
    {0.698376f, -0.715731f},
    {0.696177f, -0.717870f},
    {0.693971f, -0.720003f},
    {0.691759f, -0.722128f},
    {0.689541f, -0.724247f},
    {0.687315f, -0.726359f},
    {0.685084f, -0.728464f},
    {0.682846f, -0.730563f},
    {0.680601f, -0.732654f},
    {0.678350f, -0.734739f},
    {0.676093f, -0.736817f},
    {0.673829f, -0.738887f},
    {0.671559f, -0.740951f},
    {0.669283f, -0.743008f},
    {0.667000f, -0.745058f},
    {0.664711f, -0.747101f},
    {0.662416f, -0.749136f},
    {0.660114f, -0.751165f},
    {0.657807f, -0.753187f},
    {0.655493f, -0.755201f},
    {0.653173f, -0.757209f},
    {0.650847f, -0.759209f},
    {0.648514f, -0.761202f},
    {0.646176f, -0.763188f},
    {0.643832f, -0.765167f},
    {0.641481f, -0.767139f},
    {0.639124f, -0.769103f},
    {0.636762f, -0.771061f},
    {0.634393f, -0.773010f},
    {0.632019f, -0.774953f},
    {0.629638f, -0.776888f},
    {0.627252f, -0.778817f},
    {0.624859f, -0.780737f},
    {0.622461f, -0.782651f},
    {0.620057f, -0.784557f},
    {0.617647f, -0.786455f},
    {0.615232f, -0.788346f},
    {0.612810f, -0.790230f},
    {0.610383f, -0.792107f},
    {0.607950f, -0.793975f},
    {0.605511f, -0.795837f},
    {0.603067f, -0.797691f},
    {0.600616f, -0.799537f},
    {0.598161f, -0.801376f},
    {0.595699f, -0.803208f},
    {0.593232f, -0.805031f},
    {0.590760f, -0.806848f},
    {0.588282f, -0.808656f},
    {0.585798f, -0.810457f},
    {0.583309f, -0.812251f},
    {0.580814f, -0.814036f},
    {0.578314f, -0.815814f},
    {0.575808f, -0.817585f},
    {0.573297f, -0.819348f},
    {0.570781f, -0.821103f},
    {0.568259f, -0.822850f},
    {0.565732f, -0.824589f},
    {0.563199f, -0.826321f},
    {0.560662f, -0.828045f},
    {0.558119f, -0.829761f},
    {0.555570f, -0.831470f},
    {0.553017f, -0.833170f},
    {0.550458f, -0.834863f},
    {0.547894f, -0.836548f},
    {0.545325f, -0.838225f},
    {0.542751f, -0.839894f},
    {0.540171f, -0.841555f},
    {0.537587f, -0.843208f},
    {0.534998f, -0.844854f},
    {0.532403f, -0.846491f},
    {0.529804f, -0.848120f},
    {0.527199f, -0.849742f},
    {0.524590f, -0.851355f},
    {0.521975f, -0.852961f},
    {0.519356f, -0.854558f},
    {0.516732f, -0.856147f},
    {0.514103f, -0.857729f},
    {0.511469f, -0.859302f},
    {0.508830f, -0.860867f},
    {0.506187f, -0.862424f},
    {0.503538f, -0.863973f},
    {0.500885f, -0.865514f},
    {0.498228f, -0.867046f},
    {0.495565f, -0.868571f},
    {0.492898f, -0.870087f},
    {0.490226f, -0.871595f},
    {0.487550f, -0.873095f},
    {0.484869f, -0.874587f},
    {0.482184f, -0.876070f},
    {0.479494f, -0.877545f},
    {0.476799f, -0.879012f},
    {0.474100f, -0.880471f},
    {0.471397f, -0.881921f},
    {0.468689f, -0.883363f},
    {0.465976f, -0.884797f},
    {0.463260f, -0.886223f},
    {0.460539f, -0.887640f},
    {0.457813f, -0.889048f},
    {0.455084f, -0.890449f},
    {0.452350f, -0.891841f},
    {0.449611f, -0.893224f},
    {0.446869f, -0.894599f},
    {0.444122f, -0.895966f},
    {0.441371f, -0.897325f},
    {0.438616f, -0.898674f},
    {0.435857f, -0.900016f},
    {0.433094f, -0.901349f},
    {0.430326f, -0.902673f},
    {0.427555f, -0.903989f},
    {0.424780f, -0.905297f},
    {0.422000f, -0.906596f},
    {0.419217f, -0.907886f},
    {0.416430f, -0.909168f},
    {0.413638f, -0.910441f},
    {0.410843f, -0.911706f},
    {0.408044f, -0.912962f},
    {0.405241f, -0.914210f},
    {0.402435f, -0.915449f},
    {0.399624f, -0.916679f},
    {0.396810f, -0.917901f},
    {0.393992f, -0.919114f},
    {0.391170f, -0.920318f},
    {0.388345f, -0.921514f},
    {0.385516f, -0.922701f},
    {0.382683f, -0.923880f},
    {0.379847f, -0.925049f},
    {0.377007f, -0.926210f},
    {0.374164f, -0.927363f},
    {0.371317f, -0.928506f},
    {0.368467f, -0.929641f},
    {0.365613f, -0.930767f},
    {0.362756f, -0.931884f},
    {0.359895f, -0.932993f},
    {0.357031f, -0.934093f},
    {0.354164f, -0.935184f},
    {0.351293f, -0.936266f},
    {0.348419f, -0.937339f},
    {0.345541f, -0.938404f},
    {0.342661f, -0.939459f},
    {0.339777f, -0.940506f},
    {0.336890f, -0.941544f},
    {0.334000f, -0.942573f},
    {0.331106f, -0.943593f},
    {0.328210f, -0.944605f},
    {0.325310f, -0.945607f},
    {0.322408f, -0.946601f},
    {0.319502f, -0.947586f},
    {0.316593f, -0.948561f},
    {0.313682f, -0.949528f},
    {0.310767f, -0.950486f},
    {0.307850f, -0.951435f},
    {0.304929f, -0.952375f},
    {0.302006f, -0.953306f},
    {0.299080f, -0.954228f},
    {0.296151f, -0.955141f},
    {0.293219f, -0.956045f},
    {0.290285f, -0.956940f},
    {0.287347f, -0.957826f},
    {0.284408f, -0.958703f},
    {0.281465f, -0.959572f},
    {0.278520f, -0.960431f},
    {0.275572f, -0.961280f},
    {0.272621f, -0.962121f},
    {0.269668f, -0.962953f},
    {0.266713f, -0.963776f},
    {0.263755f, -0.964590f},
    {0.260794f, -0.965394f},
    {0.257831f, -0.966190f},
    {0.254866f, -0.966976f},
    {0.251898f, -0.967754f},
    {0.248928f, -0.968522f},
    {0.245955f, -0.969281f},
    {0.242980f, -0.970031f},
    {0.240003f, -0.970772f},
    {0.237024f, -0.971504f},
    {0.234042f, -0.972226f},
    {0.231058f, -0.972940f},
    {0.228072f, -0.973644f},
    {0.225084f, -0.974339f},
    {0.222094f, -0.975025f},
    {0.219101f, -0.975702f},
    {0.216107f, -0.976370f},
    {0.213110f, -0.977028f},
    {0.210112f, -0.977677f},
    {0.207111f, -0.978317f},
    {0.204109f, -0.978948f},
    {0.201105f, -0.979570f},
    {0.198098f, -0.980182f},
    {0.195090f, -0.980785f},
    {0.192080f, -0.981379f},
    {0.189069f, -0.981964f},
    {0.186055f, -0.982539f},
    {0.183040f, -0.983105f},
    {0.180023f, -0.983662f},
    {0.177004f, -0.984210f},
    {0.173984f, -0.984749f},
    {0.170962f, -0.985278f},
    {0.167938f, -0.985798f},
    {0.164913f, -0.986308f},
    {0.161886f, -0.986809f},
    {0.158858f, -0.987301f},
    {0.155828f, -0.987784f},
    {0.152797f, -0.988258f},
    {0.149765f, -0.988722f},
    {0.146730f, -0.989177f},
    {0.143695f, -0.989622f},
    {0.140658f, -0.990058f},
    {0.137620f, -0.990485f},
    {0.134581f, -0.990903f},
    {0.131540f, -0.991311f},
    {0.128498f, -0.991710f},
    {0.125455f, -0.992099f},
    {0.122411f, -0.992480f},
    {0.119365f, -0.992850f},
    {0.116319f, -0.993212f},
    {0.113271f, -0.993564f},
    {0.110222f, -0.993907f},
    {0.107172f, -0.994240f},
    {0.104122f, -0.994565f},
    {0.101070f, -0.994879f},
    {0.098017f, -0.995185f},
    {0.094963f, -0.995481f},
    {0.091909f, -0.995767f},
    {0.088854f, -0.996045f},
    {0.085797f, -0.996313f},
    {0.082740f, -0.996571f},
    {0.079682f, -0.996820f},
    {0.076624f, -0.997060f},
    {0.073565f, -0.997290f},
    {0.070505f, -0.997511f},
    {0.067444f, -0.997723f},
    {0.064383f, -0.997925f},
    {0.061321f, -0.998118f},
    {0.058258f, -0.998302f},
    {0.055195f, -0.998476f},
    {0.052132f, -0.998640f},
    {0.049068f, -0.998795f},
    {0.046003f, -0.998941f},
    {0.042938f, -0.999078f},
    {0.039873f, -0.999205f},
    {0.036807f, -0.999322f},
    {0.033741f, -0.999431f},
    {0.030675f, -0.999529f},
    {0.027608f, -0.999619f},
    {0.024541f, -0.999699f},
    {0.021474f, -0.999769f},
    {0.018407f, -0.999831f},
    {0.015339f, -0.999882f},
    {0.012272f, -0.999925f},
    {0.009204f, -0.999958f},
    {0.006136f, -0.999981f},
    {0.003068f, -0.999995f},
    {0.000000f, -1.000000f},
    {-0.003068f, -0.999995f},
    {-0.006136f, -0.999981f},
    {-0.009204f, -0.999958f},
    {-0.012272f, -0.999925f},
    {-0.015339f, -0.999882f},
    {-0.018407f, -0.999831f},
    {-0.021474f, -0.999769f},
    {-0.024541f, -0.999699f},
    {-0.027608f, -0.999619f},
    {-0.030675f, -0.999529f},
    {-0.033741f, -0.999431f},
    {-0.036807f, -0.999322f},
    {-0.039873f, -0.999205f},
    {-0.042938f, -0.999078f},
    {-0.046003f, -0.998941f},
    {-0.049068f, -0.998795f},
    {-0.052132f, -0.998640f},
    {-0.055195f, -0.998476f},
    {-0.058258f, -0.998302f},
    {-0.061321f, -0.998118f},
    {-0.064383f, -0.997925f},
    {-0.067444f, -0.997723f},
    {-0.070505f, -0.997511f},
    {-0.073565f, -0.997290f},
    {-0.076624f, -0.997060f},
    {-0.079682f, -0.996820f},
    {-0.082740f, -0.996571f},
    {-0.085797f, -0.996313f},
    {-0.088854f, -0.996045f},
    {-0.091909f, -0.995767f},
    {-0.094963f, -0.995481f},
    {-0.098017f, -0.995185f},
    {-0.101070f, -0.994879f},
    {-0.104122f, -0.994565f},
    {-0.107172f, -0.994240f},
    {-0.110222f, -0.993907f},
    {-0.113271f, -0.993564f},
    {-0.116319f, -0.993212f},
    {-0.119365f, -0.992850f},
    {-0.122411f, -0.992480f},
    {-0.125455f, -0.992099f},
    {-0.128498f, -0.991710f},
    {-0.131540f, -0.991311f},
    {-0.134581f, -0.990903f},
    {-0.137620f, -0.990485f},
    {-0.140658f, -0.990058f},
    {-0.143695f, -0.989622f},
    {-0.146730f, -0.989177f},
    {-0.149765f, -0.988722f},
    {-0.152797f, -0.988258f},
    {-0.155828f, -0.987784f},
    {-0.158858f, -0.987301f},
    {-0.161886f, -0.986809f},
    {-0.164913f, -0.986308f},
    {-0.167938f, -0.985798f},
    {-0.170962f, -0.985278f},
    {-0.173984f, -0.984749f},
    {-0.177004f, -0.984210f},
    {-0.180023f, -0.983662f},
    {-0.183040f, -0.983105f},
    {-0.186055f, -0.982539f},
    {-0.189069f, -0.981964f},
    {-0.192080f, -0.981379f},
    {-0.195090f, -0.980785f},
    {-0.198098f, -0.980182f},
    {-0.201105f, -0.979570f},
    {-0.204109f, -0.978948f},
    {-0.207111f, -0.978317f},
    {-0.210112f, -0.977677f},
    {-0.213110f, -0.977028f},
    {-0.216107f, -0.976370f},
    {-0.219101f, -0.975702f},
    {-0.222094f, -0.975025f},
    {-0.225084f, -0.974339f},
    {-0.228072f, -0.973644f},
    {-0.231058f, -0.972940f},
    {-0.234042f, -0.972226f},
    {-0.237024f, -0.971504f},
    {-0.240003f, -0.970772f},
    {-0.242980f, -0.970031f},
    {-0.245955f, -0.969281f},
    {-0.248928f, -0.968522f},
    {-0.251898f, -0.967754f},
    {-0.254866f, -0.966976f},
    {-0.257831f, -0.966190f},
    {-0.260794f, -0.965394f},
    {-0.263755f, -0.964590f},
    {-0.266713f, -0.963776f},
    {-0.269668f, -0.962953f},
    {-0.272621f, -0.962121f},
    {-0.275572f, -0.961280f},
    {-0.278520f, -0.960431f},
    {-0.281465f, -0.959572f},
    {-0.284408f, -0.958703f},
    {-0.287347f, -0.957826f},
    {-0.290285f, -0.956940f},
    {-0.293219f, -0.956045f},
    {-0.296151f, -0.955141f},
    {-0.299080f, -0.954228f},
    {-0.302006f, -0.953306f},
    {-0.304929f, -0.952375f},
    {-0.307850f, -0.951435f},
    {-0.310767f, -0.950486f},
    {-0.313682f, -0.949528f},
    {-0.316593f, -0.948561f},
    {-0.319502f, -0.947586f},
    {-0.322408f, -0.946601f},
    {-0.325310f, -0.945607f},
    {-0.328210f, -0.944605f},
    {-0.331106f, -0.943593f},
    {-0.334000f, -0.942573f},
    {-0.336890f, -0.941544f},
    {-0.339777f, -0.940506f},
    {-0.342661f, -0.939459f},
    {-0.345541f, -0.938404f},
    {-0.348419f, -0.937339f},
    {-0.351293f, -0.936266f},
    {-0.354164f, -0.935184f},
    {-0.357031f, -0.934093f},
    {-0.359895f, -0.932993f},
    {-0.362756f, -0.931884f},
    {-0.365613f, -0.930767f},
    {-0.368467f, -0.929641f},
    {-0.371317f, -0.928506f},
    {-0.374164f, -0.927363f},
    {-0.377007f, -0.926210f},
    {-0.379847f, -0.925049f},
    {-0.382683f, -0.923880f},
    {-0.385516f, -0.922701f},
    {-0.388345f, -0.921514f},
    {-0.391170f, -0.920318f},
    {-0.393992f, -0.919114f},
    {-0.396810f, -0.917901f},
    {-0.399624f, -0.916679f},
    {-0.402435f, -0.915449f},
    {-0.405241f, -0.914210f},
    {-0.408044f, -0.912962f},
    {-0.410843f, -0.911706f},
    {-0.413638f, -0.910441f},
    {-0.416430f, -0.909168f},
    {-0.419217f, -0.907886f},
    {-0.422000f, -0.906596f},
    {-0.424780f, -0.905297f},
    {-0.427555f, -0.903989f},
    {-0.430326f, -0.902673f},
    {-0.433094f, -0.901349f},
    {-0.435857f, -0.900016f},
    {-0.438616f, -0.898674f},
    {-0.441371f, -0.897325f},
    {-0.444122f, -0.895966f},
    {-0.446869f, -0.894599f},
    {-0.449611f, -0.893224f},
    {-0.452350f, -0.891841f},
    {-0.455084f, -0.890449f},
    {-0.457813f, -0.889048f},
    {-0.460539f, -0.887640f},
    {-0.463260f, -0.886223f},
    {-0.465976f, -0.884797f},
    {-0.468689f, -0.883363f},
    {-0.471397f, -0.881921f},
    {-0.474100f, -0.880471f},
    {-0.476799f, -0.879012f},
    {-0.479494f, -0.877545f},
    {-0.482184f, -0.876070f},
    {-0.484869f, -0.874587f},
    {-0.487550f, -0.873095f},
    {-0.490226f, -0.871595f},
    {-0.492898f, -0.870087f},
    {-0.495565f, -0.868571f},
    {-0.498228f, -0.867046f},
    {-0.500885f, -0.865514f},
    {-0.503538f, -0.863973f},
    {-0.506187f, -0.862424f},
    {-0.508830f, -0.860867f},
    {-0.511469f, -0.859302f},
    {-0.514103f, -0.857729f},
    {-0.516732f, -0.856147f},
    {-0.519356f, -0.854558f},
    {-0.521975f, -0.852961f},
    {-0.524590f, -0.851355f},
    {-0.527199f, -0.849742f},
    {-0.529804f, -0.848120f},
    {-0.532403f, -0.846491f},
    {-0.534998f, -0.844854f},
    {-0.537587f, -0.843208f},
    {-0.540171f, -0.841555f},
    {-0.542751f, -0.839894f},
    {-0.545325f, -0.838225f},
    {-0.547894f, -0.836548f},
    {-0.550458f, -0.834863f},
    {-0.553017f, -0.833170f},
    {-0.555570f, -0.831470f},
    {-0.558119f, -0.829761f},
    {-0.560662f, -0.828045f},
    {-0.563199f, -0.826321f},
    {-0.565732f, -0.824589f},
    {-0.568259f, -0.822850f},
    {-0.570781f, -0.821103f},
    {-0.573297f, -0.819348f},
    {-0.575808f, -0.817585f},
    {-0.578314f, -0.815814f},
    {-0.580814f, -0.814036f},
    {-0.583309f, -0.812251f},
    {-0.585798f, -0.810457f},
    {-0.588282f, -0.808656f},
    {-0.590760f, -0.806848f},
    {-0.593232f, -0.805031f},
    {-0.595699f, -0.803208f},
    {-0.598161f, -0.801376f},
    {-0.600616f, -0.799537f},
    {-0.603067f, -0.797691f},
    {-0.605511f, -0.795837f},
    {-0.607950f, -0.793975f},
    {-0.610383f, -0.792107f},
    {-0.612810f, -0.790230f},
    {-0.615232f, -0.788346f},
    {-0.617647f, -0.786455f},
    {-0.620057f, -0.784557f},
    {-0.622461f, -0.782651f},
    {-0.624859f, -0.780737f},
    {-0.627252f, -0.778817f},
    {-0.629638f, -0.776888f},
    {-0.632019f, -0.774953f},
    {-0.634393f, -0.773010f},
    {-0.636762f, -0.771061f},
    {-0.639124f, -0.769103f},
    {-0.641481f, -0.767139f},
    {-0.643832f, -0.765167f},
    {-0.646176f, -0.763188f},
    {-0.648514f, -0.761202f},
    {-0.650847f, -0.759209f},
    {-0.653173f, -0.757209f},
    {-0.655493f, -0.755201f},
    {-0.657807f, -0.753187f},
    {-0.660114f, -0.751165f},
    {-0.662416f, -0.749136f},
    {-0.664711f, -0.747101f},
    {-0.667000f, -0.745058f},
    {-0.669283f, -0.743008f},
    {-0.671559f, -0.740951f},
    {-0.673829f, -0.738887f},
    {-0.676093f, -0.736817f},
    {-0.678350f, -0.734739f},
    {-0.680601f, -0.732654f},
    {-0.682846f, -0.730563f},
    {-0.685084f, -0.728464f},
    {-0.687315f, -0.726359f},
    {-0.689541f, -0.724247f},
    {-0.691759f, -0.722128f},
    {-0.693971f, -0.720003f},
    {-0.696177f, -0.717870f},
    {-0.698376f, -0.715731f},
    {-0.700569f, -0.713585f},
    {-0.702755f, -0.711432f},
    {-0.704934f, -0.709273f},
    {-0.707107f, -0.707107f},
    {-0.709273f, -0.704934f},
    {-0.711432f, -0.702755f},
    {-0.713585f, -0.700569f},
    {-0.715731f, -0.698376f},
    {-0.717870f, -0.696177f},
    {-0.720003f, -0.693971f},
    {-0.722128f, -0.691759f},
    {-0.724247f, -0.689541f},
    {-0.726359f, -0.687315f},
    {-0.728464f, -0.685084f},
    {-0.730563f, -0.682846f},
    {-0.732654f, -0.680601f},
    {-0.734739f, -0.678350f},
    {-0.736817f, -0.676093f},
    {-0.738887f, -0.673829f},
    {-0.740951f, -0.671559f},
    {-0.743008f, -0.669283f},
    {-0.745058f, -0.667000f},
    {-0.747101f, -0.664711f},
    {-0.749136f, -0.662416f},
    {-0.751165f, -0.660114f},
    {-0.753187f, -0.657807f},
    {-0.755201f, -0.655493f},
    {-0.757209f, -0.653173f},
    {-0.759209f, -0.650847f},
    {-0.761202f, -0.648514f},
    {-0.763188f, -0.646176f},
    {-0.765167f, -0.643832f},
    {-0.767139f, -0.641481f},
    {-0.769103f, -0.639124f},
    {-0.771061f, -0.636762f},
    {-0.773010f, -0.634393f},
    {-0.774953f, -0.632019f},
    {-0.776888f, -0.629638f},
    {-0.778817f, -0.627252f},
    {-0.780737f, -0.624859f},
    {-0.782651f, -0.622461f},
    {-0.784557f, -0.620057f},
    {-0.786455f, -0.617647f},
    {-0.788346f, -0.615232f},
    {-0.790230f, -0.612810f},
    {-0.792107f, -0.610383f},
    {-0.793975f, -0.607950f},
    {-0.795837f, -0.605511f},
    {-0.797691f, -0.603067f},
    {-0.799537f, -0.600616f},
    {-0.801376f, -0.598161f},
    {-0.803208f, -0.595699f},
    {-0.805031f, -0.593232f},
    {-0.806848f, -0.590760f},
    {-0.808656f, -0.588282f},
    {-0.810457f, -0.585798f},
    {-0.812251f, -0.583309f},
    {-0.814036f, -0.580814f},
    {-0.815814f, -0.578314f},
    {-0.817585f, -0.575808f},
    {-0.819348f, -0.573297f},
    {-0.821103f, -0.570781f},
    {-0.822850f, -0.568259f},
    {-0.824589f, -0.565732f},
    {-0.826321f, -0.563199f},
    {-0.828045f, -0.560662f},
    {-0.829761f, -0.558119f},
    {-0.831470f, -0.555570f},
    {-0.833170f, -0.553017f},
    {-0.834863f, -0.550458f},
    {-0.836548f, -0.547894f},
    {-0.838225f, -0.545325f},
    {-0.839894f, -0.542751f},
    {-0.841555f, -0.540171f},
    {-0.843208f, -0.537587f},
    {-0.844854f, -0.534998f},
    {-0.846491f, -0.532403f},
    {-0.848120f, -0.529804f},
    {-0.849742f, -0.527199f},
    {-0.851355f, -0.524590f},
    {-0.852961f, -0.521975f},
    {-0.854558f, -0.519356f},
    {-0.856147f, -0.516732f},
    {-0.857729f, -0.514103f},
    {-0.859302f, -0.511469f},
    {-0.860867f, -0.508830f},
    {-0.862424f, -0.506187f},
    {-0.863973f, -0.503538f},
    {-0.865514f, -0.500885f},
    {-0.867046f, -0.498228f},
    {-0.868571f, -0.495565f},
    {-0.870087f, -0.492898f},
    {-0.871595f, -0.490226f},
    {-0.873095f, -0.487550f},
    {-0.874587f, -0.484869f},
    {-0.876070f, -0.482184f},
    {-0.877545f, -0.479494f},
    {-0.879012f, -0.476799f},
    {-0.880471f, -0.474100f},
    {-0.881921f, -0.471397f},
    {-0.883363f, -0.468689f},
    {-0.884797f, -0.465976f},
    {-0.886223f, -0.463260f},
    {-0.887640f, -0.460539f},
    {-0.889048f, -0.457813f},
    {-0.890449f, -0.455084f},
    {-0.891841f, -0.452350f},
    {-0.893224f, -0.449611f},
    {-0.894599f, -0.446869f},
    {-0.895966f, -0.444122f},
    {-0.897325f, -0.441371f},
    {-0.898674f, -0.438616f},
    {-0.900016f, -0.435857f},
    {-0.901349f, -0.433094f},
    {-0.902673f, -0.430326f},
    {-0.903989f, -0.427555f},
    {-0.905297f, -0.424780f},
    {-0.906596f, -0.422000f},
    {-0.907886f, -0.419217f},
    {-0.909168f, -0.416430f},
    {-0.910441f, -0.413638f},
    {-0.911706f, -0.410843f},
    {-0.912962f, -0.408044f},
    {-0.914210f, -0.405241f},
    {-0.915449f, -0.402435f},
    {-0.916679f, -0.399624f},
    {-0.917901f, -0.396810f},
    {-0.919114f, -0.393992f},
    {-0.920318f, -0.391170f},
    {-0.921514f, -0.388345f},
    {-0.922701f, -0.385516f},
    {-0.923880f, -0.382683f},
    {-0.925049f, -0.379847f},
    {-0.926210f, -0.377007f},
    {-0.927363f, -0.374164f},
    {-0.928506f, -0.371317f},
    {-0.929641f, -0.368467f},
    {-0.930767f, -0.365613f},
    {-0.931884f, -0.362756f},
    {-0.932993f, -0.359895f},
    {-0.934093f, -0.357031f},
    {-0.935184f, -0.354164f},
    {-0.936266f, -0.351293f},
    {-0.937339f, -0.348419f},
    {-0.938404f, -0.345541f},
    {-0.939459f, -0.342661f},
    {-0.940506f, -0.339777f},
    {-0.941544f, -0.336890f},
    {-0.942573f, -0.334000f},
    {-0.943593f, -0.331106f},
    {-0.944605f, -0.328210f},
    {-0.945607f, -0.325310f},
    {-0.946601f, -0.322408f},
    {-0.947586f, -0.319502f},
    {-0.948561f, -0.316593f},
    {-0.949528f, -0.313682f},
    {-0.950486f, -0.310767f},
    {-0.951435f, -0.307850f},
    {-0.952375f, -0.304929f},
    {-0.953306f, -0.302006f},
    {-0.954228f, -0.299080f},
    {-0.955141f, -0.296151f},
    {-0.956045f, -0.293219f},
    {-0.956940f, -0.290285f},
    {-0.957826f, -0.287347f},
    {-0.958703f, -0.284408f},
    {-0.959572f, -0.281465f},
    {-0.960431f, -0.278520f},
    {-0.961280f, -0.275572f},
    {-0.962121f, -0.272621f},
    {-0.962953f, -0.269668f},
    {-0.963776f, -0.266713f},
    {-0.964590f, -0.263755f},
    {-0.965394f, -0.260794f},
    {-0.966190f, -0.257831f},
    {-0.966976f, -0.254866f},
    {-0.967754f, -0.251898f},
    {-0.968522f, -0.248928f},
    {-0.969281f, -0.245955f},
    {-0.970031f, -0.242980f},
    {-0.970772f, -0.240003f},
    {-0.971504f, -0.237024f},
    {-0.972226f, -0.234042f},
    {-0.972940f, -0.231058f},
    {-0.973644f, -0.228072f},
    {-0.974339f, -0.225084f},
    {-0.975025f, -0.222094f},
    {-0.975702f, -0.219101f},
    {-0.976370f, -0.216107f},
    {-0.977028f, -0.213110f},
    {-0.977677f, -0.210112f},
    {-0.978317f, -0.207111f},
    {-0.978948f, -0.204109f},
    {-0.979570f, -0.201105f},
    {-0.980182f, -0.198098f},
    {-0.980785f, -0.195090f},
    {-0.981379f, -0.192080f},
    {-0.981964f, -0.189069f},
    {-0.982539f, -0.186055f},
    {-0.983105f, -0.183040f},
    {-0.983662f, -0.180023f},
    {-0.984210f, -0.177004f},
    {-0.984749f, -0.173984f},
    {-0.985278f, -0.170962f},
    {-0.985798f, -0.167938f},
    {-0.986308f, -0.164913f},
    {-0.986809f, -0.161886f},
    {-0.987301f, -0.158858f},
    {-0.987784f, -0.155828f},
    {-0.988258f, -0.152797f},
    {-0.988722f, -0.149765f},
    {-0.989177f, -0.146730f},
    {-0.989622f, -0.143695f},
    {-0.990058f, -0.140658f},
    {-0.990485f, -0.137620f},
    {-0.990903f, -0.134581f},
    {-0.991311f, -0.131540f},
    {-0.991710f, -0.128498f},
    {-0.992099f, -0.125455f},
    {-0.992480f, -0.122411f},
    {-0.992850f, -0.119365f},
    {-0.993212f, -0.116319f},
    {-0.993564f, -0.113271f},
    {-0.993907f, -0.110222f},
    {-0.994240f, -0.107172f},
    {-0.994565f, -0.104122f},
    {-0.994879f, -0.101070f},
    {-0.995185f, -0.098017f},
    {-0.995481f, -0.094963f},
    {-0.995767f, -0.091909f},
    {-0.996045f, -0.088854f},
    {-0.996313f, -0.085797f},
    {-0.996571f, -0.082740f},
    {-0.996820f, -0.079682f},
    {-0.997060f, -0.076624f},
    {-0.997290f, -0.073565f},
    {-0.997511f, -0.070505f},
    {-0.997723f, -0.067444f},
    {-0.997925f, -0.064383f},
    {-0.998118f, -0.061321f},
    {-0.998302f, -0.058258f},
    {-0.998476f, -0.055195f},
    {-0.998640f, -0.052132f},
    {-0.998795f, -0.049068f},
    {-0.998941f, -0.046003f},
    {-0.999078f, -0.042938f},
    {-0.999205f, -0.039873f},
    {-0.999322f, -0.036807f},
    {-0.999431f, -0.033741f},
    {-0.999529f, -0.030675f},
    {-0.999619f, -0.027608f},
    {-0.999699f, -0.024541f},
    {-0.999769f, -0.021474f},
    {-0.999831f, -0.018407f},
    {-0.999882f, -0.015339f},
    {-0.999925f, -0.012272f},
    {-0.999958f, -0.009204f},
    {-0.999981f, -0.006136f},
    {-0.999995f, -0.003068f},
    */
};
#endif
