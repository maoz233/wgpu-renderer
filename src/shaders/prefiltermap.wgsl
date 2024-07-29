const PI: f32 = 3.14159265359;
const invAtan = vec2(0.1591, 0.3183);
const SAMPLE_COUNT = 1024u;


struct Face {
    forward: vec3f,
    up: vec3f,
    right: vec3f,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba32float, write>;

// D: Normal Distribution Function
fn TrowbridgeReitzGGX(normal: vec3f, halfwayDir: vec3f, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let NdotH = max(dot(normal, halfwayDir), 0.0);
  let NdotH2 = NdotH * NdotH;

  let numerator = a2;
  var denominator = (NdotH2 * (a2 - 1.0) + 1.0);
  denominator = PI * denominator * denominator;

  return numerator / denominator; 
}

// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// efficient VanDerCorpus calculation.
fn RadicalInverseVdC(bits: uint) -> f32 {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return (bits) * 2.3283064365386963e-10; // / 0x100000000
}

fn Hammersley(i: uint, N: uint) -> vec2f {
  return vec2f(f32(i)/f32(N), RadicalInverseVdC(i));
}

fn ImportanceSampleGGX(Xi: vec2f,  N: vec3f, roughness: f32) -> vec3f {
  let a = roughness*roughness;

  let phi = 2.0 * PI * Xi.x;
  let cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
  let sinTheta = sqrt(1.0 - cosTheta * cosTheta);

  // from spherical coordinates to cartesian coordinates - halfway vector
  vec3 H;
  H.x = cos(phi) * sinTheta;
  H.y = sin(phi) * sinTheta;
  H.z = cosTheta;

  // from tangent-space H vector to world-space sample vector
  vec3 up          = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tangent   = normalize(cross(up, N));
  vec3 bitangent = cross(N, tangent);

  vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
  return normalize(sampleVec);
}

@compute
@workgroup_size(16, 16, 1)
fn compute_main(@builtin(global_invocation_id) gid: vec3u) {
    // If texture size is not divisible by 32, we need to make sure we don't try to write to pixels that don't exist.
    if gid.x >= u32(textureDimensions(dst).x) {
        return;
    }

    let faces: array<Face, 6> = array(
        // FACES +X
        Face(
            vec3(1.0, 0.0, 0.0),  // forward
            vec3(0.0, 1.0, 0.0),  // up
            vec3(0.0, 0.0, -1.0), // right
        ),
        // FACES -X
        Face (
            vec3(-1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0),
        ),
        // FACES +Y
        Face (
            vec3(0.0, -1.0, 0.0),
            vec3(0.0, 0.0, 1.0),
            vec3(1.0, 0.0, 0.0),
        ),
        // FACES -Y
        Face (
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, -1.0),
            vec3(1.0, 0.0, 0.0),
        ),
        // FACES +Z
        Face (
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 1.0, 0.0),
            vec3(1.0, 0.0, 0.0),
        ),
        // FACES -Z
        Face (
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, 0.0, 0.0),
        ),
    );

    // Get texture coords relative to cubemap face
    let dstDimensions = vec2<f32>(textureDimensions(dst));
    let cubeUV = vec2<f32>(gid.xy) / dstDimensions * 2.0 - 1.0;

    // Get spherical coordinate from cubeUV
    let face = faces[gid.z];
    let spherical = normalize(face.forward + face.right * cubeUV.x + face.up * cubeUV.y);
    let eqUV = vec2(atan2(sampleVec.z, sampleVec.x), asin(sampleVec.y)) * invAtan + 0.5;

    let N = spherical;
    let R = N;
    let V = R;

    let eqPixel = vec2<i32>(eqUV * vec2<f32>(textureDimensions(src)));

    textureStore(dst, gid.xy, gid.z, vec4f(eqPixel, 1.0));
}
