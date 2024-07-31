const PI: f32 = 3.14159265359;
const invAtan = vec2f(0.1591, 0.3183);
const SAMPLE_COUNT = 1024u;


struct Face {
    forward: vec3f,
    up: vec3f,
    right: vec3f,
}

@group(0) @binding(0) var<uniform> roughness: f32;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d_array<rgba32float, write>;

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
fn RadicalInverseVdC(bits: u32) -> f32 {
  var val = (bits << 16u) | (bits >> 16u);
  val = ((val & 0x55555555u) << 1u) | ((val & 0xAAAAAAAAu) >> 1u);
  val = ((val & 0x33333333u) << 2u) | ((val & 0xCCCCCCCCu) >> 2u);
  val = ((val & 0x0F0F0F0Fu) << 4u) | ((val & 0xF0F0F0F0u) >> 4u);
  val = ((val & 0x00FF00FFu) << 8u) | ((val & 0xFF00FF00u) >> 8u);
  return f32(val) * 2.3283064365386963e-10; // / 0x100000000
}

fn Hammersley(i: u32, N: u32) -> vec2f {
  return vec2f(f32(i)/f32(N), RadicalInverseVdC(i));
}

fn ImportanceSampleGGX(Xi: vec2f,  N: vec3f, roughness: f32) -> vec3f {
  let a = roughness*roughness;

  let phi = 2.0 * PI * Xi.x;
  let cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
  let sinTheta = sqrt(1.0 - cosTheta * cosTheta);

  // from spherical coordinates to cartesian coordinates - halfway vector
  let H = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

  // from tangent-space H vector to world-space sample vector
  var up: vec3f;
  if abs(N.z) < 0.999 {
    up = vec3f(0.0, 0.0, 1.0);
  } else {
    up = vec3f(1.0, 0.0, 0.0);
  }

  let tangent = normalize(cross(up, N));
  let bitangent = cross(N, tangent);

  let sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
  return normalize(sampleVec);
}

@compute @workgroup_size(16, 16, 1)
fn compute_main(@builtin(global_invocation_id) gid: vec3u) {
    // If texture size is not divisible by 32, we need to make sure we don't try to write to pixels that don't exist.
    if gid.x >= u32(textureDimensions(dst).x) || gid.y >= u32(textureDimensions(dst).y) {
      return;
    }

    let faces: array<Face, 6> = array(
        // FACES +X
        Face(
            vec3f(1.0, 0.0, 0.0),  // forward
            vec3f(0.0, 1.0, 0.0),  // up
            vec3f(0.0, 0.0, -1.0), // right
        ),
        // FACES -X
        Face (
            vec3f(-1.0, 0.0, 0.0),
            vec3f(0.0, 1.0, 0.0),
            vec3f(0.0, 0.0, 1.0),
        ),
        // FACES +Y
        Face (
            vec3f(0.0, -1.0, 0.0),
            vec3f(0.0, 0.0, 1.0),
            vec3f(1.0, 0.0, 0.0),
        ),
        // FACES -Y
        Face (
            vec3f(0.0, 1.0, 0.0),
            vec3f(0.0, 0.0, -1.0),
            vec3f(1.0, 0.0, 0.0),
        ),
        // FACES +Z
        Face (
            vec3f(0.0, 0.0, 1.0),
            vec3f(0.0, 1.0, 0.0),
            vec3f(1.0, 0.0, 0.0),
        ),
        // FACES -Z
        Face (
            vec3f(0.0, 0.0, -1.0),
            vec3f(0.0, 1.0, 0.0),
            vec3f(-1.0, 0.0, 0.0),
        ),
    );

    // Get texture coords relative to cubemap face
    let dstDimensions = vec2f(textureDimensions(dst));
    let cubeUV = vec2f(gid.xy) / dstDimensions * 2.0 - 1.0;

    // Get spherical coordinate from cubeUV
    let face = faces[gid.z];
    let spherical = normalize(face.forward + face.right * cubeUV.x + face.up * cubeUV.y);

    let N = spherical;
    let R = N;
    let V = R;

    var prefilteredColor = vec3f(0.0);
    var totalWeight = 0.0;

    for(var i: u32 = 0u; i < SAMPLE_COUNT; i = i + 1) {
      let Xi = Hammersley(i, SAMPLE_COUNT);
      let H = ImportanceSampleGGX(Xi, N, roughness);
      let L = normalize(2.0 * dot(V, H) * H - V);

      let NdotL = max(dot(N, L), 0.0);
      if(NdotL > 0.0) {
        let D = TrowbridgeReitzGGX(N, H, roughness);
        let NdotH = max(dot(N, H), 0.0);
        let HdotV = max(dot(H, V), 0.0);
        let pdf = D * NdotH / (4.0 * HdotV) + 0.0001; 

        let resolution = 512.0; // resolution of source cubemap (per face)
        let saTexel  = 4.0 * PI / (6.0 * resolution * resolution);
        let saSample = 1.0 / (f32(SAMPLE_COUNT) * pdf + 0.0001);

        var mipLevel: f32;
        if  roughness == 0.0 {
          mipLevel = 0.0;
        } else {
          mipLevel = 0.5 * log2(saSample / saTexel);
        }
        let mipLevel0 = i32(floor(mipLevel));
        let mipLevel1 = mipLevel0 + 1;
        let mixFactor = mipLevel - f32(mipLevel0);

        let eqUV0 = vec2f(atan2(L.z, L.x), asin(L.y)) * invAtan + 0.5;
        let eqPixel0 = vec2i(eqUV0 * vec2f(textureDimensions(src, mipLevel0)));
        let texel0 = textureLoad(src, eqPixel0, mipLevel0).rgb * NdotL;

        let eqUV1 = vec2f(atan2(L.z, L.x), asin(L.y)) * invAtan + 0.5;
        let eqPixel1 = vec2i(eqUV1 * vec2f(textureDimensions(src, mipLevel1)));
        let texel1 = textureLoad(src, eqPixel1, mipLevel1).rgb * NdotL;

        prefilteredColor += mix(texel0, texel1, mixFactor);
        totalWeight += NdotL;
      } 
    }

    textureStore(dst, gid.xy, gid.z, vec4f(prefilteredColor/totalWeight, 1.0));
}
