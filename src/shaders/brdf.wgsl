const PI = 3.14159265359;
const SAMPLE_COUNT = 1024u;

@group(0) @binding(0) var<uniform> size: f32;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

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

// G: Geometry Function
fn GeometrySmith(normal: vec3f, viewDir: vec3f, lightDir: vec3f, roughness: f32) -> f32 {
  let NdotV = max(dot(normal, viewDir), 0.0);
  let ggx1 = GeometrySmithGGX(NdotV, roughness);
  let NdotL = max(dot(normal, lightDir), 0.0);
  let ggx2 = GeometrySmithGGX(NdotL, roughness);

  return ggx1 * ggx2;
}

fn GeometrySmithGGX(cosTheta: f32, roughness: f32) -> f32 {
  let r = roughness;
  let k = (r * r) / 2.0;

  let numerator = cosTheta;
  let denominator = (cosTheta * (1.0 - k) + k);

  return numerator / denominator;
}

fn IntegrateBRDF( NdotV: f32, roughness: f32) -> vec2f {
    let V = vec3f(sqrt(1.0 - NdotV*NdotV), 0.0, NdotV);

    var A = 0.0;
    var B = 0.0;

    let N = vec3(0.0, 0.0, 1.0);

    for(var i = 0u; i < SAMPLE_COUNT; i += 1) {
        let Xi = Hammersley(i, SAMPLE_COUNT);
        let H  = ImportanceSampleGGX(Xi, N, roughness);
        let L  = normalize(2.0 * dot(V, H) * H - V);

        let NdotL = max(L.z, 0.0);
        let NdotH = max(H.z, 0.0);
        let VdotH = max(dot(V, H), 0.0);

        if NdotL > 0.0 {
            let G = GeometrySmith(N, V, L, roughness);
            let G_Vis = (G * VdotH) / (NdotH * NdotV);
            let Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    A /= f32(SAMPLE_COUNT);
    B /= f32(SAMPLE_COUNT);
    return vec2f(A, B);
}

@compute @workgroup_size(16, 16)
fn compute_main(@builtin(global_invocation_id) gid: vec3u) {
    if gid.x >= u32(textureDimensions(dst).x) || gid.y >= u32(textureDimensions(dst).y) {
      return;
    }

    let integratedBRDF = IntegrateBRDF(f32(gid.x) / (size - 1.0), (size - f32(gid.y)) / (size - 1.0));
    textureStore(dst, gid.xy, vec4f(integratedBRDF, 0.0, 1.0));
}
