struct VertexInput {
  @location(0) position : vec3f,
  @location(1) texCoord : vec2f,
  @location(2) normal : vec3f
};

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) texCoord : vec2f,
  @location(1) normal : vec3f,
  @location(2) pos: vec3f,
};

struct Transform {
  model: mat4x4f,
  normal: mat4x4f,
  view: mat4x4f,
  projection: mat4x4f,
};

@group(0) @binding(0) var<uniform> transform: Transform;
@group(0) @binding(1) var<uniform> viewPos: vec3f;
@group(0) @binding(2) var<uniform> lightPositions: array<vec3f, 4>;
@group(0) @binding(3) var<uniform> lightColors: array<vec3f, 4>;
@group(1) @binding(0) var albedoMap: texture_2d<f32>;
@group(1) @binding(1) var metallicRoughnessMap: texture_2d<f32>;
@group(1) @binding(2) var normalMap: texture_2d<f32>;
@group(1) @binding(3) var<uniform> emissiveFactor: vec3f;
@group(1) @binding(4) var emissiveMap: texture_2d<f32>;
@group(1) @binding(5) var occlusionMap: texture_2d<f32>;
@group(2) @binding(0) var irradianceMap: texture_cube<f32>;
@group(3) @binding(0) var sampler2D: sampler;
@group(3) @binding(1) var samplerCube: sampler;

@vertex
fn vs_main(vertData: VertexInput) -> VertexOut {
  var output: VertexOut;

  output.position = transform.projection * transform.view * transform.model * vec4f(vertData.position, 1.0);
  output.texCoord = vertData.texCoord;
  output.normal = (transform.normal * vec4f(vertData.normal, 0.0)).xyz;
  output.pos = (transform.normal * vec4f(vertData.position, 0.0)).xyz;

  return output;
}

const PI = 3.14159265359;

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

// F: Fresnel Equation
fn FresnelSchlickApproximation(cosTheta: f32, f0: vec3f, roughness: f32) -> vec3f {
  return f0 + (max(vec3f(1.0 -roughness), f0) - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
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
  let r = (roughness + 1.0);
  let k = (r * r) / 8.0;

  let numerator = cosTheta;
  let denominator = (cosTheta * (1.0 - k) + k);

  return numerator / denominator;
}

// ACES: Academy Color Encoding System
fn ToneMapACES(hdr: vec3f) -> vec3f {
    let m1 = mat3x3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777,
    );
    let m2 = mat3x3(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602,
    );
    let v = m1 * hdr;
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return clamp(m2 * (a / b), vec3(0.0), vec3(1.0));
}


@fragment
fn fs_main(fragData: VertexOut) -> @location(0) vec4f {
  let albedo = pow(textureSample(albedoMap, sampler2D, fragData.texCoord).rgb, vec3f(2.2));
  var normal = textureSample(normalMap, sampler2D, fragData.texCoord).xyz * 2.0 - 1.0;
  let q1 = dpdx(fragData.pos);
  let q2 = dpdy(fragData.pos);
  let st1 = dpdx(fragData.texCoord);
  let st2 = dpdy(fragData.texCoord);
  let N = normalize(fragData.normal);
  let T = normalize(q1 * st2.y - q2 * st1.y);
  let B = -normalize(cross(N, T));
  let TBN = mat3x3f(T, B, N);
  normal = normalize(TBN * normal);
  let metalness = textureSample(metallicRoughnessMap, sampler2D, fragData.texCoord).b;
  let roughness = textureSample(metallicRoughnessMap, sampler2D, fragData.texCoord).g;
  let emissive = textureSample(emissiveMap, sampler2D, fragData.texCoord).rgb * emissiveFactor;
  let occlusion = textureSample(occlusionMap, sampler2D, fragData.texCoord).r;

  let viewDir = normalize(viewPos - fragData.pos);
  var f0 = vec3f(0.04);
  f0 = mix(f0, albedo, metalness);

  // relflectance equation
  var Lo = vec3f(0.0);

  for(var i = 0; i < 4; i++) {
    let lightDir = normalize(lightPositions[i] - fragData.pos);
    let halfwayDir = normalize(viewDir + lightDir);
    let distance = length(lightPositions[i] - fragData.pos);
    let attenuation = 1.0 / pow(distance, 2.0);
    let radiance = lightColors[i] * attenuation;

    // BRDF: Cook-Torrance
    let D = TrowbridgeReitzGGX(normal, halfwayDir, roughness);
    let F = FresnelSchlickApproximation(max(dot(halfwayDir, viewDir), 0.0), f0, roughness);
    let G = GeometrySmith(normal, viewDir, lightDir, roughness);

    let kS = F;
    var kD = vec3f(1.0) - kS;
    kD *= 1.0 - metalness;

    let numerator = D * F * G;
    let denominator = 4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0) + 0.001;
    let specular = numerator / denominator;

    Lo += (kD * albedo / PI + specular) * radiance * dot(normal, lightDir);
  }

  let kS = FresnelSchlickApproximation(max(dot(normal, viewDir), 0.0), f0, roughness);
  var kD = vec3f(1.0) - kS;
  kD *= 1.0 - metalness;
  let irradiance = textureSample(irradianceMap, samplerCube, normal).rgb;
  let diffuse  = irradiance * albedo;
  let ambient =( kD * diffuse) * occlusion;

  var color = ambient + Lo  + emissive;
  // HDR Tone Mapping
  color = ToneMapACES(color);
  // Gamma Correction
  color = pow(color, vec3f(1.0 / 2.2));

  return vec4f(color, 1.0);
}
