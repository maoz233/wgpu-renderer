struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) pos: vec4f,
};


@group(0) @binding(0) var<uniform> matrix: mat4x4f;
@group(0) @binding(1) var cubeSampler: sampler;
@group(1) @binding(0) var cubeTexture: texture_cube<f32>;


@vertex
fn vs_main(@builtin(vertex_index)  vertexIndex: u32) -> VertexOut {
  let positions = array(vec2f(-1, 3), vec2f(-1,-1), vec2f(3, -1));

  var output: VertexOut;

  output.position = vec4f(positions[vertexIndex], 1.0, 1.0);
  output.pos = output.position;

  return output;
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
    return clamp(m2 * (a / b), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs_main(fragData: VertexOut) -> @location(0) vec4f {
  let texCoord = matrix * fragData.pos;
  // WebGPU uses a right-handed coordinate system, but cubemaps are an exception, using a left-handed coordinate system
  var color = textureSample(cubeTexture, cubeSampler, normalize(texCoord.xyz / texCoord.w) * vec3f(1, 1, -1)).rgb;
  // HDR Tone Mapping
  color = ToneMapACES(color);
  // Gamma Correction
  color = pow(color, vec3f(1.0/2.2));

  return vec4f(color, 1.0);
}
