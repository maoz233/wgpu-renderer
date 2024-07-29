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

@fragment
fn fs_main(fragData: VertexOut) -> @location(0) vec4f {
  let texCoord = matrix * fragData.pos;
  // WebGPU uses a right-handed coordinate system, but cubemaps are an exception, using a left-handed coordinate system
  var color = textureSample(cubeTexture, cubeSampler, normalize(texCoord.xyz / texCoord.w)).rgb;

  return vec4f(color, 1.0);
}
