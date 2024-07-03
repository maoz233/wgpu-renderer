struct VertexInput {
  @location(0) position: vec4f,
  @location(1) color: vec3f
};

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f
};

struct Transform {
  mvp: mat4x4f,
};

@group(0) @binding(0) var<uniform> transform: Transform;

@vertex
fn vs_main(vertData: VertexInput) -> VertexOut {
  var output: VertexOut;

  output.position = transform.mvp * vertData.position;
  output.color = vec4f(vertData.color, 1.0);

  return output;
}

@fragment
fn fs_main(fragData: VertexOut) -> @location(0) vec4f {
  return fragData.color;
}
