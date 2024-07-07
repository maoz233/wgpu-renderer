struct VertexInput {
  @location(0) position : vec4f,
  @location(1) texCoord : vec2f
};

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) texCoord : vec2f
};


@group(0) @binding(0) var<uniform> mvp : mat4x4f;
@group(1) @binding(0) var myTexture : texture_2d<f32>;
@group(2) @binding(0) var mySampler : sampler;

@vertex
fn vs_main(vertData : VertexInput) -> VertexOut {
  var output : VertexOut;

  output.position = mvp * vertData.position;
  output.texCoord = vertData.texCoord;

  return output;
}

@fragment
fn fs_main(fragData : VertexOut) -> @location(0) vec4f {
  return textureSample(myTexture, mySampler, fragData.texCoord);
}
