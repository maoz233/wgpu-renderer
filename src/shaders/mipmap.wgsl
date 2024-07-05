
struct VertexInput {
  @location(0) position: vec4f,
  @location(1) texCoord: vec2f,
};

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) texCoord: vec2f,
};

@vertex
fn vs_main(vertData: VertexInput) -> VertexOut {
  var out: VertexOut;
  out.position = vec4f(vertData.position);
  out.texCoord = vertData.texCoord;
  return out;
}

@group(0) @binding(0) var mySampler: sampler;
@group(0) @binding(1) var myTexture: texture_2d<f32>;

@fragment 
fn fs_main(fragData: VertexOut) -> @location(0) vec4f {
  return textureSample(myTexture, mySampler, fragData.texCoord);
}
