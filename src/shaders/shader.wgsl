struct VertexInput {
  @location(0) position : vec3f,
  @location(1) texCoord : vec2f,
  @location(2) normal : vec3f
};

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) texCoord : vec2f,
  @location(1) normal : vec3f
};

struct Transform {
  model : mat4x4f,
  view : mat4x4f,
  projection : mat4x4f,
};

@group(0) @binding(0) var<uniform> transform : Transform;
@group(1) @binding(0) var containerTexture : texture_2d<f32>;
@group(1) @binding(1) var faceTexture : texture_2d<f32>;
@group(2) @binding(0) var mySampler : sampler;

@vertex
fn vs_main(vertData : VertexInput) -> VertexOut {
  var output : VertexOut;

  output.position = transform.projection * transform.view * transform.model * vec4f(vertData.position, 1.0);
  output.texCoord = vertData.texCoord;
  output.normal = (transform.model * vec4f(vertData.normal, 0.0)).xyz;

  return output;
}

@fragment
fn fs_main(fragData : VertexOut) -> @location(0) vec4f {
  var containerColor = textureSample(containerTexture, mySampler, fragData.texCoord);
  var faceColor = textureSample(faceTexture, mySampler, fragData.texCoord);
  var textureColor = mix(containerColor, faceColor, 0.2);

  return textureColor;
}
