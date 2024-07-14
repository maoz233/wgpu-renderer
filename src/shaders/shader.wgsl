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

struct Light {
  direction: vec3f,
  ambient: vec3f,
  diffuse: vec3f,
  specular: vec3f
};

@group(0) @binding(0) var<uniform> transform: Transform;
@group(0) @binding(1) var<uniform> viewPos: vec3f;
@group(0) @binding(2) var<uniform> light: Light;
@group(1) @binding(0) var<uniform> materialShininess: f32;
@group(1) @binding(1) var materialDiffuse: texture_2d<f32>;
@group(1) @binding(2) var materialSpecular: texture_2d<f32>;
@group(1) @binding(3) var materialCube: texture_cube<f32>;
@group(2) @binding(0) var materialSampler: sampler;

@vertex
fn vs_main(vertData: VertexInput) -> VertexOut {
  var output: VertexOut;

  output.position = transform.projection * transform.view * transform.model * vec4f(vertData.position, 1.0);
  output.texCoord = vertData.texCoord;
  output.normal = (transform.normal * vec4f(vertData.normal, 0.0)).xyz;
  output.pos = (transform.normal * vec4f(vertData.position, 0.0)).xyz;

  return output;
}

@fragment
fn fs_main(fragData: VertexOut) -> @location(0) vec4f {
  // ambient
  var ambient = light.ambient * textureSample(materialDiffuse, materialSampler, fragData.texCoord).rgb;

  // diffuse
  var normal = normalize(fragData.normal);
  var lightDir = normalize(-light.direction);
  var diff = max(dot(normal, lightDir), 0.0);
  var diffuse = light.diffuse * diff * textureSample(materialDiffuse, materialSampler, fragData.texCoord).rgb;

  // specular
  var viewDir = normalize(viewPos - fragData.pos);
  var reflectDir = reflect(light.direction, normal);
  var spec = pow(max(dot(viewDir, reflectDir), 0.0), materialShininess);
  var specular = light.specular * spec * textureSample(materialSpecular, materialSampler, fragData.texCoord).rgb;  

  var result = ambient + diffuse + specular;

  viewDir = normalize(fragData.pos -viewPos);
  reflectDir = reflect(viewDir, normal);
  result = textureSample(materialCube, materialSampler, reflectDir * vec3f(1, 1, -1)).rgb;

  return vec4f(result, 1.0);
}
