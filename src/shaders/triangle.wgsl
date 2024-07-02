struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f
}

struct OurStruct {
  scale : vec2f,
  offset : vec2f,
};

@group(0) @binding(0) var<uniform> ourStruct : OurStruct;

@vertex
fn vertex_main(@location(0) position : vec4f,
@location(1) color : vec4f) -> VertexOut
{
  var output : VertexOut;
  output.position = vec4f(position.xy * ourStruct.scale + ourStruct.offset, position.z, position.w);
  output.color = color;
  return output;
}

@fragment
fn fragment_main(fragData : VertexOut) -> @location(0) vec4f
{
  return fragData.color;
}
