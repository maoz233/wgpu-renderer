const PI: f32 = 3.14159265359;
const invAtan = vec2f(0.1591, 0.3183);

struct Face {
    forward: vec3f,
    up: vec3f,
    right: vec3f,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba32float, write>;

@compute @workgroup_size(16, 16, 1)
fn compute_main(@builtin(global_invocation_id) gid: vec3u) {
    // If texture size is not divisible by 32, we need to make sure we don't try to write to pixels that don't exist.
    if gid.x >= u32(textureDimensions(dst).x) ||  gid.y >= u32(textureDimensions(dst).y) {
        return;
    }

    let faces: array<Face, 6> = array(
        // FACES +X
        Face(
            vec3f(1.0, 0.0, 0.0),  // forward
            vec3f(0.0, 1.0, 0.0),  // up
            vec3f(0.0, 0.0, -1.0), // right
        ),
        // FACES -X
        Face (
            vec3f(-1.0, 0.0, 0.0),
            vec3f(0.0, 1.0, 0.0),
            vec3f(0.0, 0.0, 1.0),
        ),
        // FACES +Y
        Face (
            vec3f(0.0, -1.0, 0.0),
            vec3f(0.0, 0.0, 1.0),
            vec3f(1.0, 0.0, 0.0),
        ),
        // FACES -Y
        Face (
            vec3f(0.0, 1.0, 0.0),
            vec3f(0.0, 0.0, -1.0),
            vec3f(1.0, 0.0, 0.0),
        ),
        // FACES +Z
        Face (
            vec3f(0.0, 0.0, 1.0),
            vec3f(0.0, 1.0, 0.0),
            vec3f(1.0, 0.0, 0.0),
        ),
        // FACES -Z
        Face (
            vec3f(0.0, 0.0, -1.0),
            vec3f(0.0, 1.0, 0.0),
            vec3f(-1.0, 0.0, 0.0),
        ),
    );

    // Get texture coords relative to cubemap face
    let dstDimensions = vec2f(textureDimensions(dst));
    let cubeUV = vec2f(gid.xy) / dstDimensions * 2.0 - 1.0;

    // Get spherical coordinate from cubeUV
    let face = faces[gid.z];
    let spherical = normalize(face.forward + face.right * cubeUV.x + face.up * cubeUV.y);

    // tangent space calculation from origin point
    var up    = vec3f(0.0, 1.0, 0.0);
    let right = normalize(cross(up, spherical));
    up = normalize(cross(spherical, right));

    var irradiance = vec3f(0.0);
    let sampleDelta = 0.025;
    var nrSamples = 0;
    for(var phi = 0.0; phi < 2.0 * PI; phi += sampleDelta){
        for(var theta = 0.0; theta < 0.5 * PI; theta += sampleDelta){
            // spherical to cartesian (in tangent space)
            let tangentSample = vec3f(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
            // tangent space to world
            let sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * spherical;
            let eqUV = vec2f(atan2(sampleVec.z, sampleVec.x), asin(sampleVec.y)) * invAtan + 0.5;
            let eqPixel = vec2i(eqUV * vec2f(textureDimensions(src)));
            // We use textureLoad() as textureSample() is not allowed in compute shaders
            irradiance += textureLoad(src, eqPixel, 0).rgb * cos(theta) * sin(theta);
            nrSamples++;
        }
    }
    irradiance = PI * irradiance * (1.0 / f32(nrSamples));

    textureStore(dst, gid.xy, gid.z, vec4f(irradiance, 1.0));
}
