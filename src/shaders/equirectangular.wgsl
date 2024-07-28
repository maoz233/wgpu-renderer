const PI: f32 = 3.14159265359;
const invAtan = vec2(0.1591, 0.3183);

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
    if gid.x >= u32(textureDimensions(dst).x) {
        return;
    }

    let faces: array<Face, 6> = array(
        // FACES +X
        Face(
            vec3(1.0, 0.0, 0.0),  // forward
            vec3(0.0, 1.0, 0.0),  // up
            vec3(0.0, 0.0, -1.0), // right
        ),
        // FACES -X
        Face (
            vec3(-1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0),
        ),
        // FACES +Y
        Face (
            vec3(0.0, -1.0, 0.0),
            vec3(0.0, 0.0, 1.0),
            vec3(1.0, 0.0, 0.0),
        ),
        // FACES -Y
        Face (
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, -1.0),
            vec3(1.0, 0.0, 0.0),
        ),
        // FACES +Z
        Face (
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 1.0, 0.0),
            vec3(1.0, 0.0, 0.0),
        ),
        // FACES -Z
        Face (
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, 0.0, 0.0),
        ),
    );

    // Get texture coords relative to cubemap face
    let dstDimensions = vec2<f32>(textureDimensions(dst));
    let cubeUV = vec2<f32>(gid.xy) / dstDimensions * 2.0 - 1.0;

    // Get spherical coordinate from cubeUV
    let face = faces[gid.z];
    let spherical = normalize(face.forward + face.right * cubeUV.x + face.up * cubeUV.y);

    // Get coordinate on the equirectangular texture
    let eqUV = vec2(atan2(spherical.z, spherical.x), asin(spherical.y)) * invAtan + 0.5;
    let eqPixel = vec2<i32>(eqUV * vec2<f32>(textureDimensions(src)));

    // We use textureLoad() as textureSample() is not allowed in compute shaders
    let sample = textureLoad(src, eqPixel, 0);

    textureStore(dst, gid.xy, gid.z, sample);
}
