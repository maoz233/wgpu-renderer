declare type glTFObjAny = {
  [key: string]: any;
};

declare type glTFBase = {
  extensions: glTFObjAny;
  extras: glTFObjAny;
  name?: glTFObjAny;
};

declare type glTFAccessor = {
  bufferView?: number;
  componentType: 5120 | 5121 | 5122 | 5123 | 5125 | 5126;
  count: number;
  max?: number[];
  min?: number[];
  normalized?: boolean;
  sparse?: number;
  type: "SCALAR" | "VEC2" | "VEC3" | "VEC4" | "MAT2" | "MAT3" | "MAT4";
} & glTFBase;

declare type glTFAsset = {
  copyright?: string;
  generator?: string;
  version: string;
  minVersion?: string;
} & glTFBase;

declare type glTFBufferView = {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
  target?: 34962 | 34963;
} & glTFBase;

declare type glTFBuffer = {
  byteLength: number;
  uri?: string;
} & glTFBase;

declare type glTFImageBase = {
  bufferView?: number;
  mimeType?: "image/jpeg" | "image/png";
  uri?: string;
} & ({ bufferView: number } | { uri: string });

declare type glTFImage = glTFImageBase & glTFBase;

declare type glTFTextureInfo = {
  index: number;
  texCoord?: number;
} & glTFBase;

declare type glTFMaterial = {
  alphaCutoff?: number;
  alphaMode?: "OPAQUE" | "MASK" | "BLEND";
  doubleSided?: boolean;
  emissiveFactor?: [number, number, number];
  emissiveTexture?: glTFTextureInfo;
  normalTexture?: {
    scale?: number;
  } & glTFTextureInfo &
    glTFBase;
  occlusionTexture?: {
    strength?: number;
  } & glTFTextureInfo &
    glTFBase;
  pbrMetallicRoughness?: {
    baseColorFactor?: [number, number, number, number];
    baseColorTexture?: glTFTextureInfo;
    metallicFactor?: number;
    metallicRoughnessTexture?: glTFTextureInfo;
    roughnessFactor?: number;
  } & glTFBase;
} & glTFBase;

declare type glTFPrimitive = {
  attributes: {
    NORMAL: number;
    POSITION: number;
    TEXCOORD_0: number;
  };
  indices?: number;
  material?: number;
  mode?: number;
  targets?: number[];
} & glTFBase;

declare type glTFMesh = {
  primitives: glTFPrimitive[];
} & glTFBase;

declare type glTFNodeBase = {
  camera?: number;
  children?: number[];
  mesh?: number;
  skin?: number;
  weights?: number[];
} & (
  | {
      rotation?: [number, number, number, number];
      scale?: [number, number, number];
      translation: [number, number, number];
    }
  | {
      matrix?: [
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
      ];
    }
);

declare type glTFNode = glTFNodeBase & glTFBase;

declare type glTFSampler = {
  magFilter: 9728 | 9729;
  minFilter: 9728 | 9729 | 9984 | 9985 | 9986 | 9987;
  wrapS: 33071 | 33648 | 10497;
  wrapT: 33071 | 33648 | 10497;
} & glTFBase;

declare type glTFScene = {
  nodes?: number[];
} & glTFBase;

declare type glTFTexture = {
  sampler?: number;
  source?: number;
} & glTFBase;

export type glTF = {
  accessors?: glTFAccessor[];
  asset: glTFAsset;
  bufferViews?: glTFBufferView[];
  buffers?: glTFBuffer[];
  extensionsUsed?: string[];
  extensionsRequired?: string[];
  images?: glTFImage[];
  materials?: glTFMaterial[];
  meshes?: glTFMesh[];
  nodes?: glTFNode[];
  samplers?: glTFSampler[];
  scene?: number;
  scenes?: glTFScene[];
  textures?: glTFTexture[];
} & glTFBase;
