import type {
  glTF,
  glTFBuffer,
  glTFMesh,
  glTFNode,
  glTFPrimitive,
} from "@/types/glTF";
import { calculateByteLength } from "./utils";
import { mat4, quat, vec3, type Mat4 } from "wgpu-matrix";

export const ComponentType = {
  5120: "sint8",
  5121: "uint8",
  5122: "sint16",
  5123: "uint16",
  5125: "uint32",
  5126: "float32",
};

export const ComponentCount = {
  SCALAR: "",
  VEC2: "x2",
  VEC3: "x3",
  VEC4: "x4",
  MAT2: "x8",
  MAT3: "x12",
  MAT4: "x16",
};

export type Layout = {
  format: GPUVertexFormat;
  offset: number;
};

export type Geometry = {
  model: Mat4;
  vertices: Float32Array;
  arrayStride: number;
  position: Layout;
  normal: Layout;
  texCoord: Layout;
  indices: Uint16Array;
  textures: {
    baseColorURI: string;
    metallicRoughnessURI: string;
    normalURI: string;
    emissive: [number, number, number];
    emissiveURI: string;
    occlusionURI: string;
  };
};

export default class glTFLoader {
  private pathname: string;

  private gltf: glTF;
  private buffers: ArrayBuffer[];
  private geometries: Geometry[];

  public constructor() {
    this.loadBinBuffer = this.loadBinBuffer.bind(this);
  }

  public async load(url: string): Promise<void> {
    const index = url.lastIndexOf("/");
    this.pathname = url.slice(0, index + 1);

    await this.loadJSON(url);
    await this.loadBinBuffers();
    this.generateGeometries();
  }

  public getGeometries() {
    return this.geometries;
  }

  private async loadJSON(url: string): Promise<void> {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch .gltf file: ${url}`);
    }

    this.gltf = await res.json();
    if ("2.0" !== this.gltf.asset.version) {
      throw new Error(`Unsupported glTF version: ${this.gltf.asset.version}`);
    }
  }

  private async loadBinBuffers(): Promise<void> {
    this.buffers = await Promise.all(this.gltf.buffers.map(this.loadBinBuffer));
  }

  private async loadBinBuffer(buffer: glTFBuffer): Promise<ArrayBuffer> {
    const res = await fetch(this.pathname + buffer.uri);
    if (!res.ok) {
      throw new Error(`Failed to fetch .bin file: ${buffer.uri}`);
    }

    const arrayBuffer = await res.arrayBuffer();
    if (arrayBuffer.byteLength !== buffer.byteLength) {
      throw new Error("Wrong buffer byte length.");
    }

    return arrayBuffer;
  }

  private generateGeometries(): void {
    const nodes = this.gltf.nodes;
    this.geometries = new Array<Geometry>(nodes.length);

    nodes.forEach((node: glTFNode, index: number): void => {
      let model: Mat4;
      if ("matrix" in node) {
        model = mat4.create(...node.matrix);
      } else {
        model = mat4.identity();

        if ("scale" in node) {
          model = mat4.scale(model, vec3.create(...node.scale));
        }

        if ("translation" in node) {
          model = mat4.translate(model, vec3.create(...node.translation));
        }

        if ("rotation" in node) {
          const q = quat.fromValues(...node.rotation);
          model = mat4.mul(model, mat4.fromQuat(q));
        }
      }

      const mesh = this.gltf.meshes[node.mesh];
      mesh.primitives.map((primitive: glTFPrimitive): void => {
        if ("mode" in primitive && 4 !== primitive.mode) {
          return;
        }

        // positions
        const positionAccessor =
          this.gltf.accessors[primitive.attributes.POSITION];
        const positionType = (ComponentType[positionAccessor.componentType] +
          ComponentCount[positionAccessor.type]) as GPUVertexFormat;
        const positionOffset = calculateByteLength();
        const positionBufferview =
          this.gltf.bufferViews[positionAccessor.bufferView];
        const positionBuffer = new Float32Array(
          this.buffers[positionBufferview.buffer],
          positionBufferview.byteOffset,
          positionBufferview.byteLength / Float32Array.BYTES_PER_ELEMENT
        );

        // texture coordinates
        const texCoordAccessor =
          this.gltf.accessors[primitive.attributes.TEXCOORD_0];
        const texCoordType = (ComponentType[texCoordAccessor.componentType] +
          ComponentCount[texCoordAccessor.type]) as GPUVertexFormat;
        const texCoordOffset = calculateByteLength(positionType);
        const texCoordBufferview =
          this.gltf.bufferViews[texCoordAccessor.bufferView];
        const texCoordlBuffer = new Float32Array(
          this.buffers[texCoordBufferview.buffer],
          texCoordBufferview.byteOffset,
          texCoordBufferview.byteLength / Float32Array.BYTES_PER_ELEMENT
        );

        // normals
        const normalAccessor = this.gltf.accessors[primitive.attributes.NORMAL];
        const normalType = (ComponentType[normalAccessor.componentType] +
          ComponentCount[normalAccessor.type]) as GPUVertexFormat;
        const normalOffset =
          calculateByteLength(positionType) + calculateByteLength(texCoordType);
        const normalBufferview =
          this.gltf.bufferViews[normalAccessor.bufferView];
        const normalBuffer = new Float32Array(
          this.buffers[normalBufferview.buffer],
          normalBufferview.byteOffset,
          normalBufferview.byteLength / Float32Array.BYTES_PER_ELEMENT
        );

        const arrayStride =
          calculateByteLength(positionType) +
          calculateByteLength(texCoordType) +
          calculateByteLength(normalType);

        // vertices array
        const vertices = new Float32Array(
          (positionAccessor.count * arrayStride) /
            Float32Array.BYTES_PER_ELEMENT
        );
        let pOffset = 0;
        let nOffset = 0;
        let tOffset = 0;
        let stride = 8;
        for (let i = 0; i < positionAccessor.count; ++i) {
          const position = [
            positionBuffer[pOffset++],
            positionBuffer[pOffset++],
            positionBuffer[pOffset++],
          ];
          vertices.set(position, stride * i);

          const texCoord = [
            texCoordlBuffer[tOffset++],
            texCoordlBuffer[tOffset++],
          ];
          vertices.set(texCoord, stride * i + 3);

          const normal = [
            normalBuffer[nOffset++],
            normalBuffer[nOffset++],
            normalBuffer[nOffset++],
          ];
          vertices.set(normal, stride * i + 5);
        }

        // indices
        const indexAccessor = this.gltf.accessors[primitive.indices];
        const indexBufferView = this.gltf.bufferViews[indexAccessor.bufferView];
        const indexBuffer = this.buffers[indexBufferView.buffer];

        // indices array
        const indices = new Uint16Array(
          indexBuffer,
          indexBufferView.byteOffset,
          indexBufferView.byteLength / Uint16Array.BYTES_PER_ELEMENT
        );

        const material = this.gltf.materials[primitive.material];
        // base color texture
        const bcTerxtureIndex =
          material.pbrMetallicRoughness.baseColorTexture.index;
        const bcImageIndex = this.gltf.textures[bcTerxtureIndex].source;
        const baseColorURI = this.pathname + this.gltf.images[bcImageIndex].uri;
        // metallic roughness texture
        const mrTextureIndex =
          material.pbrMetallicRoughness.metallicRoughnessTexture.index;
        const mrImageIndex = this.gltf.textures[mrTextureIndex].source;
        const metallicRoughnessURI =
          this.pathname + this.gltf.images[mrImageIndex].uri;
        // normal texture
        const nTextureIndex = material.normalTexture.index;
        const nImageIndex = this.gltf.textures[nTextureIndex].source;
        const normalURI = this.pathname + this.gltf.images[nImageIndex].uri;
        // emissive factor texture
        const emissive = material.emissiveFactor;
        const eTextureIndex = material.emissiveTexture.index;
        const eImageIndex = this.gltf.textures[eTextureIndex].source;
        const emissiveURI = this.pathname + this.gltf.images[eImageIndex].uri;
        // occlusion texture
        const oTextureIndex = material.occlusionTexture.index;
        const oImageIndex = this.gltf.textures[oTextureIndex].source;
        const occlusionURI = this.pathname + this.gltf.images[oImageIndex].uri;

        this.geometries[index] = {
          model,
          vertices,
          arrayStride,
          position: {
            format: positionType,
            offset: positionOffset,
          },
          normal: {
            format: normalType,
            offset: normalOffset,
          },
          texCoord: {
            format: texCoordType,
            offset: texCoordOffset,
          },
          indices,
          textures: {
            baseColorURI,
            metallicRoughnessURI,
            normalURI,
            emissive,
            emissiveURI,
            occlusionURI,
          },
        };
      });
    });
  }
}
