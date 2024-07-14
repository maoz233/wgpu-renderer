import shaderCode from "@/shaders/shader.wgsl";
import mipmapShaderCode from "@/shaders/mipmap.wgsl";
import skyboxShaderCode from "@/shaders/skybox.wgsl";
import { mat4, utils, vec3 } from "wgpu-matrix";
import { GUI, GUIController } from "dat.gui";
import {
  loadImageBitmap,
  calculateMipLevelCount,
  RollingAverage,
} from "@/utils";
import Camera from "./camera";

const fpsAvg = new RollingAverage();
const cpuTimeAvg = new RollingAverage();
const gpuTimeAvg = new RollingAverage();

export default class Renderer {
  private camera: Camera;

  private adapter: GPUAdapter;
  private device: GPUDevice;
  private hasTimestamp: boolean;
  private querySet: GPUQuerySet;
  private resolveBuffer: GPUBuffer;
  private resultBuffer: GPUBuffer;
  private canvas: HTMLCanvasElement;
  private context: GPUCanvasContext;
  private depthTexture: GPUTexture;
  private depthTextureView: GPUTextureView;
  private presentationFormat: GPUTextureFormat;

  private renderPipeline: GPURenderPipeline;
  private vertexBuffer: GPUBuffer;
  private indexBuffer: GPUBuffer;
  private bindGroups: Array<Array<GPUBindGroup>>;
  private transformUniformBuffer: GPUBuffer;
  private viewPositionUniformBuffer: GPUBuffer;
  private lightUniformBuffer: GPUBuffer;
  private materialShininessUniformBuffer: GPUBuffer;

  private skyboxRenderPipeline: GPURenderPipeline;
  private skyboxBindGroup: GPUBindGroup;
  private skyboxUniformBuffer: GPUBuffer;

  private current: number;

  private fpsController: GUIController;
  private cpuTimeController: GUIController;
  private gpuTimeController: GUIController;
  private mipmapsController: GUIController;
  private addressModeUController: GUIController;
  private addressModeVController: GUIController;
  private magFilterController: GUIController;
  private minFilterController: GUIController;

  public constructor() {
    this.camera = new Camera();

    this.bindGroups = new Array<Array<GPUBindGroup>>(
      new Array<GPUBindGroup>(),
      new Array<GPUBindGroup>(),
      new Array<GPUBindGroup>()
    );

    this.current = 0;

    this.drawFrame = this.drawFrame.bind(this);
  }

  public async render(): Promise<void> {
    await this.init();
    this.run();
  }

  private async init(): Promise<void> {
    this.checkWebGPUSupport();
    await this.requestAdapter();
    await this.requestDevice();
    this.getCanvas();
    this.configContext();
    this.createDepthTexture();

    this.createRenderPipeline();
    this.createVertexBuffer();
    this.createIndexBuffer();
    this.createUniformBuffer();
    await this.createTexture();

    this.createSkyboxRenderPipeline();
    await this.createSkyboxUniformBuffer();

    this.initGUI();
  }

  private checkWebGPUSupport(): void {
    if (!navigator.gpu) {
      throw Error("WebGPU not supported.");
    }
  }

  private async requestAdapter(): Promise<void> {
    this.adapter = await navigator.gpu?.requestAdapter();
    if (!this.adapter) {
      throw Error("Failed to request WebGPU adapter.");
    }
  }

  private async requestDevice(): Promise<void> {
    this.hasTimestamp = this.adapter?.features.has("timestamp-query");
    const requiredFeatures: GPUFeatureName[] = this.hasTimestamp
      ? ["timestamp-query"]
      : [];

    this.device = await this.adapter?.requestDevice({
      label: `GPU Device ${
        this.hasTimestamp && "with feature: timestamp-query"
      }`,
      requiredFeatures,
    });
    if (!this.device) {
      throw Error("Failed to request WebGPU device.");
    }

    if (this.hasTimestamp) {
      this.querySet = this.device.createQuerySet({
        label: "GPU Query Set with Type: timestamp",
        type: "timestamp",
        count: 2,
      });

      this.resolveBuffer = this.device.createBuffer({
        label: "GPU Buffer: Resolve",
        size: this.querySet.count * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });

      this.resultBuffer = this.device.createBuffer({
        label: "GPU Buffer: Result",
        size: this.resolveBuffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }

    this.device.lost.then((info) => {
      if (info.reason !== "destroyed") {
        throw new Error(`WebGPU device was lost: ${info.message}`);
      }
    });
  }

  private getCanvas(): void {
    this.canvas = document.querySelector("canvas");
    if (!this.canvas) {
      throw Error("Failed to find canvas element.");
    }

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const canvas = entry.target as HTMLCanvasElement;
        const width = entry.contentBoxSize[0].inlineSize;
        const height = entry.contentBoxSize[0].blockSize;
        canvas.width = Math.max(
          1,
          Math.min(width, this.device.limits.maxTextureDimension2D)
        );
        canvas.height = Math.max(
          1,
          Math.min(height, this.device.limits.maxTextureDimension2D)
        );
        this.depthTexture.destroy();
        this.createDepthTexture();
      }
    });
    observer.observe(this.canvas);
  }

  private configContext(): void {
    this.context = this.canvas.getContext("webgpu");
    if (!this.context) {
      throw Error("Failed to get WebGPU context from canvas.");
    }

    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: "premultiplied",
    });
  }

  private async createDepthTexture(): Promise<void> {
    this.depthTexture = this.device.createTexture({
      label: "GPU Texture: Depth Texture",
      size: [this.canvas.width, this.canvas.height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.depthTextureView = this.depthTexture.createView({
      label: "GPU Texture View: Depth Texture View",
    });
  }

  private createRenderPipeline(): void {
    const shaderModule = this.createShaderModule(
      "GPU Shader Module",
      shaderCode
    );

    this.renderPipeline = this.device.createRenderPipeline({
      label: "GPU Render Pipeline",
      layout: "auto",
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: (3 + 2 + 3) * Float32Array.BYTES_PER_ELEMENT,
            stepMode: "vertex" as GPUVertexStepMode,
            attributes: [
              {
                format: "float32x3" as GPUVertexFormat,
                offset: 0,
                shaderLocation: 0,
              },
              {
                format: "float32x2" as GPUVertexFormat,
                offset: 3 * Float32Array.BYTES_PER_ELEMENT,
                shaderLocation: 1,
              },
              {
                format: "float32x3" as GPUVertexFormat,
                offset: (3 + 2) * Float32Array.BYTES_PER_ELEMENT,
                shaderLocation: 2,
              },
            ],
          },
        ],
      },
      primitive: {
        topology: "triangle-list",
        cullMode: "back",
      },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [
          {
            format: this.presentationFormat,
          },
        ],
      },
    });
  }

  private createVertexBuffer(): void {
    // prettier-ignore
    const vertices = new Float32Array([
      -1.0,  1.0,  1.0, 0.0, 0.0,  0.0,  0.0,  1.0, // 0
      -1.0,  1.0,  1.0, 1.0, 0.0, -1.0,  0.0,  0.0, // 1
      -1.0,  1.0,  1.0, 0.0, 1.0,  0.0,  1.0,  0.0, // 2
      -1.0, -1.0,  1.0, 0.0, 1.0,  0.0,  0.0,  1.0, // 3
      -1.0, -1.0,  1.0, 1.0, 1.0, -1.0,  0.0,  0.0, // 4
      -1.0, -1.0,  1.0, 0.0, 0.0,  0.0, -1.0,  0.0, // 5
       1.0,  1.0,  1.0, 1.0, 0.0,  0.0,  0.0,  1.0, // 6
       1.0,  1.0,  1.0, 0.0, 0.0,  1.0,  0.0,  0.0, // 7
       1.0,  1.0,  1.0, 1.0, 1.0,  0.0,  1.0,  0.0, // 8
       1.0, -1.0,  1.0, 1.0, 1.0,  0.0,  0.0,  1.0, // 9
       1.0, -1.0,  1.0, 0.0, 1.0,  1.0,  0.0,  0.0, // 10
       1.0, -1.0,  1.0, 1.0, 0.0,  0.0, -1.0,  0.0, // 11
      -1.0,  1.0, -1.0, 1.0, 0.0,  0.0,  0.0, -1.0, // 12
      -1.0,  1.0, -1.0, 0.0, 0.0, -1.0,  0.0,  0.0, // 13
      -1.0,  1.0, -1.0, 0.0, 0.0,  0.0,  1.0,  0.0, // 14
      -1.0, -1.0, -1.0, 1.0, 1.0,  0.0,  0.0, -1.0, // 15
      -1.0, -1.0, -1.0, 0.0, 1.0, -1.0,  0.0,  0.0, // 16
      -1.0, -1.0, -1.0, 0.0, 1.0,  0.0, -1.0,  0.0, // 17
       1.0,  1.0, -1.0, 0.0, 0.0,  0.0,  0.0, -1.0, // 18
       1.0,  1.0, -1.0, 1.0, 0.0,  1.0,  0.0,  0.0, // 19
       1.0,  1.0, -1.0, 1.0, 0.0,  0.0,  1.0,  0.0, // 20
       1.0, -1.0, -1.0, 0.0, 1.0,  0.0,  0.0, -1.0, // 21
       1.0, -1.0, -1.0, 1.0, 1.0,  1.0,  0.0,  0.0, // 22
       1.0, -1.0, -1.0, 1.0, 1.0,  0.0, -1.0,  0.0, // 23
    ]);

    this.vertexBuffer = this.createBuffer(
      "GPU Buffer: Vertex",
      vertices.byteLength,
      GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    );

    this.device.queue.writeBuffer(
      this.vertexBuffer,
      0,
      vertices.buffer,
      0,
      vertices.byteLength
    );
  }

  private createIndexBuffer(): void {
    // prettier-ignore
    const indices = new Uint32Array([
       0,  3,  6,  6,  3,  9,
       7, 10, 19, 19, 10, 22,
      18, 21, 12, 12, 21, 15,
      13, 16,  1,  1, 16,  4,
      14,  2, 20, 20,  2,  8,
       5, 17, 11, 11, 17, 23,
    ]);

    this.indexBuffer = this.createBuffer(
      "GPU Buffer: Index",
      indices.byteLength,
      GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
    );

    this.device.queue.writeBuffer(
      this.indexBuffer,
      0,
      indices.buffer,
      0,
      indices.byteLength
    );
  }

  private createUniformBuffer(): void {
    this.transformUniformBuffer = this.createBuffer(
      `GPU Uniform Buffer: Transform`,
      (4 * 4 + 4 * 4 + 4 * 4 + 4 * 4) * Float32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    this.viewPositionUniformBuffer = this.createBuffer(
      `GPU Uniform Buffer: View Position`,
      4 * Float32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    this.lightUniformBuffer = this.createBuffer(
      `GPU Uniform Buffer: Light`,
      (4 + 4 + 4 + 4) * Float32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    const bindGroup = this.device.createBindGroup({
      label: `GPU Bind Group 0: Transform, View Position, Light`,
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.transformUniformBuffer,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.viewPositionUniformBuffer,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.lightUniformBuffer,
          },
        },
      ],
    });
    this.bindGroups[0].push(bindGroup);
  }

  private async createTexture(): Promise<void> {
    const contianerImageBitmap = await loadImageBitmap("images/container.png");
    const contianerSpecularImageBitmap = await loadImageBitmap(
      "images/container_specular.png"
    );
    const cubeImageBitmaps = await Promise.all(
      [
        "images/LeadenhallMarket/pos-x.jpg",
        "images/LeadenhallMarket/neg-x.jpg",
        "images/LeadenhallMarket/pos-y.jpg",
        "images/LeadenhallMarket/neg-y.jpg",
        "images/LeadenhallMarket/pos-z.jpg",
        "images/LeadenhallMarket/neg-z.jpg",
      ].map(loadImageBitmap)
    );

    this.materialShininessUniformBuffer = this.createBuffer(
      `GPU Uniform Buffer: Material Shininess`,
      4 * Float32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    for (let i = 0; i < 2; ++i) {
      const contianerMipLevelCount = i
        ? calculateMipLevelCount(
            contianerImageBitmap.width,
            contianerImageBitmap.height
          )
        : 1;

      const contianerTexture = this.createTexture2DFromSource(
        `GPU Texture: Contianer ${i}`,
        contianerImageBitmap,
        contianerMipLevelCount
      );

      const contianerTextureView = contianerTexture.createView({
        label: `GPU Texture View: Contianer ${i}`,
      });

      const containerSpecularMipLevelCount = i
        ? calculateMipLevelCount(
            contianerSpecularImageBitmap.width,
            contianerSpecularImageBitmap.height
          )
        : 1;

      const contianerSpecularTexture = this.createTexture2DFromSource(
        `GPU Texture: Contianer Specular ${i}`,
        contianerSpecularImageBitmap,
        containerSpecularMipLevelCount
      );

      const contianerSpecularTextureView = contianerSpecularTexture.createView({
        label: `GPU Texture View: Contianer Specular ${i}`,
      });

      const cubeMipLevelCount = i
        ? calculateMipLevelCount(
            cubeImageBitmaps[0].width,
            cubeImageBitmaps[0].height
          )
        : 1;
      const cubeTexture = this.createTextureCubeFromSources(
        `GPU Texture: Cube ${i}`,
        cubeImageBitmaps,
        cubeMipLevelCount
      );
      const cubeTextureView = cubeTexture.createView({
        label: `GPU Texture View: Cube ${i}`,
        dimension: "cube",
      });

      const bindGroup = this.device.createBindGroup({
        label: `GPU Bind Group 1: Container, Contianer Specular, Cube,  ${i}`,
        layout: this.renderPipeline.getBindGroupLayout(1),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: this.materialShininessUniformBuffer,
            },
          },
          {
            binding: 1,
            resource: contianerTextureView,
          },
          {
            binding: 2,
            resource: contianerSpecularTextureView,
          },
          {
            binding: 3,
            resource: cubeTextureView,
          },
        ],
      });
      this.bindGroups[1].push(bindGroup);
    }

    for (let i = 0; i < 16; ++i) {
      const sampler = this.device.createSampler({
        label: `GPU Sampler: Sampler ${i}`,
        addressModeU: i & 1 ? "repeat" : "clamp-to-edge",
        addressModeV: i & 2 ? "repeat" : "clamp-to-edge",
        magFilter: i & 4 ? "linear" : "nearest",
        minFilter: i & 8 ? "linear" : "nearest",
      });

      const bindGroup = this.device.createBindGroup({
        label: `GPU Bind Group 2: Sampler ${i}`,
        layout: this.renderPipeline.getBindGroupLayout(2),
        entries: [
          {
            binding: 0,
            resource: sampler,
          },
        ],
      });
      this.bindGroups[2].push(bindGroup);
    }
  }

  private createSkyboxRenderPipeline(): void {
    const skyboxShaderModule = this.createShaderModule(
      "GPU Shader Module: Skybox",
      skyboxShaderCode
    );

    this.skyboxRenderPipeline = this.device.createRenderPipeline({
      label: "GPU Render Pipeline: Skybox",
      layout: "auto",
      vertex: {
        module: skyboxShaderModule,
        entryPoint: "vs_main",
      },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
      },
      fragment: {
        module: skyboxShaderModule,
        entryPoint: "fs_main",
        targets: [
          {
            format: this.presentationFormat,
          },
        ],
      },
    });
  }

  private async createSkyboxUniformBuffer(): Promise<void> {
    this.skyboxUniformBuffer = this.createBuffer(
      `GPU Uniform Buffer: Skybox`,
      4 * 4 * Float32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    const skyboxSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
    });

    const skyboxImageBitmaps = await Promise.all(
      [
        "images/LeadenhallMarket/pos-x.jpg",
        "images/LeadenhallMarket/neg-x.jpg",
        "images/LeadenhallMarket/pos-y.jpg",
        "images/LeadenhallMarket/neg-y.jpg",
        "images/LeadenhallMarket/pos-z.jpg",
        "images/LeadenhallMarket/neg-z.jpg",
      ].map(loadImageBitmap)
    );

    const skyboxMipLevelCount = calculateMipLevelCount(
      skyboxImageBitmaps[0].width,
      skyboxImageBitmaps[0].height
    );
    const skyboxTexture = this.createTextureCubeFromSources(
      `GPU Texture: Skybox`,
      skyboxImageBitmaps,
      skyboxMipLevelCount
    );
    const skyboxTextureView = skyboxTexture.createView({
      label: `GPU Texture View: Skybox`,
      dimension: "cube",
    });

    this.skyboxBindGroup = this.device.createBindGroup({
      label: `GPU Bind Group: Skybox`,
      layout: this.skyboxRenderPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.skyboxUniformBuffer,
          },
        },
        {
          binding: 1,
          resource: skyboxTextureView,
        },
        {
          binding: 2,
          resource: skyboxSampler,
        },
      ],
    });
  }

  private createShaderModule(label: string, code: string): GPUShaderModule {
    const shaderModule = this.device.createShaderModule({
      label,
      code,
    });

    return shaderModule;
  }

  private createBuffer(
    label: string,
    size: number,
    usage: GPUBufferUsageFlags
  ): GPUBuffer {
    const buffer = this.device.createBuffer({
      label,
      size,
      usage,
    });

    return buffer;
  }

  private createTexture2DFromSource(
    label: string,
    source: GPUImageCopyExternalImageSource,
    mipLevelCount: number
  ): GPUTexture {
    let width: number;
    let height: number;
    if (source instanceof HTMLVideoElement) {
      width = source.videoWidth;
      height = source.videoHeight;
    } else if (source instanceof VideoFrame) {
      width = source.codedWidth;
      height = source.codedHeight;
    } else {
      width = source.width;
      height = source.height;
    }

    const texture = this.device.createTexture({
      label,
      format: "rgba8unorm",
      mipLevelCount,
      size: [width, height],
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.copySourceToTexture(this.device, source, texture);

    return texture;
  }

  private createTextureCubeFromSources(
    label: string,
    sources: Array<GPUImageCopyExternalImageSource>,
    mipLevelCount: number
  ): GPUTexture {
    const source = sources[0];

    let width: number;
    let height: number;
    if (source instanceof HTMLVideoElement) {
      width = source.videoWidth;
      height = source.videoHeight;
    } else if (source instanceof VideoFrame) {
      width = source.codedWidth;
      height = source.codedHeight;
    } else {
      width = source.width;
      height = source.height;
    }
    const length = sources.length;

    const texture = this.device.createTexture({
      label,
      format: "rgba8unorm",
      mipLevelCount,
      size: [width, height, length],
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.copySourcesToTexture(this.device, sources, texture);

    return texture;
  }

  private copySourceToTexture(
    device: GPUDevice,
    source: GPUImageCopyExternalImageSource,
    texture: GPUTexture
  ): void {
    let width: number;
    let height: number;
    if (source instanceof HTMLVideoElement) {
      width = source.videoWidth;
      height = source.videoHeight;
    } else if (source instanceof VideoFrame) {
      width = source.codedWidth;
      height = source.codedHeight;
    } else {
      width = source.width;
      height = source.height;
    }

    device.queue.copyExternalImageToTexture(
      { source },
      { texture },
      { width, height }
    );

    if (texture.mipLevelCount > 1) {
      this.generateMipmaps()(device, texture);
    }
  }

  private copySourcesToTexture(
    device: GPUDevice,
    sources: Array<GPUImageCopyExternalImageSource>,
    texture: GPUTexture
  ): void {
    sources.forEach(
      (source: GPUImageCopyExternalImageSource, layer: number) => {
        let width: number;
        let height: number;
        if (source instanceof HTMLVideoElement) {
          width = source.videoWidth;
          height = source.videoHeight;
        } else if (source instanceof VideoFrame) {
          width = source.codedWidth;
          height = source.codedHeight;
        } else {
          width = source.width;
          height = source.height;
        }

        device.queue.copyExternalImageToTexture(
          { source },
          { texture, origin: [0, 0, layer] },
          { width, height }
        );
      }
    );

    if (texture.mipLevelCount > 1) {
      this.generateMipmaps()(device, texture);
    }
  }

  private generateMipmaps(): (d: GPUDevice, t: GPUTexture) => void {
    let sampler: GPUSampler;
    let module: GPUShaderModule;
    const pipelineByFormat = new Map<GPUTextureFormat, GPURenderPipeline>();

    const context = this;

    return function generateMipmaps(device: GPUDevice, texture: GPUTexture) {
      if (!module) {
        module = device.createShaderModule({
          label: "GPU Shader Module: Mipmap Generation",
          code: mipmapShaderCode,
        });
        sampler = device.createSampler({
          minFilter: "linear",
        });
      }

      if (!pipelineByFormat.has(texture.format)) {
        const pipeline = device.createRenderPipeline({
          label: "GPU Render Pipeline: Mipmap Generation ",
          layout: "auto",
          vertex: {
            module: module,
            entryPoint: "vs_main",
            buffers: [
              {
                arrayStride: (4 + 2) * Float32Array.BYTES_PER_ELEMENT,
                stepMode: "vertex" as GPUVertexStepMode,
                attributes: [
                  {
                    format: "float32x4" as GPUVertexFormat,
                    offset: 0,
                    shaderLocation: 0,
                  },
                  {
                    format: "float32x2" as GPUVertexFormat,
                    offset: 4 * Float32Array.BYTES_PER_ELEMENT,
                    shaderLocation: 1,
                  },
                ],
              },
            ],
          },
          primitive: {
            topology: "triangle-list",
          },
          fragment: {
            module,
            entryPoint: "fs_main",
            targets: [
              {
                format: texture.format,
              },
            ],
          },
        });
        pipelineByFormat.set(texture.format, pipeline);
      }

      const pipeline = pipelineByFormat.get(texture.format);

      // vertex buffer
      // prettier-ignore
      const vertices = new Float32Array([
        -1.0,  1.0, 0.0, 1.0, 0.0, 0.0,
        -1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 
         1.0,  1.0, 0.0, 1.0, 1.0, 0.0, 
         1.0, -1.0, 0.0, 1.0, 1.0, 1.0,
      ]);

      const vertexBuffer = context.createBuffer(
        "GPU Buffer: Mipmaps Generation Vertex",
        vertices.byteLength,
        GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
      );

      device.queue.writeBuffer(
        vertexBuffer,
        0,
        vertices.buffer,
        0,
        vertices.byteLength
      );

      // index buffer
      const indices = new Uint32Array([0, 1, 2, 2, 1, 3]);

      const indexBuffer = context.createBuffer(
        "GPU Buffer: Mipmap Generation Index ",
        indices.byteLength,
        GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
      );

      device.queue.writeBuffer(
        indexBuffer,
        0,
        indices.buffer,
        0,
        indices.byteLength
      );

      const encoder = device.createCommandEncoder({
        label: "GPU Command Encoder: Mipmap Generation",
      });

      let width = texture.width;
      let height = texture.height;
      let baseMipLevel = 0;
      while (width > 1 || height > 1) {
        width = Math.max(1, (width / 2) | 0);
        height = Math.max(1, (height / 2) | 0);

        if (1 == texture.depthOrArrayLayers) {
          const bindGroup = device.createBindGroup({
            label: "GPU Bind Group: Mipmap Generation",
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              {
                binding: 0,
                resource: sampler,
              },
              {
                binding: 1,
                resource: texture.createView({
                  label: "GPU Texture View: Mipmap Generation",
                  baseMipLevel,
                  mipLevelCount: 1,
                }),
              },
            ],
          });

          ++baseMipLevel;

          const colorAttachments: GPURenderPassColorAttachment[] = [
            {
              view: texture.createView({
                label: "GPU Texture View: Mipmap Generation Render Target",
                baseMipLevel,
                mipLevelCount: 1,
              }),
              loadOp: "clear",
              storeOp: "store",
            },
          ];

          const passDescriptor: GPURenderPassDescriptor = {
            label: "GPU Render Pass Descriptor: Mipmap Generation",
            colorAttachments,
          };

          const pass = encoder.beginRenderPass(passDescriptor);
          pass.setPipeline(pipeline);
          pass.setVertexBuffer(0, vertexBuffer);
          pass.setIndexBuffer(indexBuffer, "uint32");
          pass.setBindGroup(0, bindGroup);
          pass.drawIndexed(6);
          pass.end();
        } else if (6 == texture.depthOrArrayLayers) {
          for (let layer = 0; layer < texture.depthOrArrayLayers; ++layer) {
            const bindGroup = device.createBindGroup({
              label: "GPU Bind Group: Mipmap Generation",
              layout: pipeline.getBindGroupLayout(0),
              entries: [
                {
                  binding: 0,
                  resource: sampler,
                },
                {
                  binding: 1,
                  resource: texture.createView({
                    label: "GPU Texture View: Mipmap Generation",
                    dimension: "2d",
                    baseMipLevel,
                    mipLevelCount: 1,
                    baseArrayLayer: layer,
                    arrayLayerCount: 1,
                  }),
                },
              ],
            });

            const colorAttachments: GPURenderPassColorAttachment[] = [
              {
                view: texture.createView({
                  label: "GPU Texture View: Mipmap Generation Render Target",
                  dimension: "2d",
                  baseMipLevel: baseMipLevel + 1,
                  mipLevelCount: 1,
                  baseArrayLayer: layer,
                  arrayLayerCount: 1,
                }),
                loadOp: "clear",
                storeOp: "store",
              },
            ];

            const passDescriptor: GPURenderPassDescriptor = {
              label: "GPU Render Pass Descriptor: Mipmap Generation",
              colorAttachments,
            };

            const pass = encoder.beginRenderPass(passDescriptor);
            pass.setPipeline(pipeline);
            pass.setVertexBuffer(0, vertexBuffer);
            pass.setIndexBuffer(indexBuffer, "uint32");
            pass.setBindGroup(0, bindGroup);
            pass.drawIndexed(6);
            pass.end();
          }

          ++baseMipLevel;
        }
      }

      device.queue.submit([encoder.finish()]);
    };
  }

  private initGUI(): void {
    const gui = new GUI({
      name: "My GUI",
      autoPlace: true,
      hideable: true,
      width: 300,
    });

    // profiling GUI
    const profiler = { fps: "0", cpuTime: "0", gpuTime: "0" };

    const profilerGUI = gui.addFolder("Profiler");
    profilerGUI.closed = false;
    this.fpsController = profilerGUI
      .add({ display: profiler.fps }, "display")
      .name("FPS");
    this.cpuTimeController = profilerGUI
      .add({ display: profiler.cpuTime }, "display")
      .name("CPU Time (ms)");
    this.gpuTimeController = profilerGUI
      .add({ display: profiler.gpuTime }, "display")
      .name("GPU Time (µs)");

    // texture options GUI
    const textureOptions = {
      mimaps: true,
      addressModeU: "repeat",
      addressModeV: "repeat",
      magFilter: "linear",
      minFilter: "linear",
    };

    const textureOptionsGUI = gui.addFolder("Texture Options");
    textureOptionsGUI.closed = false;
    this.mipmapsController = textureOptionsGUI
      .add(textureOptions, "mimaps")
      .name("Mipmaps");
    this.addressModeUController = textureOptionsGUI
      .add(textureOptions, "addressModeU")
      .options(["repeat", "clamp-to-edge"])
      .name("addressModeU");
    this.addressModeVController = textureOptionsGUI
      .add(textureOptions, "addressModeV")
      .options(["repeat", "clamp-to-edge"])
      .name("addressModeV");
    this.magFilterController = textureOptionsGUI
      .add(textureOptions, "magFilter")
      .options(["linear", "nearest"])
      .name("magFilter");
    this.minFilterController = textureOptionsGUI
      .add(textureOptions, "minFilter")
      .options(["linear", "nearest"])
      .name("minFilter");
  }

  private run(): void {
    requestAnimationFrame(this.drawFrame);
  }

  private drawFrame(now: number): void {
    fpsAvg.value = 1000 / (now - this.current);
    this.fpsController.setValue(fpsAvg.value.toFixed(1));
    this.current = now;

    const startTime = performance.now();

    // transform values
    const transformValues = new Float32Array(
      this.transformUniformBuffer.size / Float32Array.BYTES_PER_ELEMENT
    );

    // model matrix
    const model = mat4.identity();
    transformValues.set(model, 0);

    // noraml matrix
    const normal = mat4.transpose(mat4.inverse(model));
    transformValues.set(normal, model.length);

    // view matrix
    const view = this.camera.view;
    transformValues.set(view, model.length + normal.length);

    // projection matrix
    const aspect = this.canvas.width / this.canvas.height;
    const projection = mat4.perspective(
      utils.degToRad(45.0),
      aspect,
      1.0,
      100.0
    );

    transformValues.set(projection, model.length + normal.length + view.length);

    this.device.queue.writeBuffer(
      this.transformUniformBuffer,
      0,
      transformValues.buffer,
      transformValues.byteOffset,
      transformValues.byteLength
    );

    // skybox uniform buffer values
    const skyboxValues = new Float32Array(
      this.skyboxUniformBuffer.size / Float32Array.BYTES_PER_ELEMENT
    );
    skyboxValues.set(mat4.inverse(mat4.mul(projection, view)), 0);

    this.device.queue.writeBuffer(
      this.skyboxUniformBuffer,
      0,
      skyboxValues.buffer,
      skyboxValues.byteOffset,
      skyboxValues.byteLength
    );

    // camera position
    this.device.queue.writeBuffer(
      this.viewPositionUniformBuffer,
      0,
      this.camera.position.buffer,
      this.camera.position.byteOffset,
      this.camera.position.byteLength
    );

    // light vlaues
    const lightValues = new Float32Array(
      this.lightUniformBuffer.size / Float32Array.BYTES_PER_ELEMENT
    );
    const lightDir = vec3.normalize(vec3.create(0.0, -0.2, -0.75));
    lightValues.set(lightDir, 0);
    const lightAmbient = vec3.create(0.2, 0.2, 0.2);
    lightValues.set(lightAmbient, 4);
    const lightDiffuse = vec3.create(0.5, 0.5, 0.5);
    lightValues.set(lightDiffuse, 8);
    const lightSpecular = vec3.create(1.0, 1.0, 1.0);
    lightValues.set(lightSpecular, 12);

    this.device.queue.writeBuffer(
      this.lightUniformBuffer,
      0,
      lightValues.buffer,
      lightValues.byteOffset,
      lightValues.byteLength
    );

    // material shininess
    const shininess = new Float32Array([32.0]);
    this.device.queue.writeBuffer(
      this.materialShininessUniformBuffer,
      0,
      shininess.buffer,
      shininess.byteOffset,
      shininess.byteLength
    );

    // texture index
    const textureIndex = this.mipmapsController.getValue() ? 1 : 0;

    // sampler index
    const addressModeU = this.addressModeUController.getValue();
    const addressModeV = this.addressModeVController.getValue();
    const magFilter = this.magFilterController.getValue();
    const minFilter = this.minFilterController.getValue();
    const samplerIndex =
      (addressModeU === "repeat" ? 1 : 0) +
      (addressModeV === "repeat" ? 2 : 0) +
      (magFilter === "linear" ? 4 : 0) +
      (minFilter === "linear" ? 8 : 0);

    const commandEncoder = this.device.createCommandEncoder({
      label: "GPU Command Encoder: Draw Frame",
    });

    const renderTargetTextureView = this.context
      .getCurrentTexture()
      .createView({ label: "GPU Texture View: Canvas Texture View" });
    const colorAttachments: GPURenderPassColorAttachment[] = [
      {
        view: renderTargetTextureView,
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ];

    const renderPassDescriptor: GPURenderPassDescriptor = {
      label: "GPU Renderpass Descriptor: Draw Frame",
      colorAttachments,
      depthStencilAttachment: {
        view: this.depthTextureView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    };

    if (this.hasTimestamp) {
      renderPassDescriptor.timestampWrites = {
        querySet: this.querySet,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      };
    }

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.renderPipeline);
    passEncoder.setVertexBuffer(0, this.vertexBuffer);
    passEncoder.setIndexBuffer(this.indexBuffer, "uint32");
    passEncoder.setBindGroup(0, this.bindGroups[0][0]);
    passEncoder.setBindGroup(1, this.bindGroups[1][textureIndex]);
    passEncoder.setBindGroup(2, this.bindGroups[2][samplerIndex]);
    passEncoder.drawIndexed(36);

    passEncoder.setPipeline(this.skyboxRenderPipeline);
    passEncoder.setBindGroup(0, this.skyboxBindGroup);
    passEncoder.draw(3);

    passEncoder.end();

    if (this.hasTimestamp) {
      commandEncoder.resolveQuerySet(
        this.querySet,
        0,
        this.querySet.count,
        this.resolveBuffer,
        0
      );

      if ("unmapped" === this.resultBuffer.mapState) {
        commandEncoder.copyBufferToBuffer(
          this.resolveBuffer,
          0,
          this.resultBuffer,
          0,
          this.resultBuffer.size
        );
      }
    }

    this.device.queue.submit([commandEncoder.finish()]);

    cpuTimeAvg.value = performance.now() - startTime;
    this.cpuTimeController.setValue(cpuTimeAvg.value.toFixed(1));

    if (this.hasTimestamp && "unmapped" === this.resultBuffer.mapState) {
      this.resultBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const times = new BigInt64Array(this.resultBuffer.getMappedRange());
        gpuTimeAvg.value = Number(times[1] - times[0]) / 1000;
        this.gpuTimeController.setValue(gpuTimeAvg.value.toFixed(1));

        this.resultBuffer.unmap();
      });
    }

    requestAnimationFrame(this.drawFrame);
  }
}
