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
import Camera from "@/camera";
import glTFLoader, { type Geometry } from "@/loader";

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
  private presentationFormat: GPUTextureFormat;
  private multisamplingTexture: GPUTexture;
  private multisamplingTextureView: GPUTextureView;
  private depthTextures: Array<GPUTexture>;
  private depthTextureViews: Array<GPUTextureView>;

  private geometries: Array<Geometry>;

  private renderPipelines: Array<GPURenderPipeline>;
  private vertexBuffer: GPUBuffer;
  private indexBuffer: GPUBuffer;
  private bindGroups: Array<Array<GPUBindGroup>>;
  private transformUniformBuffer: GPUBuffer;
  private viewPositionUniformBuffer: GPUBuffer;
  private lightUniformBuffer: GPUBuffer;
  private materialShininessUniformBuffer: GPUBuffer;

  private skyboxRenderPipelines: Array<GPURenderPipeline>;
  private skyboxBindGroups: Array<Array<GPUBindGroup>>;
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
  private msaaController: GUIController;
  private skyboxController: GUIController;

  public constructor() {
    this.camera = new Camera();

    this.depthTextures = new Array<GPUTexture>(2);
    this.depthTextureViews = new Array<GPUTextureView>(2);

    this.renderPipelines = new Array<GPURenderPipeline>(2);
    this.bindGroups = new Array<Array<GPUBindGroup>>(
      new Array<GPUBindGroup>(),
      new Array<GPUBindGroup>(),
      new Array<GPUBindGroup>(),
      new Array<GPUBindGroup>()
    );

    this.skyboxRenderPipelines = new Array<GPURenderPipeline>(2);
    this.skyboxBindGroups = new Array<Array<GPUBindGroup>>(
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
    this.createMultisamplingTexture();
    this.createDepthTextures();

    await this.loadModel();

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
      throw new Error("WebGPU not supported.");
    }
  }

  private async requestAdapter(): Promise<void> {
    this.adapter = await navigator.gpu?.requestAdapter();
    if (!this.adapter) {
      throw new Error("Failed to request WebGPU adapter.");
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
      throw new Error("Failed to request WebGPU device.");
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
      throw new Error("Failed to find canvas element.");
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

        this.createMultisamplingTexture();
        this.createDepthTextures();
      }
    });
    observer.observe(this.canvas);
  }

  private configContext(): void {
    this.context = this.canvas.getContext("webgpu");
    if (!this.context) {
      throw new Error("Failed to get WebGPU context from canvas.");
    }

    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: "premultiplied",
    });
  }

  private createMultisamplingTexture(): void {
    if (this.multisamplingTexture) {
      this.multisamplingTexture.destroy();
    }

    this.multisamplingTexture = this.device.createTexture({
      label: "GPU Texture: Multisampling Texture",
      size: [this.canvas.width, this.canvas.height],
      format: this.presentationFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      sampleCount: 4,
    });

    this.multisamplingTextureView = this.multisamplingTexture.createView({
      label: "GPU Texture View: Multisampling Texture View",
    });
  }

  private createDepthTextures(): void {
    for (let i = 0; i < 2; ++i) {
      if (this.depthTextures[i]) {
        this.depthTextures[i].destroy();
      }

      this.depthTextures[i] = this.device.createTexture({
        label: `GPU Texture: Depth Texture ${i && "with MSAA"}`,
        size: [this.canvas.width, this.canvas.height],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        sampleCount: i ? 4 : 1,
      });

      this.depthTextureViews[i] = this.depthTextures[i].createView({
        label: `GPU Texture View: Depth Texture View ${i && "with MSAA"}`,
      });
    }
  }

  private async loadModel(): Promise<void> {
    const loader = new glTFLoader();
    await loader.load("models/DamagedHelmet/DamagedHelmet.gltf");

    this.geometries = loader.getGeometries();
  }

  private createRenderPipeline(): void {
    const shaderModule = this.createShaderModule(
      "GPU Shader Module",
      shaderCode
    );

    const renderPipelineDescriptor: GPURenderPipelineDescriptor = {
      label: "GPU Render Pipeline",
      layout: "auto",
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: this.geometries[0].arrayStride,
            stepMode: "vertex" as GPUVertexStepMode,
            attributes: [
              {
                format: this.geometries[0].position.format,
                offset: this.geometries[0].position.offset,
                shaderLocation: 0,
              },
              {
                format: this.geometries[0].texCoord.format,
                offset: this.geometries[0].texCoord.offset,
                shaderLocation: 1,
              },
              {
                format: this.geometries[0].normal.format,
                offset: this.geometries[0].normal.offset,
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
      multisample: {
        count: 1,
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
    };

    for (let i = 0; i < 2; ++i) {
      if (i) {
        renderPipelineDescriptor.label = "GPU Render Pipeline with MSAA";
        renderPipelineDescriptor.multisample.count = 4;
      }

      this.renderPipelines[i] = this.device.createRenderPipeline(
        renderPipelineDescriptor
      );
    }
  }

  private createVertexBuffer(): void {
    // prettier-ignore
    const vertices = this.geometries[0].vertices;

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
    const indices = this.geometries[0].indices;

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

    for (let i = 0; i < 2; ++i) {
      const bindGroup = this.device.createBindGroup({
        label: `GPU Bind Group 0: Transform, View Position, Light ${
          i && "with MSAA"
        }`,
        layout: this.renderPipelines[i].getBindGroupLayout(0),
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
  }

  private async createTexture(): Promise<void> {
    this.materialShininessUniformBuffer = this.createBuffer(
      `GPU Uniform Buffer: Material Shininess`,
      4 * Float32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    const baseColorImageBitmap = await loadImageBitmap(
      this.geometries[0].textures.baseColorURI
    );
    const metallicRoughnessImageBitmap = await loadImageBitmap(
      this.geometries[0].textures.metallicRoughnessURI
    );
    const normalImageBitmap = await loadImageBitmap(
      this.geometries[0].textures.normalURI
    );
    const emissive = this.geometries[0].textures.emissive;
    const emissiveImageBitmap = await loadImageBitmap(
      this.geometries[0].textures.emissiveURI
    );
    const occlusionImageBitmap = await loadImageBitmap(
      this.geometries[0].textures.occlusionURI
    );

    for (let i = 0; i < 4; ++i) {
      const mipIndex = i & 1 ? 1 : 0;
      const renderPipelineIndex = i & 2 ? 1 : 0;

      const baseColorMipLevelCount = mipIndex
        ? calculateMipLevelCount(
            baseColorImageBitmap.width,
            baseColorImageBitmap.height
          )
        : 1;
      const baseColorTexture = this.createTexture2DFromSource(
        `GPU Texture: Base Color ${mipIndex && "with Mipmaps"}`,
        baseColorImageBitmap,
        baseColorMipLevelCount
      );
      const baseColorTextureView = baseColorTexture.createView({
        label: `GPU Texture View: Base Color ${mipIndex && "with Mipmaps"}`,
      });

      const metallicRoughnessMipLevelCount = mipIndex
        ? calculateMipLevelCount(
            metallicRoughnessImageBitmap.width,
            metallicRoughnessImageBitmap.height
          )
        : 1;
      const metallicRoughnessTexture = this.createTexture2DFromSource(
        `GPU Texture: Metallic Roughness ${mipIndex && "with Mipmaps"}`,
        metallicRoughnessImageBitmap,
        metallicRoughnessMipLevelCount
      );
      const metallicRoughnessTextureView = metallicRoughnessTexture.createView({
        label: `GPU Texture View: Metallic Roughness ${mipIndex && "with Mipmaps"}`,
      });

      const noramlMipLevelCount = mipIndex
        ? calculateMipLevelCount(
            normalImageBitmap.width,
            normalImageBitmap.height
          )
        : 1;
      const normalTexture = this.createTexture2DFromSource(
        `GPU Texture: Noraml ${mipIndex && "with Mipmaps"}`,
        normalImageBitmap,
        noramlMipLevelCount
      );
      const normalTextureView = normalTexture.createView({
        label: `GPU Texture View: Noraml ${mipIndex && "with Mipmaps"}`,
      });

      const emissiveMipLevelCount = mipIndex
        ? calculateMipLevelCount(
            emissiveImageBitmap.width,
            emissiveImageBitmap.height
          )
        : 1;
      const emissiveTexture = this.createTexture2DFromSource(
        `GPU Texture: Emissive ${mipIndex && "with Mipmaps"}`,
        emissiveImageBitmap,
        emissiveMipLevelCount
      );
      const emissiveTextureView = emissiveTexture.createView({
        label: `GPU Texture View: Emissive ${mipIndex && "with Mipmaps"}`,
      });

      const occlusionMipLevelCount = mipIndex
        ? calculateMipLevelCount(
            occlusionImageBitmap.width,
            occlusionImageBitmap.height
          )
        : 1;
      const occlusionTexture = this.createTexture2DFromSource(
        `GPU Texture: Occlusion ${mipIndex && "with Mipmaps"}`,
        occlusionImageBitmap,
        occlusionMipLevelCount
      );
      const occlusionTextureView = occlusionTexture.createView({
        label: `GPU Texture View: Occlusion ${mipIndex && "with Mipmaps"}`,
      });

      const bindGroup = this.device.createBindGroup({
        label: `GPU Bind Group 1: Material ${
          mipIndex && "with Mipmaps"
        } ${renderPipelineIndex && "with MSAA"}`,
        layout: this.renderPipelines[renderPipelineIndex].getBindGroupLayout(1),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: this.materialShininessUniformBuffer,
            },
          },
          {
            binding: 1,
            resource: baseColorTextureView,
          },
          {
            binding: 2,
            resource: metallicRoughnessTextureView,
          },
        ],
      });
      this.bindGroups[1].push(bindGroup);
    }

    const leadenhallMarketImageBitmaps = await Promise.all(
      [
        "images/LeadenhallMarket/pos-x.jpg",
        "images/LeadenhallMarket/neg-x.jpg",
        "images/LeadenhallMarket/pos-y.jpg",
        "images/LeadenhallMarket/neg-y.jpg",
        "images/LeadenhallMarket/pos-z.jpg",
        "images/LeadenhallMarket/neg-z.jpg",
      ].map(loadImageBitmap)
    );
    const pureSkyImageBitmaps = await Promise.all(
      [
        "images/PureSky/right.jpg",
        "images/PureSky/left.jpg",
        "images/PureSky/top.jpg",
        "images/PureSky/bottom.jpg",
        "images/PureSky/front.jpg",
        "images/PureSky/back.jpg",
      ].map(loadImageBitmap)
    );
    const cubeImageBitmaps = [
      leadenhallMarketImageBitmaps,
      pureSkyImageBitmaps,
    ];

    for (let i = 0; i < 8; ++i) {
      const mipIndex = i & 1 ? 1 : 0;
      const renderPipelineIndex = i & 2 ? 1 : 0;
      const skyboxIndex = i & 4 ? 1 : 0;

      const cubeMipLevelCount = i
        ? calculateMipLevelCount(
            cubeImageBitmaps[skyboxIndex][0].width,
            cubeImageBitmaps[skyboxIndex][0].height
          )
        : 1;

      const cubeTexture = this.createTextureCubeFromSources(
        `GPU Texture: Cube ${skyboxIndex}`,
        cubeImageBitmaps[skyboxIndex],
        cubeMipLevelCount
      );

      const cubeTextureView = cubeTexture.createView({
        label: `GPU Texture View: Cube ${skyboxIndex}`,
        dimension: "cube",
      });

      const bindGroup = this.device.createBindGroup({
        label: `GPU Bind Group 2: Cube ${mipIndex && "with Mipmaps"} ${
          renderPipelineIndex && "with MSAA"
        }`,
        layout: this.renderPipelines[renderPipelineIndex].getBindGroupLayout(2),
        entries: [
          {
            binding: 0,
            resource: cubeTextureView,
          },
        ],
      });
      this.bindGroups[2].push(bindGroup);
    }

    for (let i = 0; i < 32; ++i) {
      const sampler = this.device.createSampler({
        label: `GPU Sampler: Sampler ${i}`,
        addressModeU: i & 1 ? "repeat" : "clamp-to-edge",
        addressModeV: i & 2 ? "repeat" : "clamp-to-edge",
        magFilter: i & 4 ? "linear" : "nearest",
        minFilter: i & 8 ? "linear" : "nearest",
      });

      const renderPipelineIndex = i & 16 ? 1 : 0;
      const bindGroup = this.device.createBindGroup({
        label: `GPU Bind Group 2 : Sampler ${i}, ${
          renderPipelineIndex && "with MSAA"
        }`,
        layout: this.renderPipelines[renderPipelineIndex].getBindGroupLayout(3),
        entries: [
          {
            binding: 0,
            resource: sampler,
          },
        ],
      });
      this.bindGroups[3].push(bindGroup);
    }
  }

  private createSkyboxRenderPipeline(): void {
    const skyboxShaderModule = this.createShaderModule(
      "GPU Shader Module: Skybox",
      skyboxShaderCode
    );

    const skyboxRenderPipelineDescriptor: GPURenderPipelineDescriptor = {
      label: `GPU Render Pipeline: Skybox`,
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
      multisample: {
        count: 1,
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
    };

    for (let i = 0; i < 2; ++i) {
      if (i) {
        skyboxRenderPipelineDescriptor.label = "GPU Render Pipeline with MSAA";
        skyboxRenderPipelineDescriptor.multisample.count = 4;
      }

      this.skyboxRenderPipelines[i] = this.device.createRenderPipeline(
        skyboxRenderPipelineDescriptor
      );
    }
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

    for (let i = 0; i < 2; ++i) {
      const bindGroup = this.device.createBindGroup({
        label: `GPU Bind Group: Skybox Matrix, Sampler ${i && "with MSAA"}`,
        layout: this.skyboxRenderPipelines[i].getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: this.skyboxUniformBuffer,
            },
          },
          {
            binding: 1,
            resource: skyboxSampler,
          },
        ],
      });
      this.skyboxBindGroups[0].push(bindGroup);
    }

    const leadenhallMarketImageBitmaps = await Promise.all(
      [
        "images/LeadenhallMarket/pos-x.jpg",
        "images/LeadenhallMarket/neg-x.jpg",
        "images/LeadenhallMarket/pos-y.jpg",
        "images/LeadenhallMarket/neg-y.jpg",
        "images/LeadenhallMarket/pos-z.jpg",
        "images/LeadenhallMarket/neg-z.jpg",
      ].map(loadImageBitmap)
    );
    const pureSkyImageBitmaps = await Promise.all(
      [
        "images/PureSky/right.jpg",
        "images/PureSky/left.jpg",
        "images/PureSky/top.jpg",
        "images/PureSky/bottom.jpg",
        "images/PureSky/front.jpg",
        "images/PureSky/back.jpg",
      ].map(loadImageBitmap)
    );

    const skyboxImageBitmaps = [
      leadenhallMarketImageBitmaps,
      pureSkyImageBitmaps,
    ];

    const skyboxTextureViews = [];
    for (let i = 0; i < 2; ++i) {
      const skyboxMipLevelCount = calculateMipLevelCount(
        skyboxImageBitmaps[i][0].width,
        skyboxImageBitmaps[i][0].height
      );
      const skyboxTexture = this.createTextureCubeFromSources(
        `GPU Texture: Skybox ${i}`,
        skyboxImageBitmaps[i],
        skyboxMipLevelCount
      );

      skyboxTextureViews.push(
        skyboxTexture.createView({
          label: `GPU Texture View: Skybox ${i}`,
          dimension: "cube",
        })
      );
    }

    for (let i = 0; i < 4; ++i) {
      const renderPipelineIndex = i & 1 ? 1 : 0;
      const skyboxIndex = i & 2 ? 1 : 0;

      const bindGroup = this.device.createBindGroup({
        label: `GPU Bind Group: Skybox Texture ${
          renderPipelineIndex && "with MSAA"
        }`,
        layout:
          this.skyboxRenderPipelines[renderPipelineIndex].getBindGroupLayout(1),
        entries: [
          {
            binding: 0,
            resource: skyboxTextureViews[skyboxIndex],
          },
        ],
      });
      this.skyboxBindGroups[1].push(bindGroup);
    }
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

    // multisampling anti-aliasing GUI
    const msaaOptions = {
      enable: false,
    };

    const antiAliasingGUI = gui.addFolder("Multisampling Anti-Aliasing (MSAA)");
    antiAliasingGUI.closed = false;
    this.msaaController = antiAliasingGUI
      .add(msaaOptions, "enable")
      .name("Enable");

    // skybox GUI
    const skyboxOptions = {
      skybox: "Leadenhall Market",
    };

    const skyboxGUI = gui.addFolder("Skybox");
    skyboxGUI.closed = false;
    this.skyboxController = skyboxGUI
      .add(skyboxOptions, "skybox")
      .options(["Leadenhall Market", "Pure Sky"])
      .name("Skybox");
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
    const model = this.geometries[0].model;
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

    // sampler index
    const addressModeU = this.addressModeUController.getValue();
    const addressModeV = this.addressModeVController.getValue();
    const magFilter = this.magFilterController.getValue();
    const minFilter = this.minFilterController.getValue();
    const samplerIndex =
      (addressModeU === "repeat" ? 1 : 0) +
      (addressModeV === "repeat" ? 2 : 0) +
      (magFilter === "linear" ? 4 : 0) +
      (minFilter === "linear" ? 8 : 0) +
      (this.msaaController.getValue() ? 16 : 0);

    const commandEncoder = this.device.createCommandEncoder({
      label: "GPU Command Encoder: Draw Frame",
    });

    const canvasTexture = this.context.getCurrentTexture();
    const canvasTextureView = canvasTexture.createView({
      label: "GPU Texture View: Canvas Texture View",
    });

    const colorAttachments: GPURenderPassColorAttachment[] = [
      {
        view: canvasTextureView,
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ];
    if (this.msaaController.getValue()) {
      colorAttachments[0].view = this.multisamplingTextureView;
      colorAttachments[0].resolveTarget = canvasTextureView;
    }

    const depthTextureViewIndex = this.msaaController.getValue() ? 1 : 0;
    const renderPassDescriptor: GPURenderPassDescriptor = {
      label: "GPU Renderpass Descriptor: Draw Frame",
      colorAttachments,
      depthStencilAttachment: {
        view: this.depthTextureViews[depthTextureViewIndex],
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

    // mipmaps index
    const mipIndex = this.mipmapsController.getValue() ? 1 : 0;

    // render pipeline(w/ MSAA, w/o MSAA) index
    const renderPipelineIndex = this.msaaController.getValue() ? 1 : 0;

    // texture index
    const textureIndex = (mipIndex ? 1 : 0) + (renderPipelineIndex ? 2 : 0);

    // skybox index
    const skyboxIndex =
      this.skyboxController.getValue() === "Leadenhall Market" ? 0 : 1;

    // cubemap index
    const cubemapIndex =
      (mipIndex ? 1 : 0) +
      (renderPipelineIndex ? 2 : 0) +
      (skyboxIndex ? 4 : 0);

    // skybox texture index
    const skyboxTextureIndex =
      (renderPipelineIndex ? 1 : 0) + (skyboxIndex ? 2 : 0);

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.renderPipelines[renderPipelineIndex]);
    passEncoder.setVertexBuffer(0, this.vertexBuffer);
    passEncoder.setIndexBuffer(this.indexBuffer, "uint16");
    passEncoder.setBindGroup(0, this.bindGroups[0][renderPipelineIndex]);
    passEncoder.setBindGroup(1, this.bindGroups[1][textureIndex]);
    passEncoder.setBindGroup(2, this.bindGroups[2][cubemapIndex]);
    passEncoder.setBindGroup(3, this.bindGroups[3][samplerIndex]);
    passEncoder.drawIndexed(this.geometries[0].indices.length);

    passEncoder.setPipeline(this.skyboxRenderPipelines[renderPipelineIndex]);
    passEncoder.setBindGroup(0, this.skyboxBindGroups[0][renderPipelineIndex]);
    passEncoder.setBindGroup(1, this.skyboxBindGroups[1][skyboxTextureIndex]);
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
