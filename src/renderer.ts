import shaderCode from "@/shaders/shader.wgsl";
import mipmapShaderCode from "@/shaders/mipmap.wgsl";
import skyboxShaderCode from "@/shaders/skybox.wgsl";
import equirectangularShaderCode from "@/shaders/equirectangular.wgsl";
import irradiancemapShaderCode from "@/shaders/irradiancemap.wgsl";
import prefiltermapShaderCode from "@/shaders/prefiltermap.wgsl";
import brdfShaderCode from "@/shaders/brdf.wgsl";
import { mat4, utils, vec3 } from "wgpu-matrix";
import { GUI, GUIController } from "dat.gui";
import {
  loadImageBitmap,
  calculateMipLevelCount,
  RollingAverage,
} from "@/utils";
import Camera from "@/camera";
import glTFLoader, { type Geometry } from "@/glTF";
import HDRLoader, { type HDR } from "@/hdr";

enum CubemapType {
  SKYBOX = "Skybox",
  IRRADIANCE = "Irradiance",
  PREFILTER = "Prefilter",
}

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

  private hdrs: Array<HDR>;

  private renderPipelines: Array<GPURenderPipeline>;
  private vertexBuffer: GPUBuffer;
  private indexBuffer: GPUBuffer;
  private bindGroups: Array<Array<GPUBindGroup>>;
  private transformUniformBuffer: GPUBuffer;
  private viewPositionUniformBuffer: GPUBuffer;
  private lightPositionsUniformBuffer: GPUBuffer;
  private lightColorsUniformBuffer: GPUBuffer;

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

    await this.loadHDR();

    this.createRenderPipeline();
    this.createVertexBuffer();
    this.createIndexBuffer();
    this.createUniformBuffer();
    await this.createTexture();

    this.createSkyboxRenderPipeline();
    await this.createSkyboxUniformBuffer();

    this.initGUI();

    this.canvas.setAttribute("style", "visibility: visible;");
    const spinner = document.querySelector("div.lds-spinner");
    if (spinner) {
      document.body.removeChild(spinner);
    }
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
    const hasFilterableFloat = this.adapter?.features.has("float32-filterable");
    if (!hasFilterableFloat) {
      throw new Error("WebGPU device does not support float32 filterable.");
    }

    this.hasTimestamp = this.adapter?.features.has("timestamp-query");

    const requiredFeatures: GPUFeatureName[] = this.hasTimestamp
      ? ["timestamp-query", "float32-filterable"]
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
    this.canvas.setAttribute("style", "visibility: hidden;");

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

  private async loadHDR(): Promise<void> {
    const loader0 = new HDRLoader();
    await loader0.load("images/puresky_4k.hdr");
    const loader1 = new HDRLoader();
    await loader1.load("images/syferfontein_0d_clear_puresky_4k.hdr");
    this.hdrs = [loader0.hdr, loader1.hdr];
  }

  private createRenderPipeline(): void {
    const shaderModule = this.createShaderModule(
      "GPU Shader Module",
      shaderCode
    );

    const bindGroupLayout0 = this.device.createBindGroupLayout({
      label: "GPU Bind Group 0 Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform" as GPUBufferBindingType },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" as GPUBufferBindingType },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" as GPUBufferBindingType },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" as GPUBufferBindingType },
        },
      ],
    });
    const bindGroupLayout1 = this.device.createBindGroupLayout({
      label: "GPU Bind Group 1 Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: "float" as GPUTextureSampleType,
            viewDimension: "2d" as GPUTextureViewDimension,
            multisampled: false,
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: "float" as GPUTextureSampleType,
            viewDimension: "2d" as GPUTextureViewDimension,
            multisampled: false,
          },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: "float" as GPUTextureSampleType,
            viewDimension: "2d" as GPUTextureViewDimension,
            multisampled: false,
          },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" as GPUBufferBindingType },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: "float" as GPUTextureSampleType,
            viewDimension: "2d" as GPUTextureViewDimension,
            multisampled: false,
          },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: "float" as GPUTextureSampleType,
            viewDimension: "2d" as GPUTextureViewDimension,
            multisampled: false,
          },
        },
      ],
    });
    const bindGroupLayout2 = this.device.createBindGroupLayout({
      label: "GPU Bind Group 2 Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: "float" as GPUTextureSampleType,
            viewDimension: "cube" as GPUTextureViewDimension,
            multisampled: false,
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: "float" as GPUTextureSampleType,
            viewDimension: "cube" as GPUTextureViewDimension,
            multisampled: false,
          },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: "float" as GPUTextureSampleType,
            viewDimension: "2d" as GPUTextureViewDimension,
            multisampled: false,
          },
        },
      ],
    });
    const bindGroupLayout3 = this.device.createBindGroupLayout({
      label: "GPU Bind Group 3 Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" as GPUSamplerBindingType },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" as GPUSamplerBindingType },
        },
      ],
    });
    const renderPipelineLayout = this.device.createPipelineLayout({
      label: "GPU Render Pipeline Layout",
      bindGroupLayouts: [
        bindGroupLayout0,
        bindGroupLayout1,
        bindGroupLayout2,
        bindGroupLayout3,
      ],
    });

    const renderPipelineDescriptor: GPURenderPipelineDescriptor = {
      label: "GPU Render Pipeline",
      layout: renderPipelineLayout,
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

    this.lightPositionsUniformBuffer = this.createBuffer(
      `GPU Uniform Buffer: Light Positions`,
      4 * 4 * Float32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    this.lightColorsUniformBuffer = this.createBuffer(
      `GPU Uniform Buffer: Light Colors`,
      4 * 4 * Float32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    for (let i = 0; i < 2; ++i) {
      const bindGroup = this.device.createBindGroup({
        label: `GPU Bind Group 0: Transform, View Position, Light Direction${
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
              buffer: this.lightPositionsUniformBuffer,
            },
          },
          {
            binding: 3,
            resource: {
              buffer: this.lightColorsUniformBuffer,
            },
          },
        ],
      });
      this.bindGroups[0].push(bindGroup);
    }
  }

  private async createTexture(): Promise<void> {
    const baseColorImageBitmap = await loadImageBitmap(
      this.geometries[0].textures.baseColorURI
    );
    const metallicRoughnessImageBitmap = await loadImageBitmap(
      this.geometries[0].textures.metallicRoughnessURI
    );
    const normalImageBitmap = await loadImageBitmap(
      this.geometries[0].textures.normalURI
    );
    const emissive = new Float32Array(this.geometries[0].textures.emissive);
    const emissiveBuffer = this.createBuffer(
      `GPU Uniform Buffer: Emissive Factor`,
      4 * Float32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(
      emissiveBuffer,
      0,
      emissive.buffer,
      emissive.byteOffset,
      emissive.byteLength
    );
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
        "rgba8unorm-srgb",
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
        "rgba8unorm",
        metallicRoughnessMipLevelCount
      );
      const metallicRoughnessTextureView = metallicRoughnessTexture.createView({
        label: `GPU Texture View: Metallic Roughness ${
          mipIndex && "with Mipmaps"
        }`,
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
        "rgba8unorm",
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
        "rgba8unorm-srgb",
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
        "rgba8unorm",
        occlusionMipLevelCount
      );
      const occlusionTextureView = occlusionTexture.createView({
        label: `GPU Texture View: Occlusion ${mipIndex && "with Mipmaps"}`,
      });

      const bindGroup = this.device.createBindGroup({
        label: `GPU Bind Group 1: Material ${mipIndex && "with Mipmaps"} ${
          renderPipelineIndex && "with MSAA"
        }`,
        layout: this.renderPipelines[renderPipelineIndex].getBindGroupLayout(1),
        entries: [
          {
            binding: 0,
            resource: baseColorTextureView,
          },
          {
            binding: 1,
            resource: metallicRoughnessTextureView,
          },
          {
            binding: 2,
            resource: normalTextureView,
          },
          {
            binding: 3,
            resource: {
              buffer: emissiveBuffer,
            },
          },
          {
            binding: 4,
            resource: emissiveTextureView,
          },
          {
            binding: 5,
            resource: occlusionTextureView,
          },
        ],
      });
      this.bindGroups[1].push(bindGroup);
    }

    let irradianceMaps: GPUTexture[] = [];
    let prefilterMaps: GPUTexture[] = [];
    for (let i = 0; i < 2; ++i) {
      irradianceMaps.push(
        this.generateCubemap()(
          this.device,
          this.hdrs[i],
          32,
          CubemapType.IRRADIANCE
        )
      );

      prefilterMaps.push(
        this.generateCubemap()(
          this.device,
          this.hdrs[i],
          128,
          CubemapType.PREFILTER
        )
      );
    }

    const brdfLUT = this.generateBRDFLUT()(this.device, 512);
    const brdfLUTView = brdfLUT.createView({
      label: "GPU Texture View: BRDF LUT",
    });

    for (let i = 0; i < 8; ++i) {
      const mipIndex = i & 1 ? 1 : 0;
      const renderPipelineIndex = i & 2 ? 1 : 0;
      const skyboxIndex = i & 4 ? 1 : 0;

      const irradiancemapView = irradianceMaps[skyboxIndex].createView({
        label: `GPU Texture View: Irradiancemap ${skyboxIndex}`,
        dimension: "cube",
      });

      const prefiltermapView = prefilterMaps[skyboxIndex].createView({
        label: `GPU Texture View: Prefiltermap ${skyboxIndex}`,
        dimension: "cube",
      });

      const bindGroup = this.device.createBindGroup({
        label: `GPU Bind Group 2: Irradiancemap, Prefiltermap ${
          mipIndex && "with Mipmaps"
        } ${renderPipelineIndex && "with MSAA"}`,
        layout: this.renderPipelines[renderPipelineIndex].getBindGroupLayout(2),
        entries: [
          {
            binding: 0,
            resource: irradiancemapView,
          },
          {
            binding: 1,
            resource: prefiltermapView,
          },
          {
            binding: 2,
            resource: brdfLUTView,
          },
        ],
      });
      this.bindGroups[2].push(bindGroup);
    }

    for (let i = 0; i < 32; ++i) {
      const sampler2D = this.device.createSampler({
        label: `GPU Sampler: Sampler 2D ${i}`,
        addressModeU: i & 1 ? "repeat" : "clamp-to-edge",
        addressModeV: i & 2 ? "repeat" : "clamp-to-edge",
        magFilter: i & 4 ? "linear" : "nearest",
        minFilter: i & 8 ? "linear" : "nearest",
      });

      const samplerCube = this.device.createSampler({
        label: `GPU Sampler: Sampler Cube ${i}`,
        addressModeU: "clamp-to-edge",
        addressModeV: "clamp-to-edge",
        addressModeW: "clamp-to-edge",
        magFilter: "linear",
        minFilter: "linear",
        mipmapFilter: "linear",
      });

      const renderPipelineIndex = i & 16 ? 1 : 0;
      const bindGroup = this.device.createBindGroup({
        label: `GPU Bind Group 3 : Sampler ${i}, ${
          renderPipelineIndex && "with MSAA"
        }`,
        layout: this.renderPipelines[renderPipelineIndex].getBindGroupLayout(3),
        entries: [
          {
            binding: 0,
            resource: sampler2D,
          },
          {
            binding: 1,
            resource: samplerCube,
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

    const bindGroupLayout0 = this.device.createBindGroupLayout({
      label: "GPU Bind Group 0 Layout: Skybox",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: {
            type: "uniform" as GPUBufferBindingType,
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" as GPUSamplerBindingType },
        },
      ],
    });
    const bindGroupLayout1 = this.device.createBindGroupLayout({
      label: "GPU Bind Group 1 Layout: Skybox",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: "float" as GPUTextureSampleType,
            viewDimension: "cube" as GPUTextureViewDimension,
            multisampled: false,
          },
        },
      ],
    });
    const skyboxRenderPipelineLayout = this.device.createPipelineLayout({
      label: "GPU Render Pipeline Layout: Skybox",
      bindGroupLayouts: [bindGroupLayout0, bindGroupLayout1],
    });
    const skyboxRenderPipelineDescriptor: GPURenderPipelineDescriptor = {
      label: `GPU Render Pipeline: Skybox`,
      layout: skyboxRenderPipelineLayout,
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
      label: "GPU Sampler: Skybox",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
      addressModeW: "clamp-to-edge",
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

    const skyboxTextureViews = [];
    for (let i = 0; i < 2; ++i) {
      const skyboxTexture = this.generateCubemap()(
        this.device,
        this.hdrs[i],
        1440,
        CubemapType.SKYBOX
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
    format: GPUTextureFormat,
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
      format,
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

  private generateCubemap(): (
    d: GPUDevice,
    h: HDR,
    s: number,
    t: CubemapType
  ) => GPUTexture {
    const modules = new Map<CubemapType, GPUShaderModule>();
    const computePipelines = new Map<CubemapType, GPUComputePipeline>();

    const context = this;

    return function generateCubemap(
      device: GPUDevice,
      hdr: HDR,
      size: number,
      type: CubemapType
    ): GPUTexture {
      if (!modules.has(type)) {
        let code: string;
        switch (type) {
          case CubemapType.SKYBOX:
            code = equirectangularShaderCode;
            break;
          case CubemapType.IRRADIANCE:
            code = irradiancemapShaderCode;
            break;
          case CubemapType.PREFILTER:
            code = prefiltermapShaderCode;
            break;
          default:
            throw new Error(`Invliad cubemap type: ${type}`);
        }
        const module = device.createShaderModule({
          label: `GPU Shader Module: Cubemap Generation ${type}`,
          code,
        });
        modules.set(type, module);
      }

      if (!computePipelines.has(type)) {
        let bindGroupLayout: GPUBindGroupLayout;
        if (type !== CubemapType.PREFILTER) {
          bindGroupLayout = device.createBindGroupLayout({
            label: "GPU Bind Group Layout: Cubemap Generation",
            entries: [
              {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                texture: {
                  sampleType: "float" as GPUTextureSampleType,
                  viewDimension: "2d" as GPUTextureViewDimension,
                  multisampled: false,
                },
              },
              {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                storageTexture: {
                  access: "write-only" as GPUStorageTextureAccess,
                  format: "rgba32float" as GPUTextureFormat,
                  viewDimension: "2d-array" as GPUTextureViewDimension,
                },
              },
            ],
          });
        } else {
          bindGroupLayout = device.createBindGroupLayout({
            label: "GPU Bind Group Layout: Cubemap Generation",
            entries: [
              {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "uniform" as GPUBufferBindingType },
              },
              {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                texture: {
                  sampleType: "float" as GPUTextureSampleType,
                  viewDimension: "2d" as GPUTextureViewDimension,
                  multisampled: false,
                },
              },
              {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                storageTexture: {
                  access: "write-only" as GPUStorageTextureAccess,
                  format: "rgba32float" as GPUTextureFormat,
                  viewDimension: "2d-array" as GPUTextureViewDimension,
                },
              },
            ],
          });
        }

        const computePipelineLayout = device.createPipelineLayout({
          label: "GPU Compute Pipeline Layout: Cubemap Generation",
          bindGroupLayouts: [bindGroupLayout],
        });
        const computePipeline = device.createComputePipeline({
          label: "GPU Compute Pipeline: Cubemap Generation",
          layout: computePipelineLayout,
          compute: {
            module: modules.get(type),
            entryPoint: "compute_main",
          },
        });
        computePipelines.set(type, computePipeline);
      }

      let dstTexture: GPUTexture;

      const workgroupsNum = Math.floor((size + 15) / 16);

      const encoder = device.createCommandEncoder({
        label: "GPU Command Encoder: Cubemap Generation",
      });

      if (type !== CubemapType.PREFILTER) {
        const srcTexture = device.createTexture({
          label: "GPU Texture: Cubemap Generation Source",
          size: [hdr.width, hdr.height],
          format: "rgba32float",
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        device.queue.writeTexture(
          {
            texture: srcTexture,
            mipLevel: 0,
            origin: [0, 0, 0],
          },
          hdr.data,
          {
            offset: 0,
            bytesPerRow: hdr.width * 4 * Float32Array.BYTES_PER_ELEMENT,
            rowsPerImage: hdr.height,
          },
          {
            width: hdr.width,
            height: hdr.height,
          }
        );
        const srcTextureView = srcTexture.createView({
          label: "GPU Texture View: Cubemap Generation Source",
          format: "rgba32float",
          dimension: "2d",
        });

        dstTexture = device.createTexture({
          label: "GPU Texture: Cubemap Generation Destination",
          size: [size, size, 6],
          format: "rgba32float",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_SRC,
        });
        const dstTextureView = dstTexture.createView({
          label: "GPU Texture View: Cubemap Generation Destination",
          format: "rgba32float",
          dimension: "2d-array",
        });

        const bindGroup = device.createBindGroup({
          label: "GPU Bind Group: Cubemap Generation",
          layout: computePipelines.get(type).getBindGroupLayout(0),
          entries: [
            {
              binding: 0,
              resource: srcTextureView,
            },
            {
              binding: 1,
              resource: dstTextureView,
            },
          ],
        });

        const pass = encoder.beginComputePass({
          label: "GPU Compute Pass: Cubemap Generation",
        });
        pass.setPipeline(computePipelines.get(type));
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(workgroupsNum, workgroupsNum, 6);
        pass.end();
      } else {
        const srcTexture = device.createTexture({
          label: "GPU Texture: Cubemap Generation Source",
          size: [hdr.width, hdr.height],
          mipLevelCount: calculateMipLevelCount(hdr.width, hdr.height),
          format: "rgba32float",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
        });
        device.queue.writeTexture(
          {
            texture: srcTexture,
            mipLevel: 0,
            origin: [0, 0, 0],
          },
          hdr.data,
          {
            offset: 0,
            bytesPerRow: hdr.width * 4 * Float32Array.BYTES_PER_ELEMENT,
            rowsPerImage: hdr.height,
          },
          {
            width: hdr.width,
            height: hdr.height,
          }
        );
        context.generateMipmaps()(device, srcTexture);

        const srcTextureView = srcTexture.createView({
          label: "GPU Texture View: Cubemap Generation Source",
          format: "rgba32float",
          dimension: "2d",
        });

        const maxMipLevels = 5;

        dstTexture = device.createTexture({
          label: "GPU Texture: Cubemap Generation Destination",
          size: [size, size, 6],
          mipLevelCount: maxMipLevels,
          format: "rgba32float",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_SRC,
        });

        for (let mip = 0; mip < maxMipLevels; ++mip) {
          const roughness = new Float32Array([mip / (maxMipLevels - 1)]);
          const roughnessUniformBuffer = context.createBuffer(
            `GPU Uniform Buffer: Cubemap Generation Roughness`,
            4,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
          );
          device.queue.writeBuffer(
            roughnessUniformBuffer,
            0,
            roughness.buffer,
            roughness.byteOffset,
            roughness.byteLength
          );

          const dstTextureView = dstTexture.createView({
            label: "GPU Texture View: Cubemap Generation Destination",
            format: "rgba32float",
            dimension: "2d-array",
            baseMipLevel: mip,
            mipLevelCount: 1,
          });

          const bindGroup = device.createBindGroup({
            label: "GPU Bind Group: Cubemap Generation",
            layout: computePipelines.get(type).getBindGroupLayout(0),
            entries: [
              {
                binding: 0,
                resource: {
                  buffer: roughnessUniformBuffer,
                },
              },
              {
                binding: 1,
                resource: srcTextureView,
              },
              {
                binding: 2,
                resource: dstTextureView,
              },
            ],
          });

          const pass = encoder.beginComputePass({
            label: "GPU Compute Pass: Cubemap Generation",
          });
          pass.setPipeline(computePipelines.get(type));
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(workgroupsNum, workgroupsNum, 6);
          pass.end();
        }
      }

      device.queue.submit([encoder.finish()]);

      return dstTexture;
    };
  }

  private generateBRDFLUT(): (d: GPUDevice, s: number) => GPUTexture {
    let module: GPUShaderModule;
    let computePipeline: GPUComputePipeline;

    const context = this;

    return function (device: GPUDevice, size: number) {
      if (!module) {
        module = device.createShaderModule({
          label: `GPU Shader Module: BRDF Integration`,
          code: brdfShaderCode,
        });
      }

      if (!computePipeline) {
        const bindGroupLayout = device.createBindGroupLayout({
          label: "GPU Bind Group Layout: BRDF Integration",
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: "uniform" as GPUBufferBindingType },
            },
            {
              binding: 1,
              visibility: GPUShaderStage.COMPUTE,
              storageTexture: {
                access: "write-only" as GPUStorageTextureAccess,
                format: "rgba16float" as GPUTextureFormat,
                viewDimension: "2d" as GPUTextureViewDimension,
              },
            },
          ],
        });
        const computePipelineLayout = device.createPipelineLayout({
          label: "GPU Compute Pipeline Layout: BRDF Integration",
          bindGroupLayouts: [bindGroupLayout],
        });
        computePipeline = device.createComputePipeline({
          label: "GPU Compute Pipeline: BRDF Integration",
          layout: computePipelineLayout,
          compute: {
            module,
            entryPoint: "compute_main",
          },
        });
      }

      const workgroupsNum = Math.floor((size + 15) / 16);

      const sizeValue = new Float32Array([size]);
      const sizeUniformBuffer = context.createBuffer(
        `GPU Uniform Buffer: Cubemap Generation Roughness`,
        4,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      );
      device.queue.writeBuffer(
        sizeUniformBuffer,
        0,
        sizeValue.buffer,
        sizeValue.byteOffset,
        sizeValue.byteLength
      );

      let brdfLUT = device.createTexture({
        label: "GPU Texture: BRDF Integration",
        size: [size, size],
        mipLevelCount: 1,
        format: "rgba16float",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.COPY_SRC,
      });

      const brdfLUTView = brdfLUT.createView({
        label: "GPU Texture View: BRDF Integration",
        format: "rgba16float",
        dimension: "2d",
      });

      const bindGroup = device.createBindGroup({
        label: "GPU Bind Group: BRDF Integration",
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: sizeUniformBuffer,
            },
          },
          {
            binding: 1,
            resource: brdfLUTView,
          },
        ],
      });

      const encoder = device.createCommandEncoder({
        label: "GPU Command Encoder: BRDF Integration",
      });

      const pass = encoder.beginComputePass({
        label: "GPU Compute Pass: BRDF Integration",
      });
      pass.setPipeline(computePipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroupsNum, workgroupsNum);
      pass.end();

      device.queue.submit([encoder.finish()]);

      return brdfLUT;
    };
  }

  private initGUI(): void {
    const gui = new GUI({
      name: "My GUI",
      autoPlace: true,
      hideable: true,
      width: 300,
    });

    if (window.location.hash !== "#debug") {
      gui.hide()
    }

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
      skybox: "Pure Sky",
    };

    const skyboxGUI = gui.addFolder("Skybox");
    skyboxGUI.closed = false;
    this.skyboxController = skyboxGUI
      .add(skyboxOptions, "skybox")
      .options(["Pure Sky", "Clear Pure Sky"])
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

    // light positions
    const lightPositions = new Float32Array(
      this.lightPositionsUniformBuffer.size / Float32Array.BYTES_PER_ELEMENT
    );

    lightPositions.set(vec3.create(10.0, 10.0, 10.0), 0);
    lightPositions.set(vec3.create(-10.0, 10.0, 10.0), 4);
    lightPositions.set(vec3.create(10.0, -10.0, 10.0), 8);
    lightPositions.set(vec3.create(-10.0, -10.0, 10.0), 12);

    this.device.queue.writeBuffer(
      this.lightPositionsUniformBuffer,
      0,
      lightPositions.buffer,
      lightPositions.byteOffset,
      lightPositions.byteLength
    );

    // light colors
    const lightColors = new Float32Array(
      this.lightColorsUniformBuffer.size / Float32Array.BYTES_PER_ELEMENT
    );

    lightColors.set(vec3.create(300.0, 300.0, 300.0), 0);
    lightColors.set(vec3.create(300.0, 300.0, 300.0), 4);
    lightColors.set(vec3.create(300.0, 300.0, 300.0), 8);
    lightColors.set(vec3.create(300.0, 300.0, 300.0), 12);

    this.device.queue.writeBuffer(
      this.lightColorsUniformBuffer,
      0,
      lightColors.buffer,
      lightColors.byteOffset,
      lightColors.byteLength
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
    const skyboxIndex = this.skyboxController.getValue() === "Pure Sky" ? 0 : 1;

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
